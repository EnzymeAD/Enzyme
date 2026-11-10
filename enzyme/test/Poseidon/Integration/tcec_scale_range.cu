// The TCEC runtime's power-of-two operand scaling on the device (built-in
// FP16TCEC backend):
//   grid:     single-entry 16 x 16 products 1.5 * 2^eA times 1.25 * 2^eB
//             through __poseidon_tcec_dgemm, over operand maxima inside and
//             outside FP32's range. Every step of the product is exact, so the
//             result is 1.875 * 2^(eA + eB) bit for bit wherever that is
//             representable in FP64.
//   tiny:     2^-66 A times 2^-100 B (entries around 1e-20 and 1e-30) gives
//             2^-166 times the product of A and B bit for bit, and so the
//             accuracy against the FP64 product that ordinary operands get.
//   ordinary: random operands reduced to digests of the result bits, through
//             the square entry (beta 0, and alpha 0.75 with beta -1.25, which
//             takes the epilogue's fused multiply-add) and the general entry
//             (A row-major, B and C column-major, beta 1). The digests are the
//             runtime's from before its scales were carried as exponents; the
//             'record' argument prints them for the runtime linked in.
//
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 %s -x none \
// RUN:     %poseidon_rt_lib -L/usr/local/cuda/lib64 -lcublas -lcudart \
// RUN:     -lstdc++ -lm -o %t.exe
// RUN: %t.exe | FileCheck %s
//
// REQUIRES: poseidon, cuda-runtime

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

extern "C" void __poseidon_tcec_dgemm(double *C, const double *A,
                                      const double *B, int N, int lda, int ldb,
                                      int ldc, int transA, int transB,
                                      double alpha, double beta,
                                      cudaStream_t stream, int mode);
extern "C" void __poseidon_tcec_dgemm_ex(double *C, const double *A,
                                         const double *B, int M, int Ncols,
                                         int K, int lda, int ldb, int ldc,
                                         int aColMajor, int bColMajor,
                                         int cColMajor, double alpha,
                                         double beta, cudaStream_t stream,
                                         int mode);

#define CK(x)                                                                  \
  do {                                                                         \
    if ((x) != cudaSuccess) {                                                  \
      printf("runtime error %s:%d %s\n", __FILE__, __LINE__, #x);              \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

namespace {

uint64_t bitsOf(double d) {
  uint64_t b;
  memcpy(&b, &d, 8);
  return b;
}

uint64_t g_state = 0x9e3779b97f4a7c15ull;
double uniform() { // [0, 1), 53 bits
  g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
  return (double)(g_state >> 11) * 0x1.0p-53;
}

// Element (r, c) a random significand scaled by a power of two per row and
// per column, stored row-major, or column-major when colMajor.
std::vector<double> operand(int rows, int cols, int colMajor) {
  std::vector<double> v((size_t)rows * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      int e = (r * 7) % 9 - 4 + (c * 5) % 9 - 4;
      v[colMajor ? (size_t)c * rows + r : (size_t)r * cols + c] =
          std::ldexp(uniform() - 0.5, e);
    }
  return v;
}

uint64_t digest(const std::vector<double> &v) {
  uint64_t h = 14695981039346656037ull;
  for (double d : v) {
    uint64_t b = bitsOf(d);
    for (int i = 0; i < 8; ++i) {
      h ^= (b >> (8 * i)) & 0xff;
      h *= 1099511628211ull;
    }
  }
  return h;
}

struct Device {
  double *A = nullptr, *B = nullptr, *C = nullptr;
  explicit Device(size_t n) {
    CK(cudaMalloc(&A, n * 8));
    CK(cudaMalloc(&B, n * 8));
    CK(cudaMalloc(&C, n * 8));
  }
  void put(double *d, const std::vector<double> &h) {
    CK(cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice));
  }
  std::vector<double> getC(size_t n) {
    CK(cudaDeviceSynchronize());
    std::vector<double> h(n);
    CK(cudaMemcpy(h.data(), C, n * 8, cudaMemcpyDeviceToHost));
    return h;
  }
};

int grid() {
  const int N = 16;
  Device d(N * N);
  std::vector<double> A(N * N, 0.0), B(N * N, 0.0), C0(N * N, 0.0);
  std::vector<std::pair<int, int>> cases;
  for (int eA = -152; eA <= 128; ++eA)
    for (int eB : {-120, -60, -10, 0, 10, 60, 120})
      cases.push_back({eA, eB});
  for (int eA : {-1073, -1000, -700, -400, -200, 200, 400, 700, 1000, 1023})
    for (int eB : {-1072, -1000, -700, -400, -200, -120, 0, 120, 200, 400, 700,
                   1000, 1023})
      cases.push_back({eA, eB});
  long representable = 0, exact = 0, others = 0, rounded = 0;
  int shown = 0;
  for (auto [eA, eB] : cases) {
    A[0] = std::ldexp(1.5, eA);
    B[0] = std::ldexp(1.25, eB);
    d.put(d.A, A);
    d.put(d.B, B);
    d.put(d.C, C0);
    __poseidon_tcec_dgemm(d.C, d.A, d.B, N, N, N, N, 0, 0, 1.0, 0.0, nullptr,
                          1);
    const double got = d.getC(N * N)[0];
    const double want = std::ldexp(1.875, eA + eB);
    // 1.875 * 2^e has four significant bits, the last at 2^(e - 3).
    const bool isRep = eA + eB >= -1074 + 3 && eA + eB <= 1023;
    const bool same = bitsOf(got) == bitsOf(want);
    if (isRep) {
      ++representable;
      exact += same;
      if (!same && shown++ < 8)
        printf("grid eA=%d eB=%d got=%a want=%a\n", eA, eB, got, want);
    } else {
      ++others;
      rounded += same;
    }
  }
  printf("grid %zu cases: representable %ld, exact %ld; not representable "
         "%ld, rounded once %ld\n",
         cases.size(), representable, exact, others, rounded);
  return exact != representable || rounded != others;
}

// Largest |C - R| / (|A||B|) of a row-major N x N product, R in FP64.
double error(const std::vector<double> &A, const std::vector<double> &B,
             const std::vector<double> &C, int N) {
  double e = 0.0;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c) {
      double ref = 0.0, mag = 0.0;
      for (int k = 0; k < N; ++k) {
        ref += A[r * N + k] * B[k * N + c];
        mag += std::fabs(A[r * N + k] * B[k * N + c]);
      }
      e = std::fmax(e, std::fabs(C[r * N + c] - ref) / mag);
    }
  return e;
}

int tiny() {
  const int N = 64;
  Device d(N * N);
  const std::vector<double> A = operand(N, N, 0), B = operand(N, N, 0);
  std::vector<double> At(A.size()), Bt(B.size());
  for (size_t i = 0; i < A.size(); ++i) {
    At[i] = std::ldexp(A[i], -66);
    Bt[i] = std::ldexp(B[i], -100);
  }
  d.put(d.A, A);
  d.put(d.B, B);
  __poseidon_tcec_dgemm(d.C, d.A, d.B, N, N, N, N, 0, 0, 1.0, 0.0, nullptr, 1);
  const std::vector<double> C = d.getC(N * N);
  d.put(d.A, At);
  d.put(d.B, Bt);
  __poseidon_tcec_dgemm(d.C, d.A, d.B, N, N, N, N, 0, 0, 1.0, 0.0, nullptr, 1);
  const std::vector<double> Ct = d.getC(N * N);
  long same = 0;
  double maxA = 0.0, maxB = 0.0;
  for (size_t i = 0; i < C.size(); ++i) {
    same += bitsOf(Ct[i]) == bitsOf(std::ldexp(C[i], -166));
    maxA = std::fmax(maxA, std::fabs(At[i]));
    maxB = std::fmax(maxB, std::fabs(Bt[i]));
  }
  const double e = error(A, B, C, N), et = error(At, Bt, Ct, N);
  const bool ok = same == (long)C.size() && et == e;
  printf("tiny max|A| %.1e max|B| %.1e: 2^-166 x ordinary %ld of %zu, error "
         "against FP64 %s ordinary's (%.3e)\n",
         maxA, maxB, same, C.size(), et == e ? "equal to" : "NOT equal to", et);
  return !ok;
}

struct OrdinaryCase {
  const char *name;
  bool square;
  int M, Ncols, K, aColMajor, bColMajor, cColMajor;
  double alpha, beta;
  uint64_t bits;
};

const OrdinaryCase kOrdinary[] = {
    {"square256", true, 256, 256, 256, 0, 0, 0, 1.0, 0.0, 0x41df4190c638113dull},
    {"square256-alpha-beta", true, 256, 256, 256, 0, 0, 0, 0.75, -1.25,
     0xc129590242f94997ull},
    {"ex343x192x1536", false, 343, 192, 1536, 0, 1, 1, 1.0, 1.0,
     0x622da3a8cbee232full},
};

int ordinary(bool record) {
  int failed = 0;
  for (const OrdinaryCase &oc : kOrdinary) {
    g_state = 0x9e3779b97f4a7c15ull;
    const std::vector<double> A = operand(oc.M, oc.K, oc.aColMajor);
    const std::vector<double> B = operand(oc.K, oc.Ncols, oc.bColMajor);
    const std::vector<double> C0 = operand(oc.M, oc.Ncols, oc.cColMajor);
    Device d(std::max(A.size(), std::max(B.size(), C0.size())));
    d.put(d.A, A);
    d.put(d.B, B);
    d.put(d.C, C0);
    if (oc.square)
      __poseidon_tcec_dgemm(d.C, d.A, d.B, oc.M, oc.M, oc.M, oc.M, 0, 0,
                            oc.alpha, oc.beta, nullptr, 1);
    else
      __poseidon_tcec_dgemm_ex(
          d.C, d.A, d.B, oc.M, oc.Ncols, oc.K, oc.aColMajor ? oc.M : oc.K,
          oc.bColMajor ? oc.K : oc.Ncols, oc.cColMajor ? oc.M : oc.Ncols,
          oc.aColMajor, oc.bColMajor, oc.cColMajor, oc.alpha, oc.beta, nullptr,
          1);
    const uint64_t h = digest(d.getC(C0.size()));
    if (record) {
      printf("%s 0x%016llxull\n", oc.name, (unsigned long long)h);
      continue;
    }
    const bool same = h == oc.bits;
    printf("ordinary %s digest=%016llx %s\n", oc.name, (unsigned long long)h,
           same ? "BITS-MATCH" : "BITS-DIFFER");
    failed |= !same;
  }
  return failed;
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "record"))
    return ordinary(true);
  int failed = grid();
  failed |= tiny();
  failed |= ordinary(false);
  printf(failed ? "FAIL\n" : "PASS\n");
  return failed;
}

// CHECK: grid 2097 cases: representable 2054, exact 2054; not representable 43, rounded once 43
// CHECK: tiny max|A| {{[0-9.]+e-[0-9]+}} max|B| {{[0-9.]+e-[0-9]+}}: 2^-166 x ordinary 4096 of 4096, error against FP64 equal to ordinary's
// CHECK: ordinary square256 digest={{[0-9a-f]+}} BITS-MATCH
// CHECK: ordinary square256-alpha-beta digest={{[0-9a-f]+}} BITS-MATCH
// CHECK: ordinary ex343x192x1536 digest={{[0-9a-f]+}} BITS-MATCH
// CHECK: PASS

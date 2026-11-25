// The GEMM dispatch runtimes against a reference DGEMM of the same library, on
// random operands whose rows and columns span 2^-8..2^8, compiled as CUDA
// (cuBLAS) or as HIP (hipBLAS). Driven by Integration/dispatch_rt_{cuda,hip}.
//
//   ozaki:  every Ozaki-II case is also reduced to a digest of its result
//           bits. The products are exact INT32 sums, the shifts come from the
//           same round-up sums of squares and directed roundings, and every
//           other step is a correctly rounded FP64 operation, so a correct
//           runtime on any device returns the bits the CUDA runtime returns on
//           an RTX 5090, whose digests are below, unless the device's log2 in
//           a shift lands on the other side of an integer. The error against
//           the reference is held to twice what that run measured.
//   direct: each reduced-precision mode is held to the a priori bound of a
//           product of operands rounded once (unit roundoff u) accumulated in
//           FP32: (2u + u^2) + gamma_K (1 + u)^2 relative to |A||B|. The
//           native DGEMM (mode 0 here) is held to gamma_K in FP64. On cuBLAS,
//           the FP32 mode (mode 4) is also held to bit exactness against
//           cublasSgemm of the operands narrowed to FP32 with the FP64 alpha
//           and beta applied after, for a general call and square row-major
//           ones, and differs from TF32 (mode 3). The TF32 mode is held to bit
//           exactness against cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 of
//           the operands rounded to TF32 as the in-kernel raise rounds them
//           (cvt.rna.tf32.f32), and the run reports which rounding the
//           library itself gives FP32 operands and whether its TF32 product
//           differs from its FP32 one on the same operands. The square
//           1100^3 product has every extent above 1024: a cuMpSGEMM linked
//           ahead of cuBLAS takes over cublasSgemm and cublasGemmEx and,
//           with CUMPSGEMM_COMPUTE_MODE unset, stops the program there.
//   empty:  every entry with K == 0 gives C = beta * C (beta = 0 writes zeros
//           over NaN), touching nothing outside the M x Ncols view, and with
//           M == 0 touches nothing.
//   streams: Ozaki-II and direct calls interleaved on two non-blocking
//           streams return the bits each returns alone.
//   negative: a negative extent stops the program with a message.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "GemmDispatch.h"
#ifndef __HIP__
#include <cublas_v2.h>
#endif

#define CK(x)                                                                  \
  do {                                                                         \
    if ((x) != cudaSuccess) {                                                  \
      printf("runtime error %s:%d %s\n", __FILE__, __LINE__, #x);              \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)
#define CB(x)                                                                  \
  do {                                                                         \
    if ((x) != CUBLAS_STATUS_SUCCESS) {                                        \
      printf("BLAS error %s:%d %s\n", __FILE__, __LINE__, #x);                 \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

namespace {

struct Shape {
  const char *name;
  int M, Ncols, K, aColMajor, bColMajor, cColMajor;
  double beta;
};

// Square N = 256, row-major; the elasticity reduce's layout (A row-major, B
// and C column-major, accumulating into C) at a skinny shape; a nearly square
// shape with the opposite layouts.
const Shape kShapes[] = {
    {"square256", 256, 256, 256, 0, 0, 0, 0.0},
    {"rect343x192x1536", 343, 192, 1536, 0, 1, 1, 1.0},
    {"pad300", 300, 300, 300, 1, 0, 0, 0.0},
};

uint64_t g_state = 0x9e3779b97f4a7c15ull;
double uniform() { // [0, 1), 53 bits
  g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
  return (double)(g_state >> 11) * 0x1.0p-53;
}

// Element (r, c) of a rows x cols operand: a random significand scaled by a
// power of two per row and per column, so that every row and column of an
// Ozaki operand gets its own shift.
std::vector<double> operand(int rows, int cols, int colMajor) {
  std::vector<double> v((size_t)rows * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      int e = (r * 7) % 9 - 4 + (c * 5) % 9 - 4;
      double x = std::ldexp(uniform() - 0.5, e);
      v[colMajor ? (size_t)c * rows + r : (size_t)r * cols + c] = x;
    }
  return v;
}

double at(const std::vector<double> &v, int rows, int cols, int colMajor,
          int r, int c) {
  return v[colMajor ? (size_t)c * rows + r : (size_t)r * cols + c];
}

uint64_t digest(const std::vector<double> &v) {
  uint64_t h = 14695981039346656037ull;
  for (double d : v) {
    uint64_t b;
    std::memcpy(&b, &d, 8);
    for (int i = 0; i < 8; ++i) {
      h ^= (b >> (8 * i)) & 0xff;
      h *= 1099511628211ull;
    }
  }
  return h;
}

struct Problem {
  const Shape &s;
  std::vector<double> A, B, C0;
  double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dR = nullptr;
  std::vector<double> absAB; // (|A||B|)(r, c), in C's layout
  int lda, ldb, ldc;

  explicit Problem(const Shape &sh) : s(sh) {
    A = operand(s.M, s.K, s.aColMajor);
    B = operand(s.K, s.Ncols, s.bColMajor);
    C0 = operand(s.M, s.Ncols, s.cColMajor);
    lda = s.aColMajor ? s.M : s.K;
    ldb = s.bColMajor ? s.K : s.Ncols;
    ldc = s.cColMajor ? s.M : s.Ncols;
    absAB.assign((size_t)s.M * s.Ncols, 0.0);
    for (int r = 0; r < s.M; ++r)
      for (int c = 0; c < s.Ncols; ++c) {
        double t = 0.0;
        for (int k = 0; k < s.K; ++k)
          t += std::fabs(at(A, s.M, s.K, s.aColMajor, r, k)) *
               std::fabs(at(B, s.K, s.Ncols, s.bColMajor, k, c));
        t += std::fabs(s.beta * at(C0, s.M, s.Ncols, s.cColMajor, r, c));
        absAB[s.cColMajor ? (size_t)c * s.M + r : (size_t)r * s.Ncols + c] = t;
      }
    CK(cudaMalloc(&dA, A.size() * 8));
    CK(cudaMalloc(&dB, B.size() * 8));
    CK(cudaMalloc(&dC, C0.size() * 8));
    CK(cudaMalloc(&dR, C0.size() * 8));
    CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice));
    reference();
  }

  void reset(double *d) {
    CK(cudaMemcpy(d, C0.data(), C0.size() * 8, cudaMemcpyHostToDevice));
  }

  // The library's DGEMM, column-major: a row-major C is computed as C^T.
  void reference() {
    static cublasHandle_t h = nullptr;
    if (!h)
      CB(cublasCreate(&h));
    reset(dR);
    const double one = 1.0;
    cublasOperation_t opA = s.aColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = s.bColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
    if (s.cColMajor)
      CB(cublasDgemm(h, opA, opB, s.M, s.Ncols, s.K, &one, dA, lda, dB, ldb,
                     &s.beta, dR, ldc));
    else
      CB(cublasDgemm(h, s.bColMajor ? CUBLAS_OP_T : CUBLAS_OP_N,
                     s.aColMajor ? CUBLAS_OP_T : CUBLAS_OP_N, s.Ncols, s.M,
                     s.K, &one, dB, ldb, dA, lda, &s.beta, dR, ldc));
    CK(cudaDeviceSynchronize());
  }

  // Largest |C - R| / (|A||B| + |beta C|) over the result.
  double error(std::vector<double> &out) {
    std::vector<double> ref(C0.size());
    out.resize(C0.size());
    CK(cudaMemcpy(out.data(), dC, out.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), dR, ref.size() * 8, cudaMemcpyDeviceToHost));
    double e = 0.0;
    for (size_t i = 0; i < out.size(); ++i)
      e = std::fmax(e, std::fabs(out[i] - ref[i]) / absAB[i]);
    return e;
  }
};

struct OzakiCase {
  int shape, nm;
  uint64_t bits; // the CUDA runtime's digest on an RTX 5090
  double err;    // and its error against cublasDgemm
};

// Recorded by the CUDA build on an RTX 5090 (dispatch_rt ozaki-record).
const OzakiCase kOzaki[] = {
    {0, 8, 0xe612b0f4774dce33ull, 7.536001e-09},
    {0, 10, 0x2bba9479fcc93da1ull, 2.888180e-11},
    {0, 13, 0xb9fbfdd828c1e983ull, 1.176637e-14},
    {0, 14, 0x30d6d2b2ae96b2b1ull, 1.059351e-15},
    {1, 10, 0x2f15c4f6b41b18c4ull, 3.294043e-11},
    {1, 13, 0x89d47b41e6ea9d69ull, 9.882169e-15},
    {2, 10, 0x841e6a1178e31414ull, 2.552330e-11},
    {2, 13, 0xd9ca278be0ec57e8ull, 1.124379e-14},
};

int ozaki(bool record) {
  int failed = 0;
  for (int si = 0; si < 3; ++si) {
    const Shape &s = kShapes[si];
    Problem p(s);
    for (const OzakiCase &oc : kOzaki) {
      if (oc.shape != si)
        continue;
      p.reset(p.dC);
      __poseidon_ozaki_dgemm(p.dC, p.dA, p.dB, s.M, s.Ncols, s.K, p.lda, p.ldb,
                             p.ldc, s.aColMajor, s.bColMajor, s.cColMajor, 1.0,
                             s.beta, nullptr, oc.nm);
      CK(cudaDeviceSynchronize());
      std::vector<double> out;
      double e = p.error(out);
      uint64_t d = digest(out);
      if (record) {
        printf("    {%d, %d, 0x%016llxull, %.6e},\n", oc.shape, oc.nm,
               (unsigned long long)d, e);
        continue;
      }
      bool bits = d == oc.bits, close = e <= 2 * oc.err;
      printf("ozaki %s nm=%d digest=%016llx %s err=%.3e %s\n", s.name, oc.nm,
             (unsigned long long)d, bits ? "BITS-MATCH" : "BITS-DIFFER", e,
             close ? "ok" : "too large");
      failed |= !bits || !close;
    }
  }
  return failed;
}

// Unit roundoff of each mode's operand format (FP64 for mode 0).
double inputRoundoff(int mode) {
  switch (mode) {
  case 0:
    return 0x1.0p-53;
  case 1:
    return 0x1.0p-11;
  case 2:
    return 0x1.0p-8;
  default:
    return 0x1.0p-24;
  }
}

#ifndef __HIP__
enum Tf32Rounding { kNoTf32, kTf32Nearest, kTf32NearestEven, kTf32Truncate };

// FP64 -> FP32, then to TF32 by `tf32`: to nearest with ties away from zero
// (cvt.rna.tf32.f32), to nearest with ties to even, or by dropping the low 13
// significand bits.
__global__ void narrow(float *dst, const double *src, size_t n, int tf32) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float f = (float)src[i];
  if (tf32 == kTf32Nearest) {
    unsigned b;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(b) : "f"(f));
    f = __uint_as_float(b);
  } else if (tf32 == kTf32NearestEven && isfinite(f)) {
    const unsigned b = __float_as_uint(f);
    f = __uint_as_float((b + 0xfffu + ((b >> 13) & 1u)) & 0xffffe000u);
  } else if (tf32 == kTf32Truncate) {
    f = __uint_as_float(__float_as_uint(f) & 0xffffe000u);
  }
  dst[i] = f;
}

__global__ void widen(double *C, const float *src, size_t n, double alpha,
                      double beta) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    C[i] = alpha * (double)src[i] + (beta == 0.0 ? 0.0 : beta * C[i]);
}

// cublasSgemm of row-major operands narrowed to FP32, in the operand order a
// row-major C needs ((A@B)^T = B^T@A^T), then the FP64 alpha and beta. With
// `fastTf32`, cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 of the operands
// narrowed with `tf32`.
void sgemmReference(double *C, const double *A, const double *B, int M,
                    int Ncols, int K, double alpha, double beta,
                    bool fastTf32 = false, int tf32 = kNoTf32) {
  static cublasHandle_t h = nullptr;
  if (!h)
    CB(cublasCreate(&h));
  float *Af, *Bf, *Cf;
  CK(cudaMalloc(&Af, (size_t)M * K * sizeof(float)));
  CK(cudaMalloc(&Bf, (size_t)K * Ncols * sizeof(float)));
  CK(cudaMalloc(&Cf, (size_t)M * Ncols * sizeof(float)));
  narrow<<<((size_t)M * K + 255) / 256, 256>>>(Af, A, (size_t)M * K, tf32);
  narrow<<<((size_t)K * Ncols + 255) / 256, 256>>>(Bf, B, (size_t)K * Ncols,
                                                   tf32);
  const float one = 1.0f, zero = 0.0f;
  if (fastTf32)
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, M, K, &one, Bf,
                    CUDA_R_32F, Ncols, Af, CUDA_R_32F, K, &zero, Cf, CUDA_R_32F,
                    Ncols, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
  else
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, M, K, &one, Bf, Ncols,
                   Af, K, &zero, Cf, Ncols));
  widen<<<((size_t)M * Ncols + 255) / 256, 256>>>(C, Cf, (size_t)M * Ncols,
                                                  alpha, beta);
  CK(cudaDeviceSynchronize());
  CK(cudaFree(Af));
  CK(cudaFree(Bf));
  CK(cudaFree(Cf));
}

std::vector<double> filled(size_t n, unsigned seed) {
  std::vector<double> v(n);
  srand(seed);
  for (double &x : v)
    x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  return v;
}

// Mode 4 on row-major operands, a general call with alpha 2.5 and beta 0.5
// and square ones, against sgemmReference bit for bit, and the square ones
// against mode 3.
int sgemm() {
  struct {
    const char *name;
    bool square;
    int M, Ncols, K;
    double alpha, beta;
  } const cases[] = {{"ex97x64x129", false, 97, 64, 129, 2.5, 0.5},
                     {"square128", true, 128, 128, 128, 1.0, 0.0},
                     {"square1100", true, 1100, 1100, 1100, 1.0, 0.0}};
  int failed = 0;
  unsigned seed = 1;
  for (const auto &c : cases) {
    const std::vector<double> A = filled((size_t)c.M * c.K, seed++),
                              B = filled((size_t)c.K * c.Ncols, seed++),
                              C0 = filled((size_t)c.M * c.Ncols, seed++);
    double *dA, *dB, *dC, *dR;
    CK(cudaMalloc(&dA, A.size() * 8));
    CK(cudaMalloc(&dB, B.size() * 8));
    CK(cudaMalloc(&dC, C0.size() * 8));
    CK(cudaMalloc(&dR, C0.size() * 8));
    CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC, C0.data(), C0.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dR, C0.data(), C0.size() * 8, cudaMemcpyHostToDevice));
    if (c.square)
      __poseidon_direct_dgemm(dC, dA, dB, c.M, c.M, c.M, c.M, c.M, c.M, 0, 0, 0,
                              c.alpha, c.beta, nullptr, 4);
    else
      __poseidon_direct_dgemm(dC, dA, dB, c.M, c.Ncols, c.K, c.K, c.Ncols,
                              c.Ncols, 0, 0, 0, c.alpha, c.beta, nullptr, 4);
    CK(cudaDeviceSynchronize());
    sgemmReference(dR, dA, dB, c.M, c.Ncols, c.K, c.alpha, c.beta);
    std::vector<double> out(C0.size()), ref(C0.size());
    CK(cudaMemcpy(out.data(), dC, out.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), dR, ref.size() * 8, cudaMemcpyDeviceToHost));
    const bool exact = !std::memcmp(out.data(), ref.data(), out.size() * 8);
    printf("sgemm %s mode=4 %s\n", c.name, exact ? "BITEXACT" : "MISMATCH");
    failed |= !exact;
    if (c.square) {
      // TF32 keeps 10 explicit significand bits, so mode 3 cannot agree.
      __poseidon_direct_dgemm(dR, dA, dB, c.M, c.M, c.M, c.M, c.M, c.M, 0, 0, 0,
                              c.alpha, c.beta, nullptr, 3);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(ref.data(), dR, ref.size() * 8, cudaMemcpyDeviceToHost));
      const bool same = !std::memcmp(out.data(), ref.data(), out.size() * 8);
      printf("sgemm %s mode=4 vs mode=3 %s\n", c.name,
             same ? "SAME" : "DIFFERENT");
      failed |= same;
      // Mode 3 against the library's TF32 product of operands rounded as the
      // raise rounds them, and the library's own rounding of FP32 operands.
      std::vector<double> rna(C0.size()), rne(C0.size()), raw(C0.size()),
          cut(C0.size()), simt(C0.size());
      auto tf32Product = [&](std::vector<double> &host, bool tf32Core,
                             int rounding) {
        CK(cudaMemcpy(dR, C0.data(), C0.size() * 8, cudaMemcpyHostToDevice));
        sgemmReference(dR, dA, dB, c.M, c.M, c.M, c.alpha, c.beta, tf32Core,
                       rounding);
        CK(cudaMemcpy(host.data(), dR, host.size() * 8,
                      cudaMemcpyDeviceToHost));
      };
      tf32Product(rna, true, kTf32Nearest);
      tf32Product(raw, true, kNoTf32);
      tf32Product(rne, true, kTf32NearestEven);
      tf32Product(cut, true, kTf32Truncate);
      tf32Product(simt, false, kTf32Nearest);
      auto equal = [](const std::vector<double> &x,
                      const std::vector<double> &y) {
        return !std::memcmp(x.data(), y.data(), x.size() * 8);
      };
      const bool exact3 = equal(ref, rna);
      printf("sgemm %s mode=3 %s\n", c.name, exact3 ? "BITEXACT" : "MISMATCH");
      failed |= !exact3;
      const bool away = equal(raw, rna), even = equal(raw, rne);
      printf("cublas %s FAST_TF32 on FP32 operands: %s\n", c.name,
             away && even      ? "rounds to nearest (no tie decided)"
             : away            ? "rounds to nearest, ties away"
             : even            ? "rounds to nearest, ties to even"
             : equal(raw, cut) ? "truncates"
                               : "neither");
      printf("cublas %s FAST_TF32 vs FP32 product of the same TF32 operands: "
             "%s\n",
             c.name, equal(rna, simt) ? "SAME" : "DIFFERENT");
    }
    CK(cudaFree(dA));
    CK(cudaFree(dB));
    CK(cudaFree(dC));
    CK(cudaFree(dR));
  }
  return failed;
}
#endif

int direct() {
  int failed = 0;
#ifndef __HIP__
  failed |= sgemm();
#endif
  for (const Shape &s : kShapes) {
    Problem p(s);
    for (int mode : {0, 1, 2, 4}) {
      p.reset(p.dC);
      if (mode == 0)
        __poseidon_native_dgemm(p.dC, p.dA, p.dB, s.M, s.Ncols, s.K, p.lda,
                                p.ldb, p.ldc, s.aColMajor, s.bColMajor,
                                s.cColMajor, 1.0, s.beta, nullptr);
      else
        __poseidon_direct_dgemm(p.dC, p.dA, p.dB, s.M, s.Ncols, s.K, p.lda,
                                p.ldb, p.ldc, s.aColMajor, s.bColMajor,
                                s.cColMajor, 1.0, s.beta, nullptr, mode);
      CK(cudaDeviceSynchronize());
      std::vector<double> out;
      double e = p.error(out);
      const double u = inputRoundoff(mode);
      const double uacc = mode == 0 ? 0x1.0p-53 : 0x1.0p-24;
      const double gamma = s.K * uacc / (1 - s.K * uacc);
      // The reference DGEMM's own error is below gamma_K in FP64.
      const double bound = (mode ? 2 * u + u * u : 0.0) +
                           gamma * (1 + u) * (1 + u) +
                           s.K * 0x1.0p-53 / (1 - s.K * 0x1.0p-53);
      bool ok = e <= bound;
      printf("direct %s mode=%d err=%.3e bound=%.3e %s\n", s.name, mode, e,
             bound, ok ? "ok" : "too large");
      failed |= !ok;
    }
  }
  return failed;
}

// K == 0 with beta 0 (over NaN), 1 and 0.5, and M == 0, for every entry this
// build has, on a column-major 37 x 29 view of a 40-row buffer.
int empty() {
  const int M = 37, Ncols = 29, ldc = 40;
  const size_t n = (size_t)ldc * Ncols;
  std::vector<double> C0(n);
  for (size_t i = 0; i < n; ++i)
    C0[i] = std::ldexp(uniform() - 0.5, (int)(i % 7));
  double *dC;
  CK(cudaMalloc(&dC, n * 8));
  struct Entry {
    const char *name;
    int mode;
  };
  const Entry entries[] = {{"native", -1}, {"ozaki", 8},
                           {"direct", 1},  {"direct", 2},
#ifndef __HIP__
                           {"direct", 3},
#endif
                           {"direct", 4}};
  auto call = [&](const Entry &e, int m, int k, double beta) {
    // A and B are never read: their extents are empty.
    if (e.mode < 0)
      __poseidon_native_dgemm(dC, nullptr, nullptr, m, Ncols, k, 1, 1, ldc, 1,
                              1, 1, 1.0, beta, nullptr);
    else if (!strcmp(e.name, "ozaki"))
      __poseidon_ozaki_dgemm(dC, nullptr, nullptr, m, Ncols, k, 1, 1, ldc, 1, 1,
                             1, 1.0, beta, nullptr, e.mode);
    else
      __poseidon_direct_dgemm(dC, nullptr, nullptr, m, Ncols, k, 1, 1, ldc, 1,
                              1, 1, 1.0, beta, nullptr, e.mode);
    CK(cudaDeviceSynchronize());
  };
  int failed = 0;
  for (const Entry &e : entries)
    for (double beta : {0.0, 1.0, 0.5}) {
      std::vector<double> in = C0, out(n);
      if (beta == 0.0)
        for (int c = 0; c < Ncols; ++c)
          for (int r = 0; r < M; ++r)
            in[(size_t)c * ldc + r] = NAN;
      CK(cudaMemcpy(dC, in.data(), n * 8, cudaMemcpyHostToDevice));
      call(e, M, 0, beta);
      CK(cudaMemcpy(out.data(), dC, n * 8, cudaMemcpyDeviceToHost));
      bool ok = true;
      for (int c = 0; c < Ncols; ++c)
        for (int r = 0; r < ldc; ++r) {
          const size_t i = (size_t)c * ldc + r;
          const double want =
              r < M ? (beta == 0.0 ? 0.0 : beta * in[i]) : in[i];
          ok &= !std::memcmp(&out[i], &want, 8);
        }
      printf("empty %s mode=%d K=0 beta=%g %s\n", e.name, e.mode, beta,
             ok ? "ok" : "WRONG");
      failed |= !ok;
    }
  for (const Entry &e : entries) {
    std::vector<double> out(n);
    CK(cudaMemcpy(dC, C0.data(), n * 8, cudaMemcpyHostToDevice));
    call(e, 0, 5, 0.0);
    CK(cudaMemcpy(out.data(), dC, n * 8, cudaMemcpyDeviceToHost));
    const bool ok = !std::memcmp(out.data(), C0.data(), n * 8);
    printf("empty %s mode=%d M=0 %s\n", e.name, e.mode, ok ? "ok" : "WRONG");
    failed |= !ok;
  }
  CK(cudaFree(dC));
  return failed;
}

// Ozaki-II nm = 10 on shapes 0 and 2, and direct FP16 on shape 1, issued in
// turn on two non-blocking streams that share each runtime's scratch.
int streams() {
  Problem p0(kShapes[0]), p1(kShapes[1]), p2(kShapes[2]);
  auto direct1 = [&](cudaStream_t st) {
    __poseidon_direct_dgemm(p1.dC, p1.dA, p1.dB, p1.s.M, p1.s.Ncols, p1.s.K,
                            p1.lda, p1.ldb, p1.ldc, p1.s.aColMajor,
                            p1.s.bColMajor, p1.s.cColMajor, 1.0, 0.0, st, 1);
  };
  auto ozaki10 = [](Problem &p, cudaStream_t st) {
    __poseidon_ozaki_dgemm(p.dC, p.dA, p.dB, p.s.M, p.s.Ncols, p.s.K, p.lda,
                           p.ldb, p.ldc, p.s.aColMajor, p.s.bColMajor,
                           p.s.cColMajor, 1.0, p.s.beta, st, 10);
  };
  // Alone first: shape 1's direct result (beta = 0 here) is the reference.
  direct1(nullptr);
  CK(cudaDeviceSynchronize());
  std::vector<double> alone(p1.C0.size());
  CK(cudaMemcpy(alone.data(), p1.dC, alone.size() * 8, cudaMemcpyDeviceToHost));

  cudaStream_t s1, s2;
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  for (int rep = 0; rep < 8; ++rep) {
    ozaki10(rep % 2 ? p2 : p0, s1);
    ozaki10(rep % 2 ? p0 : p2, s2);
    direct1(rep % 2 ? s1 : s2);
  }
  CK(cudaStreamSynchronize(s1));
  CK(cudaStreamSynchronize(s2));
  int failed = 0;
  std::vector<double> out;
  for (auto [p, want] : {std::pair<Problem *, uint64_t>{&p0, kOzaki[1].bits},
                         std::pair<Problem *, uint64_t>{&p2, kOzaki[6].bits}}) {
    p->error(out);
    const bool ok = digest(out) == want;
    printf("streams ozaki %s nm=10 %s\n", p->s.name,
           ok ? "BITS-MATCH" : "BITS-DIFFER");
    failed |= !ok;
  }
  p1.error(out);
  const bool ok = !std::memcmp(out.data(), alone.data(), out.size() * 8);
  printf("streams direct %s mode=1 %s\n", p1.s.name,
         ok ? "BITS-MATCH" : "BITS-DIFFER");
  failed |= !ok;
  CK(cudaStreamDestroy(s1));
  CK(cudaStreamDestroy(s2));
  return failed;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: %s ozaki|ozaki-record|direct|empty|streams|negative\n",
           argv[0]);
    return 2;
  }
  int failed = 0;
  if (!strcmp(argv[1], "ozaki"))
    failed = ozaki(false);
  else if (!strcmp(argv[1], "ozaki-record"))
    failed = ozaki(true);
  else if (!strcmp(argv[1], "direct"))
    failed = direct();
  else if (!strcmp(argv[1], "empty"))
    failed = empty();
  else if (!strcmp(argv[1], "streams"))
    failed = streams();
  else if (!strcmp(argv[1], "negative"))
    __poseidon_ozaki_dgemm(nullptr, nullptr, nullptr, 16, 16, -1, 16, 16, 16, 0,
                           0, 0, 1.0, 0.0, nullptr, 10);
  else
    return 2;
  printf(failed ? "FAIL\n" : "PASS\n");
  return failed;
}

// OZAKI: ozaki square256 nm=8 digest=e612b0f4774dce33 BITS-MATCH err={{.*}} ok
// OZAKI-NEXT: ozaki square256 nm=10 digest=2bba9479fcc93da1 BITS-MATCH err={{.*}} ok
// OZAKI-NEXT: ozaki square256 nm=13 digest=b9fbfdd828c1e983 BITS-MATCH err={{.*}} ok
// OZAKI-NEXT: ozaki square256 nm=14 digest=30d6d2b2ae96b2b1 BITS-MATCH err={{.*}} ok
// OZAKI-NEXT: ozaki rect343x192x1536 nm=10 digest=2f15c4f6b41b18c4 BITS-MATCH err={{.*}} ok
// OZAKI-NEXT: ozaki rect343x192x1536 nm=13 digest=89d47b41e6ea9d69 BITS-MATCH err={{.*}} ok
// OZAKI-NEXT: ozaki pad300 nm=10 digest=841e6a1178e31414 BITS-MATCH err={{.*}} ok
// OZAKI-NEXT: ozaki pad300 nm=13 digest=d9ca278be0ec57e8 BITS-MATCH err={{.*}} ok
// OZAKI-NEXT: PASS

// SGEMM: sgemm ex97x64x129 mode=4 BITEXACT
// SGEMM-NEXT: sgemm square128 mode=4 BITEXACT
// SGEMM-NEXT: sgemm square128 mode=4 vs mode=3 DIFFERENT
// SGEMM-NEXT: sgemm square128 mode=3 BITEXACT
// SGEMM-NEXT: cublas square128 FAST_TF32 on FP32 operands: {{.*}}
// SGEMM-NEXT: cublas square128 FAST_TF32 vs FP32 product of the same TF32 operands: {{SAME|DIFFERENT}}
// SGEMM-NEXT: sgemm square1100 mode=4 BITEXACT
// SGEMM-NEXT: sgemm square1100 mode=4 vs mode=3 DIFFERENT
// SGEMM-NEXT: sgemm square1100 mode=3 BITEXACT
// SGEMM-NEXT: cublas square1100 FAST_TF32 on FP32 operands: {{.*}}
// SGEMM-NEXT: cublas square1100 FAST_TF32 vs FP32 product of the same TF32 operands: {{SAME|DIFFERENT}}

// DIRECT-COUNT-12: direct {{.*}} ok
// DIRECT-NEXT: PASS

// EMPTY: empty native mode=-1 K=0 beta=0 ok
// EMPTY-NEXT: empty native mode=-1 K=0 beta=1 ok
// EMPTY-NEXT: empty native mode=-1 K=0 beta=0.5 ok
// EMPTY-NEXT: empty ozaki mode=8 K=0 beta=0 ok
// EMPTY-NEXT: empty ozaki mode=8 K=0 beta=1 ok
// EMPTY-NEXT: empty ozaki mode=8 K=0 beta=0.5 ok
// EMPTY-NEXT: empty direct mode=1 K=0 beta=0 ok
// EMPTY-NEXT: empty direct mode=1 K=0 beta=1 ok
// EMPTY-NEXT: empty direct mode=1 K=0 beta=0.5 ok
// EMPTY-NEXT: empty direct mode=2 K=0 beta=0 ok
// EMPTY-NEXT: empty direct mode=2 K=0 beta=1 ok
// EMPTY-NEXT: empty direct mode=2 K=0 beta=0.5 ok
// EMPTY-CUDA-NEXT: empty direct mode=3 K=0 beta=0 ok
// EMPTY-CUDA-NEXT: empty direct mode=3 K=0 beta=1 ok
// EMPTY-CUDA-NEXT: empty direct mode=3 K=0 beta=0.5 ok
// EMPTY-NEXT: empty direct mode=4 K=0 beta=0 ok
// EMPTY-NEXT: empty direct mode=4 K=0 beta=1 ok
// EMPTY-NEXT: empty direct mode=4 K=0 beta=0.5 ok
// EMPTY-NEXT: empty native mode=-1 M=0 ok
// EMPTY-NEXT: empty ozaki mode=8 M=0 ok
// EMPTY-NEXT: empty direct mode=1 M=0 ok
// EMPTY-NEXT: empty direct mode=2 M=0 ok
// EMPTY-CUDA-NEXT: empty direct mode=3 M=0 ok
// EMPTY-NEXT: empty direct mode=4 M=0 ok
// EMPTY-NEXT: PASS

// STREAMS: streams ozaki square256 nm=10 BITS-MATCH
// STREAMS-NEXT: streams ozaki pad300 nm=10 BITS-MATCH
// STREAMS-NEXT: streams direct rect343x192x1536 mode=1 BITS-MATCH
// STREAMS-NEXT: PASS

// NEGATIVE: Poseidon GEMM runtime: __poseidon_ozaki_dgemm with M = 16, Ncols = 16, K = -1: an extent is negative

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
//           FP32: (2u + u^2) + gamma_K (1 + u)^2 relative to |A||B|. DGEMM
//           (mode 0) is held to gamma_K in FP64.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef __HIP__
#include "hip_as_cuda.h"
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int N, int lda, int ldb,
                                       int ldc, int transA, int transB,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli);
extern "C" void __poseidon_ozaki_dgemm_ex(double *C, const double *A,
                                          const double *B, int M, int Ncols,
                                          int K, int lda, int ldb, int ldc,
                                          int aColMajor, int bColMajor,
                                          int cColMajor, double alpha,
                                          double beta, cudaStream_t stream,
                                          int num_moduli);
extern "C" void __poseidon_direct_dgemm(double *C, const double *A,
                                        const double *B, int N, int lda,
                                        int ldb, int ldc, int transA,
                                        int transB, double alpha, double beta,
                                        cudaStream_t stream, int mode);
extern "C" void __poseidon_direct_dgemm_ex(double *C, const double *A,
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
  bool square; // the square entry: row-major, leading dimension N
  int M, Ncols, K, aColMajor, bColMajor, cColMajor;
  double beta;
};

// Square N = 256; the elasticity reduce's layout (A row-major, B and C
// column-major, accumulating into C) at a skinny shape, which takes the
// rectangular arm; a nearly square shape with the opposite layouts, which takes
// the square arm with padding (S = 304).
const Shape kShapes[] = {
    {"square256", true, 256, 256, 256, 0, 0, 0, 0.0},
    {"rect343x192x1536", false, 343, 192, 1536, 0, 1, 1, 1.0},
    {"pad300", false, 300, 300, 300, 1, 0, 0, 0.0},
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
      if (s.square)
        __poseidon_ozaki_dgemm(p.dC, p.dA, p.dB, s.M, s.M, s.M, s.M, 0, 0, 1.0,
                               s.beta, nullptr, oc.nm);
      else
        __poseidon_ozaki_dgemm_ex(p.dC, p.dA, p.dB, s.M, s.Ncols, s.K, p.lda,
                                  p.ldb, p.ldc, s.aColMajor, s.bColMajor,
                                  s.cColMajor, 1.0, s.beta, nullptr, oc.nm);
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

int direct() {
  int failed = 0;
  for (const Shape &s : kShapes) {
    Problem p(s);
    for (int mode : {0, 1, 2, 4}) {
      p.reset(p.dC);
      if (s.square)
        __poseidon_direct_dgemm(p.dC, p.dA, p.dB, s.M, s.M, s.M, s.M, 0, 0, 1.0,
                                s.beta, nullptr, mode);
      else
        __poseidon_direct_dgemm_ex(p.dC, p.dA, p.dB, s.M, s.Ncols, s.K, p.lda,
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

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: %s ozaki|ozaki-record|direct\n", argv[0]);
    return 2;
  }
  int failed = 0;
  if (!strcmp(argv[1], "ozaki"))
    failed = ozaki(false);
  else if (!strcmp(argv[1], "ozaki-record"))
    failed = ozaki(true);
  else if (!strcmp(argv[1], "direct"))
    failed = direct();
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

// DIRECT-COUNT-12: direct {{.*}} ok
// DIRECT-NEXT: PASS

// The library call runtimes (cuBLAS or hipBLAS build) against the library's
// DGEMM, on random operands spanning 2^-8..2^8 (the direct cases' A 2^-24..2^8),
// for Integration/gemm_runtime_*.
//   ozaki:    each Ozaki-II case's result bits digest to what an RTX 5090
//             returns (every step is exact or correctly rounded), and its
//             error is held to twice what that run measured.
//   direct:   each reduced-precision mode is held to the a priori bound of
//             operands rounded once, a to a(1 + d) + e with |d| <= u and |e|
//             <= eta, and accumulated in FP32: (2u + u^2 + gamma_K (1 + u)^2)
//             |A||B| plus (1 + gamma_K) (eta (1 + u) (|A| 1 + 1^T |B|) + K
//             eta^2); the native mode to gamma_K. eta is half the format's
//             smallest subnormal where the device's product keeps every
//             subnormal operand, and the smallest operand it keeps where it
//             reads smaller ones as zero: the floor poseidon-calibrate-library
//             measures on the device, in the file the case is given. Rows of A
//             wholly in FP16's subnormals fail a library that reads as zero an
//             operand at or above that floor. On cuBLAS, FP32 (mode 4) and
//             TF32 (mode 3) are bit exact against cublasSgemm and FAST_TF32
//             gpublasGemmEx of the rounded operands, and differ from each
//             other.
//   empty:    K == 0 gives C = beta * C, touching nothing outside the view;
//             M == 0 touches nothing.
//   streams:  calls interleaved on two streams return the bits each returns alone.
//   negative: a negative extent stops the program with a message.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "DeviceApi.h"
#include "gemm/LibraryCalls.h"

#define CK(x)                                                                  \
  do {                                                                         \
    if ((x) != gpuSuccess) {                                                  \
      printf("runtime error %s:%d %s\n", __FILE__, __LINE__, #x);              \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)
#define CB(x)                                                                  \
  do {                                                                         \
    if ((x) != GPUBLAS_STATUS_SUCCESS) {                                        \
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

// Square row-major; A row-major with B and C column-major accumulating into C;
// a nearly square shape with the opposite layouts.
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
// power of two per row, in [rowLow, rowLow + rowSpan), and per column, in
// [-4, 4], so that every row and column of an Ozaki operand gets its own
// shift.
std::vector<double> operand(int rows, int cols, int colMajor, int rowLow = -4,
                            int rowSpan = 9) {
  std::vector<double> v((size_t)rows * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      int e = (r * 7) % rowSpan + rowLow + (c * 5) % 9 - 4;
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
  // The row sums of |A| and the column sums of |B|.
  std::vector<double> rowA, colB;
  int lda, ldb, ldc;

  // The rows of A span 2^aRowLow..2^(aRowLow + aRowSpan - 1) before their
  // column's scale.
  explicit Problem(const Shape &sh, int aRowLow = -4, int aRowSpan = 9)
      : s(sh) {
    A = operand(s.M, s.K, s.aColMajor, aRowLow, aRowSpan);
    B = operand(s.K, s.Ncols, s.bColMajor);
    C0 = operand(s.M, s.Ncols, s.cColMajor);
    lda = s.aColMajor ? s.M : s.K;
    ldb = s.bColMajor ? s.K : s.Ncols;
    ldc = s.cColMajor ? s.M : s.Ncols;
    absAB.assign((size_t)s.M * s.Ncols, 0.0);
    rowA.assign(s.M, 0.0);
    colB.assign(s.Ncols, 0.0);
    for (int r = 0; r < s.M; ++r)
      for (int k = 0; k < s.K; ++k)
        rowA[r] += std::fabs(at(A, s.M, s.K, s.aColMajor, r, k));
    for (int c = 0; c < s.Ncols; ++c)
      for (int k = 0; k < s.K; ++k)
        colB[c] += std::fabs(at(B, s.K, s.Ncols, s.bColMajor, k, c));
    for (int r = 0; r < s.M; ++r)
      for (int c = 0; c < s.Ncols; ++c) {
        double t = 0.0;
        for (int k = 0; k < s.K; ++k)
          t += std::fabs(at(A, s.M, s.K, s.aColMajor, r, k)) *
               std::fabs(at(B, s.K, s.Ncols, s.bColMajor, k, c));
        t += std::fabs(s.beta * at(C0, s.M, s.Ncols, s.cColMajor, r, c));
        absAB[s.cColMajor ? (size_t)c * s.M + r : (size_t)r * s.Ncols + c] = t;
      }
    CK(gpuMalloc(&dA, A.size() * 8));
    CK(gpuMalloc(&dB, B.size() * 8));
    CK(gpuMalloc(&dC, C0.size() * 8));
    CK(gpuMalloc(&dR, C0.size() * 8));
    CK(gpuMemcpy(dA, A.data(), A.size() * 8, gpuMemcpyHostToDevice));
    CK(gpuMemcpy(dB, B.data(), B.size() * 8, gpuMemcpyHostToDevice));
    reference();
  }

  void reset(double *d) {
    CK(gpuMemcpy(d, C0.data(), C0.size() * 8, gpuMemcpyHostToDevice));
  }

  // The library's DGEMM, column-major: a row-major C is computed as C^T.
  void reference() {
    static gpublasHandle_t h = nullptr;
    if (!h)
      CB(gpublasCreate(&h));
    reset(dR);
    const double one = 1.0;
    gpublasOperation_t opA = s.aColMajor ? GPUBLAS_OP_N : GPUBLAS_OP_T;
    gpublasOperation_t opB = s.bColMajor ? GPUBLAS_OP_N : GPUBLAS_OP_T;
    if (s.cColMajor)
      CB(gpublasDgemm(h, opA, opB, s.M, s.Ncols, s.K, &one, dA, lda, dB, ldb,
                     &s.beta, dR, ldc));
    else
      CB(gpublasDgemm(h, s.bColMajor ? GPUBLAS_OP_T : GPUBLAS_OP_N,
                     s.aColMajor ? GPUBLAS_OP_T : GPUBLAS_OP_N, s.Ncols, s.M,
                     s.K, &one, dB, ldb, dA, lda, &s.beta, dR, ldc));
    CK(gpuDeviceSynchronize());
  }

  // Largest |C - R| / allowed(r, c, |A||B| + |beta C|) over the result.
  template <class Allowed>
  double error(std::vector<double> &out, Allowed allowed) {
    std::vector<double> ref(C0.size());
    out.resize(C0.size());
    CK(gpuMemcpy(out.data(), dC, out.size() * 8, gpuMemcpyDeviceToHost));
    CK(gpuMemcpy(ref.data(), dR, ref.size() * 8, gpuMemcpyDeviceToHost));
    double e = 0.0;
    for (int r = 0; r < s.M; ++r)
      for (int c = 0; c < s.Ncols; ++c) {
        size_t i = s.cColMajor ? (size_t)c * s.M + r : (size_t)r * s.Ncols + c;
        e = std::fmax(e, std::fabs(out[i] - ref[i]) / allowed(r, c, absAB[i]));
      }
    return e;
  }
  // Largest |C - R| / (|A||B| + |beta C|) over the result.
  double error(std::vector<double> &out) {
    return error(out, [](int, int, double absAB) { return absAB; });
  }
};

struct OzakiCase {
  int shape, nm;
  uint64_t bits; // the CUDA runtime's digest on an RTX 5090
  double err;    // and its error against gpublasDgemm
};

// Recorded by the CUDA build on an RTX 5090 (gemm_runtime ozaki-record).
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
      __poseidon_ozakiii_dgemm(p.dC, p.dA, p.dB, s.M, s.Ncols, s.K, p.lda, p.ldb,
                             p.ldc, s.aColMajor, s.bColMajor, s.cColMajor, 1.0,
                             s.beta, nullptr, oc.nm);
      CK(gpuDeviceSynchronize());
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

// The smallest subnormal of each mode's operand format (none for mode 0, whose
// operands are not rounded).
double smallestSubnormal(int mode) {
  switch (mode) {
  case 0:
    return 0.0;
  case 1:
    return 0x1.0p-24;
  case 2:
    return 0x1.0p-133;
  default:
    return 0x1.0p-149;
  }
}

// The value of the row library_direct_operand_min,`cls` in `path`, the rows
// poseidon-calibrate-library's operand-min family prints: the smallest operand
// magnitude the class's product keeps on this device.
double operandFloor(const char *path, const char *cls) {
  FILE *f = fopen(path, "r");
  if (!f) {
    printf("cannot read the operand floors in %s\n", path);
    exit(2);
  }
  char key[64], line[256];
  int n = snprintf(key, sizeof key, "library_direct_operand_min,%s,", cls);
  double floor = 0.0;
  while (fgets(line, sizeof line, f))
    if (!strncmp(line, key, n))
      floor = strtod(line + n, nullptr);
  fclose(f);
  if (!(floor > 0.0)) {
    printf("%s has no row %s\n", path, key);
    exit(2);
  }
  return floor;
}

// The largest error of reading an operand too small for a relative one, in a
// mode whose product keeps no operand below `floor`: `floor` when that is
// above the format's smallest subnormal, so that operands below it are read as
// zero, else the half subnormal of rounding to nearest.
double inputUnderflow(int mode, double floor) {
  const double subnormal = smallestSubnormal(mode);
  return floor > subnormal ? floor : subnormal / 2;
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

// `floors` is the file of the device's operand floors.
int direct(const char *floors) {
  int failed = 0;
  const double floor16 = operandFloor(floors, "f16_f32");
  const double floorBf16 = operandFloor(floors, "bf16_f32");
#ifndef __HIP__
  failed |= sgemm();
#endif
  for (const Shape &s : kShapes) {
    Problem p(s, -20, 25);
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
      CK(gpuDeviceSynchronize());
      const double floor = mode == 1 ? floor16 : mode == 2 ? floorBf16 : 0.0;
      const double u = inputRoundoff(mode), eta = inputUnderflow(mode, floor);
      const double uacc = mode == 0 ? 0x1.0p-53 : 0x1.0p-24;
      const double gamma = s.K * uacc / (1 - s.K * uacc);
      // The reference DGEMM's own error is below gamma_K in FP64.
      const double bound = (mode ? 2 * u + u * u : 0.0) +
                           gamma * (1 + u) * (1 + u) +
                           s.K * 0x1.0p-53 / (1 - s.K * 0x1.0p-53);
      std::vector<double> out;
      double e = p.error(out, [&](int r, int c, double absAB) {
        return bound * absAB +
               (1 + gamma) *
                   (eta * (1 + u) * (p.rowA[r] + p.colB[c]) + s.K * eta * eta);
      });
      bool ok = e <= 1.0;
      printf("direct %s mode=%d floor=%.3e err/bound=%.3e %s\n", s.name, mode,
             floor, e, ok ? "ok" : "too large");
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
  CK(gpuMalloc(&dC, n * 8));
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
      __poseidon_ozakiii_dgemm(dC, nullptr, nullptr, m, Ncols, k, 1, 1, ldc, 1, 1,
                             1, 1.0, beta, nullptr, e.mode);
    else
      __poseidon_direct_dgemm(dC, nullptr, nullptr, m, Ncols, k, 1, 1, ldc, 1,
                              1, 1, 1.0, beta, nullptr, e.mode);
    CK(gpuDeviceSynchronize());
  };
  int failed = 0;
  for (const Entry &e : entries)
    for (double beta : {0.0, 1.0, 0.5}) {
      std::vector<double> in = C0, out(n);
      if (beta == 0.0)
        for (int c = 0; c < Ncols; ++c)
          for (int r = 0; r < M; ++r)
            in[(size_t)c * ldc + r] = NAN;
      CK(gpuMemcpy(dC, in.data(), n * 8, gpuMemcpyHostToDevice));
      call(e, M, 0, beta);
      CK(gpuMemcpy(out.data(), dC, n * 8, gpuMemcpyDeviceToHost));
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
    CK(gpuMemcpy(dC, C0.data(), n * 8, gpuMemcpyHostToDevice));
    call(e, 0, 5, 0.0);
    CK(gpuMemcpy(out.data(), dC, n * 8, gpuMemcpyDeviceToHost));
    const bool ok = !std::memcmp(out.data(), C0.data(), n * 8);
    printf("empty %s mode=%d M=0 %s\n", e.name, e.mode, ok ? "ok" : "WRONG");
    failed |= !ok;
  }
  CK(gpuFree(dC));
  return failed;
}

// Ozaki-II nm = 10 on shapes 0 and 2, and direct FP16 on shape 1, issued in
// turn on two non-blocking streams that share each runtime's scratch.
int streams() {
  Problem p0(kShapes[0]), p1(kShapes[1]), p2(kShapes[2]);
  auto direct1 = [&](gpuStream_t st) {
    __poseidon_direct_dgemm(p1.dC, p1.dA, p1.dB, p1.s.M, p1.s.Ncols, p1.s.K,
                            p1.lda, p1.ldb, p1.ldc, p1.s.aColMajor,
                            p1.s.bColMajor, p1.s.cColMajor, 1.0, 0.0, st, 1);
  };
  auto ozaki10 = [](Problem &p, gpuStream_t st) {
    __poseidon_ozakiii_dgemm(p.dC, p.dA, p.dB, p.s.M, p.s.Ncols, p.s.K, p.lda,
                           p.ldb, p.ldc, p.s.aColMajor, p.s.bColMajor,
                           p.s.cColMajor, 1.0, p.s.beta, st, 10);
  };
  // Alone first: shape 1's direct result (beta = 0 here) is the reference.
  direct1(nullptr);
  CK(gpuDeviceSynchronize());
  std::vector<double> alone(p1.C0.size());
  CK(gpuMemcpy(alone.data(), p1.dC, alone.size() * 8, gpuMemcpyDeviceToHost));

  gpuStream_t s1, s2;
  CK(gpuStreamCreateWithFlags(&s1, gpuStreamNonBlocking));
  CK(gpuStreamCreateWithFlags(&s2, gpuStreamNonBlocking));
  for (int rep = 0; rep < 8; ++rep) {
    ozaki10(rep % 2 ? p2 : p0, s1);
    ozaki10(rep % 2 ? p0 : p2, s2);
    direct1(rep % 2 ? s1 : s2);
  }
  CK(gpuStreamSynchronize(s1));
  CK(gpuStreamSynchronize(s2));
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
  CK(gpuStreamDestroy(s1));
  CK(gpuStreamDestroy(s2));
  return failed;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: %s ozaki|ozaki-record|direct <floors>|empty|streams|"
           "negative\n",
           argv[0]);
    return 2;
  }
  int failed = 0;
  if (!strcmp(argv[1], "ozaki"))
    failed = ozaki(false);
  else if (!strcmp(argv[1], "ozaki-record"))
    failed = ozaki(true);
  else if (!strcmp(argv[1], "direct")) {
    if (argc != 3) {
      printf("direct takes the file of the device's operand floors\n");
      return 2;
    }
    failed = direct(argv[2]);
  } else if (!strcmp(argv[1], "empty"))
    failed = empty();
  else if (!strcmp(argv[1], "streams"))
    failed = streams();
  else if (!strcmp(argv[1], "negative"))
    __poseidon_ozakiii_dgemm(nullptr, nullptr, nullptr, 16, 16, -1, 16, 16, 16, 0,
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

// NEGATIVE: [poseidon] __poseidon_ozakiii_dgemm with M = 16, Ncols = 16, K = -1: an extent is negative

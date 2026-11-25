//===- library.cu - price the GEMM host dispatches ------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The *_dispatch_rel rows of a cost model: the time of a host-dispatched GEMM
// over the time of the naive scalar-FP64 GEMM at a square reference shape, so
// that every dispatch family is priced in one unit:
//
//   poseidon-calibrate-library-cuda ozaki N reps   library_ozaki_rel,nm<8..14>
//                                              and library_ozaki_rel,dgemm
//   poseidon-calibrate-library-cuda ozaki N reps MxK|MxNxK...
//                                              library_ozaki_rect_rel rows:
//                                              MxK nm<k>_m<M>k<K>, MxNxK
//                                              nm<k>_m<M>n<N>k<K> and
//                                              dgemm_m<M>n<N>k<K>
//   poseidon-calibrate-library-cuda direct N reps library_direct_rel,<in>_<acc>
//   poseidon-calibrate-library-cuda tcec N reps    library_tcec_rel,fp16tcec
//
// The rows go to stdout. Built as HIP it measures hipBLAS, which has no TF32
// compute type, so tf32_f32 has no row there; tcec exists only in a build with
// cuMpSGEMM (POSEIDON_TCEC).
//
//===----------------------------------------------------------------------===//

#include "runtime/gemm/GemmRuntime.h"
#include "runtime/gemm/Schemes.h"

// The CUDA build defines it to whether cuMpSGEMM is linked.
#ifndef POSEIDON_TCEC
#define POSEIDON_TCEC 0
#endif

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/// The naive scalar-FP64 GEMM whose time is the denominator of every row.
__global__ void referenceGemm(double *C, const double *A, const double *B,
                              int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= N)
    return;
  double s = 0.0;
  for (int l = 0; l < N; ++l)
    s += A[(size_t)row * N + l] * B[(size_t)col * N + l];
  C[(size_t)col * N + row] = s;
}

namespace {

/// One measured call of an entry: its operands, extents and layouts. A
/// square product is row-major with leading dimensions N, as the pass
/// dispatches a square GEMM.
struct Gemm {
  double *C;
  const double *A, *B;
  int M, Ncols, K, lda, ldb, ldc, aColMajor, bColMajor, cColMajor;
  double beta;

  static Gemm square(double *C, const double *A, const double *B, int N) {
    return {C, A, B, N, N, N, N, N, N, 0, 0, 0, 0.0};
  }
  void native() const {
    __poseidon_native_dgemm(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor,
                            bColMajor, cColMajor, 1.0, beta, nullptr);
  }
  void ozaki(int nm) const {
    __poseidon_ozakiii_dgemm(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor,
                             bColMajor, cColMajor, 1.0, beta, nullptr, nm);
  }
  void direct(int mode) const {
    __poseidon_direct_dgemm(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor,
                            bColMajor, cColMajor, 1.0, beta, nullptr, mode);
  }
#if POSEIDON_TCEC
  void tcec(int mode) const {
    __poseidon_tcec_dgemm(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor,
                          bColMajor, cColMajor, 1.0, beta, nullptr, mode);
  }
#endif
};

/// The reference shape and the timed repetitions.
int N, Reps;
double *dA, *dB, *dC;

/// The mean time in milliseconds of `reps` calls of `fn`, after one warm-up.
template <class Fn> float timeit(Fn fn, int reps) {
  cudaEvent_t a, b;
  POSEIDON_GEMM_CK(cudaEventCreate(&a));
  POSEIDON_GEMM_CK(cudaEventCreate(&b));
  fn();
  POSEIDON_GEMM_CK(cudaDeviceSynchronize());
  POSEIDON_GEMM_CK(cudaEventRecord(a));
  for (int i = 0; i < reps; i++)
    fn();
  POSEIDON_GEMM_CK(cudaEventRecord(b));
  POSEIDON_GEMM_CK(cudaEventSynchronize(b));
  float ms;
  POSEIDON_GEMM_CK(cudaEventElapsedTime(&ms, a, b));
  POSEIDON_GEMM_CK(cudaEventDestroy(a));
  POSEIDON_GEMM_CK(cudaEventDestroy(b));
  return ms / reps;
}

/// `n` device doubles drawn uniformly from [-1, 1].
double *randomDevice(size_t n) {
  double *h = (double *)malloc(n * 8), *d;
  for (size_t i = 0; i < n; i++)
    h[i] = ((double)rand() / RAND_MAX) * 2 - 1;
  POSEIDON_GEMM_CK(cudaMalloc(&d, n * 8));
  POSEIDON_GEMM_CK(cudaMemcpy(d, h, n * 8, cudaMemcpyHostToDevice));
  free(h);
  return d;
}

/// The column count the (M, K) Ozaki-II rows are measured at. They stand for
/// a product whose column count is a run-time extent, on the assumption that
/// its cost is linear in the column count; the conversion of A is paid once
/// per call, so a call with few columns costs more per MAC than that, and a
/// product of constant extents is measured at them instead (fullRows).
const int kRectCols = 12288;

/// The time per MAC of `call` on a product with an M x K operand A and
/// `cols` columns, its operands laid out as the recognizer dispatches a
/// reduction: A row-major, B and C column-major, C accumulated.
template <class Call>
double rectPerMac(const double *A, int M, int K, int cols, Call call) {
  double *B = randomDevice((size_t)K * cols), *C;
  POSEIDON_GEMM_CK(cudaMalloc(&C, (size_t)M * cols * 8));
  POSEIDON_GEMM_CK(cudaMemset(C, 0, (size_t)M * cols * 8));
  const Gemm g{C,           A,         B,         M, cols, K,
               /*lda=*/K,   /*ldb=*/K, /*ldc=*/M, 0, 1,    1,
               /*beta=*/1.0};
  float t = timeit([&] { call(g); }, Reps);
  POSEIDON_GEMM_CK(cudaFree(B));
  POSEIDON_GEMM_CK(cudaFree(C));
  return (double)t / ((double)M * cols * K);
}

/// Prints the Ozaki-II rows of (M, K), one per modulus count.
void rectRows(int M, int K, float tRef) {
  const double refPerMac = (double)tRef / ((double)N * N * N);
  double *A = randomDevice((size_t)M * K);
  for (unsigned nm : kOzakiIIModuliCounts) {
    double rel =
        rectPerMac(A, M, K, kRectCols, [&](const Gemm &g) { g.ozaki(nm); }) /
        refPerMac;
    fprintf(stderr,
            "poseidon-calibrate-library: rect nm=%2u M=%d Ncols=%d K=%d  rel = "
            "%.6f\n",
            nm, M, kRectCols, K, rel);
    printf("library_ozaki_rect_rel,nm%u_m%dk%d,%.6f\n", nm, M, K, rel);
  }
  POSEIDON_GEMM_CK(cudaFree(A));
}

/// Prints the Ozaki-II rows, one per modulus count, and the native DGEMM row of
/// a call of exactly these extents: a call with few columns does not amortize
/// the conversion of A, and a DGEMM with few columns runs far below the square
/// rate.
void fullRows(int M, int Ncols, int K, float tRef) {
  const double refPerMac = (double)tRef / ((double)N * N * N);
  double *A = randomDevice((size_t)M * K);
  for (unsigned nm : kOzakiIIModuliCounts) {
    double rel =
        rectPerMac(A, M, K, Ncols, [&](const Gemm &g) { g.ozaki(nm); }) /
        refPerMac;
    fprintf(stderr,
            "poseidon-calibrate-library: rect nm=%2u M=%d Ncols=%d K=%d  rel = "
            "%.6f\n",
            nm, M, Ncols, K, rel);
    printf("library_ozaki_rect_rel,nm%u_m%dn%dk%d,%.6f\n", nm, M, Ncols, K,
           rel);
  }
  double rel =
      rectPerMac(A, M, K, Ncols, [](const Gemm &g) { g.native(); }) / refPerMac;
  fprintf(
      stderr,
      "poseidon-calibrate-library: rect dgemm M=%d Ncols=%d K=%d  rel = %.6f\n",
      M, Ncols, K, rel);
  printf("library_ozaki_rect_rel,dgemm_m%dn%dk%d,%.6f\n", M, Ncols, K, rel);
  POSEIDON_GEMM_CK(cudaFree(A));
}

/// Prints the row `family`,`key` at the time of `call` on the square product
/// over `tRef`.
template <class Call>
void row(const char *family, const char *key, Call call, float tRef) {
  const Gemm g = Gemm::square(dC, dA, dB, N);
  float t = timeit([&] { call(g); }, Reps);
  double rel = (double)t / (double)tRef;
  fprintf(stderr,
          "poseidon-calibrate-library: %s,%s  dispatch = %.4f ms  rel = %.6f "
          "(%.2fx vs FP64)\n",
          family, key, t, rel, tRef / t);
  printf("%s,%s,%.6f\n", family, key, rel);
}

/// The ozaki family: the square rows, or those of the shapes in argv[4...].
int ozaki(int argc, char **argv, float tRef) {
  if (argc > 4) {
    for (int i = 4; i < argc; ++i) {
      int M = 0, Ncols = 0, K = 0, end = 0;
      if (sscanf(argv[i], "%dx%dx%d%n", &M, &Ncols, &K, &end) == 3 &&
          !argv[i][end] && M > 0 && Ncols > 0 && K > 0) {
        fullRows(M, Ncols, K, tRef);
        continue;
      }
      if (sscanf(argv[i], "%dx%d%n", &M, &K, &end) != 2 || argv[i][end] ||
          M <= 0 || K <= 0) {
        fprintf(
            stderr,
            "poseidon-calibrate-library: '%s' is not a shape MxK or MxNxK\n",
            argv[i]);
        return 1;
      }
      rectRows(M, K, tRef);
    }
    return 0;
  }
  // Every modulus count the dispatch candidate generator can propose, measured
  // in one session so the rows are mutually consistent.
  char key[16];
  for (unsigned nm : kOzakiIIModuliCounts) {
    snprintf(key, sizeof(key), "nm%u", nm);
    row(
        "library_ozaki_rel", key, [&](const Gemm &g) { g.ozaki(nm); }, tRef);
  }
  // What the native DGEMM candidate runs; the warmup call amortizes the handle
  // creation.
  row(
      "library_ozaki_rel", "dgemm", [](const Gemm &g) { g.native(); }, tRef);
  return 0;
}

/// The direct family: one row per reduced-precision mode.
int direct(float tRef) {
  // The modes of __poseidon_direct_dgemm, in the class spelling of the rows.
  static const struct {
    int Mode;
    const char *Class;
  } Classes[] = {{kDirectFp16, "f16_f32"},
                 {kDirectBf16, "bf16_f32"},
#ifndef __HIP__
                 {kDirectTf32, "tf32_f32"},
#endif
                 {kDirectFp32, "f32_f32"}};
  for (const auto &C : Classes)
    row(
        "library_direct_rel", C.Class, [&](const Gemm &g) { g.direct(C.Mode); },
        tRef);
  return 0;
}

/// The positive integer `s` spells, or 0.
int positive(const char *s) {
  char *end;
  long v = strtol(s, &end, 10);
  return *s && !*end && v > 0 && v <= INT_MAX ? (int)v : 0;
}

} // namespace

int main(int argc, char **argv) {
#if POSEIDON_TCEC
  const char *families = "ozaki|direct|tcec";
#else
  const char *families = "ozaki|direct";
#endif
  const char *family = argc > 1 ? argv[1] : "";
  bool known = !strcmp(family, "ozaki") || !strcmp(family, "direct") ||
               (POSEIDON_TCEC && !strcmp(family, "tcec"));
  if (!known || argc < 4) {
    fprintf(stderr, "usage: %s %s N reps [MxK|MxNxK...]\n", argv[0], families);
    return 1;
  }
  N = positive(argv[2]);
  Reps = positive(argv[3]);
  if (!N || !Reps) {
    fprintf(stderr, "%s: N and reps are positive integers\n", argv[0]);
    return 1;
  }
  size_t sz = (size_t)N * N;
  srand(1);
  dA = randomDevice(sz);
  dB = randomDevice(sz);
  POSEIDON_GEMM_CK(cudaMalloc(&dC, sz * 8));

  float tRef = timeit(
      [] {
        const dim3 b(16, 16), g((N + 15) / 16, (N + 15) / 16);
        referenceGemm<<<g, b>>>(dC, dA, dB, N);
      },
      Reps);
  fprintf(
      stderr,
      "poseidon-calibrate-library: N=%d reps=%d  scalar-FP64 ref = %.4f ms\n",
      N, Reps, tRef);
  if (!strcmp(family, "ozaki"))
    return ozaki(argc, argv, tRef);
  if (!strcmp(family, "direct"))
    return direct(tRef);
#if POSEIDON_TCEC
  row(
      "library_tcec_rel", "fp16tcec",
      [](const Gemm &g) { g.tcec(kTcecFp16Tcec); }, tRef);
#endif
  return 0;
}

//===- dispatch_calibrate.cu - price the GEMM host dispatches -------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The *_dispatch_rel rows of a cost model: the time of a host-dispatched GEMM
// over the time of the naive scalar-FP64 GEMM at a square reference shape, so
// that every dispatch family is priced in one unit:
//
//   poseidon-dispatch-calibrate ozaki N reps   ozaki_dispatch_rel,nm<8..14>
//                                              and ozaki_dispatch_rel,dgemm
//   poseidon-dispatch-calibrate ozaki N reps MxK|MxNxK...
//                                              ozaki_dispatch_rect_rel rows
//   poseidon-dispatch-calibrate direct N reps  direct_dispatch_rel,<in>_<acc>
//   poseidon-dispatch-calibrate tcec N reps    tcec_dispatch_rel,fp16tcec
//
// The rows go to stdout. Built as HIP it measures hipBLAS, which has no TF32
// compute type, so tf32_f32 has no row there; tcec exists only in a build with
// cuMpSGEMM (POSEIDON_TCEC).
//
//===----------------------------------------------------------------------===//

#include "../../runtime/ozaki/OzakiII.h"

// The CUDA build defines it to whether cuMpSGEMM is linked.
#ifndef POSEIDON_TCEC
#define POSEIDON_TCEC 0
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __HIP__
#include "../../runtime/hip_as_cuda.h"
#else
#include <cuda_runtime.h>
#endif

// Square host dispatch (the entry a square GEMM is dispatched to).
extern "C" void __poseidon_ozaki_dgemm(double *C, const double *A,
                                       const double *B, int N, int lda, int ldb,
                                       int ldc, int transA, int transB,
                                       double alpha, double beta,
                                       cudaStream_t stream, int num_moduli);
// Rectangular host dispatch (the entry the host-GEMM recognizer emits).
extern "C" void __poseidon_ozaki_dgemm_ex(
    double *C, const double *A, const double *B, int M, int Ncols, int K,
    int lda, int ldb, int ldc, int aColMajor, int bColMajor, int cColMajor,
    double alpha, double beta, cudaStream_t stream, int num_moduli);
extern "C" void __poseidon_direct_dgemm(double *C, const double *A,
                                        const double *B, int N, int lda,
                                        int ldb, int ldc, int transA,
                                        int transB, double alpha, double beta,
                                        cudaStream_t stream, int mode);
#if POSEIDON_TCEC
extern "C" void __poseidon_tcec_dgemm(double *C, const double *A,
                                      const double *B, int N, int lda, int ldb,
                                      int ldc, int transA, int transB,
                                      double alpha, double beta,
                                      cudaStream_t stream, int mode);
#endif

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e) {                                                                   \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,                   \
              cudaGetErrorString(e));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// The naive scalar-FP64 GEMM, the cost model's FP64 baseline (per-MAC
// throughput of a triple-loop reduction): the denominator of every rel.
__global__ void ref_gemm(double *C, const double *A, const double *B, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= N)
    return;
  double s = 0.0;
  for (int l = 0; l < N; ++l)
    s += A[(size_t)row * N + l] * B[(size_t)col * N + l];
  C[(size_t)col * N + row] = s;
}

// The reference shape, the timed repetitions, and the modulus count or mode
// the dispatch under test runs with.
static int N = 2048, REPS = 30, ARG = 0;
static double *dA, *dB, *dC;

static void run_ref() {
  dim3 b(16, 16), g((N + 15) / 16, (N + 15) / 16);
  ref_gemm<<<g, b>>>(dC, dA, dB, N);
}
static void run_oz() {
  __poseidon_ozaki_dgemm(dC, dA, dB, N, N, N, N, 0, 0, 1.0, 0.0, 0, ARG);
}
static void run_dr() {
  __poseidon_direct_dgemm(dC, dA, dB, N, N, N, N, 0, 0, 1.0, 0.0, 0, ARG);
}
#if POSEIDON_TCEC
static void run_tc() {
  __poseidon_tcec_dgemm(dC, dA, dB, N, N, N, N, 0, 0, 1.0, 0.0, 0, 1);
}
#endif

static float timeit(void (*fn)(), int reps) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  fn();
  CK(cudaDeviceSynchronize()); // warmup
  CK(cudaEventRecord(a));
  for (int i = 0; i < reps; i++)
    fn();
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms;
  CK(cudaEventElapsedTime(&ms, a, b));
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return ms / reps;
}

static double *rand_dev(size_t n) {
  double *h = (double *)malloc(n * 8), *d;
  for (size_t i = 0; i < n; i++)
    h[i] = ((double)rand() / RAND_MAX) * 2 - 1;
  CK(cudaMalloc(&d, n * 8));
  CK(cudaMemcpy(d, h, n * 8, cudaMemcpyHostToDevice));
  free(h);
  return d;
}

// The rectangular arm of __poseidon_ozaki_dgemm_ex, which a host GEMM takes
// when padding it to a square would waste work. Its cost is linear in the
// column count, so its rows are keyed by (M, K) alone and measured at a large
// column count. The operands are laid out as the recognizer dispatches a
// reduce: A row-major M x K, B and C column-major, C accumulated.
static int RM, RN, RK;
static double *rA, *rB, *rC;

static void run_rect() {
  __poseidon_ozaki_dgemm_ex(rC, rA, rB, RM, RN, RK, /*lda=*/RK, /*ldb=*/RK,
                            /*ldc=*/RM, /*aColMajor=*/0, /*bColMajor=*/1,
                            /*cColMajor=*/1, 1.0, 1.0, 0, ARG);
}

// Dispatch time per MAC at `cols` columns and the current modulus count.
static double rect_per_mac(int cols) {
  RN = cols;
  rB = rand_dev((size_t)RK * cols);
  CK(cudaMalloc(&rC, (size_t)RM * cols * 8));
  CK(cudaMemset(rC, 0, (size_t)RM * cols * 8));
  float t = timeit(run_rect, REPS);
  CK(cudaFree(rB));
  CK(cudaFree(rC));
  return (double)t / ((double)RM * cols * RK);
}

static void rect_rows(int M, int K, float t_ref) {
  const double ref_per_mac = (double)t_ref / ((double)N * N * N);
  RM = M;
  RK = K;
  rA = rand_dev((size_t)M * K);
  for (unsigned nm : kOzIIModuliCounts) {
    ARG = nm;
    double rel = rect_per_mac(12288) / ref_per_mac;
    fprintf(stderr,
            "[dispatch_calibrate] rect nm=%2u M=%d Ncols=12288 K=%d  rel = "
            "%.6f\n",
            nm, M, K, rel);
    printf("ozaki_dispatch_rect_rel,nm%u_m%dk%d,%.6f\n", nm, M, K, rel);
  }
  CK(cudaFree(rA));
}

// The native DGEMM through the same entry (nm = 0) at a product's own extents:
// a DGEMM with few columns runs far below its rate at many, so a call whose
// extents are constants of its compile is priced from this row, keyed by them.
static void full_row(int M, int Ncols, int K, float t_ref) {
  const double ref_per_mac = (double)t_ref / ((double)N * N * N);
  RM = M;
  RK = K;
  rA = rand_dev((size_t)M * K);
  ARG = 0;
  double rel = rect_per_mac(Ncols) / ref_per_mac;
  fprintf(stderr,
          "[dispatch_calibrate] rect dgemm M=%d Ncols=%d K=%d  rel = %.6f\n", M,
          Ncols, K, rel);
  printf("ozaki_dispatch_rect_rel,dgemm_m%dn%dk%d,%.6f\n", M, Ncols, K, rel);
  CK(cudaFree(rA));
}

// One row, rel = (time of `fn`) / t_ref.
static void row(const char *family, const char *key, void (*fn)(),
                float t_ref) {
  float t = timeit(fn, REPS);
  double rel = (double)t / (double)t_ref;
  fprintf(stderr,
          "[dispatch_calibrate] %s,%s  dispatch = %.4f ms  rel = %.6f "
          "(%.2fx vs FP64)\n",
          family, key, t, rel, t_ref / t);
  printf("%s,%s,%.6f\n", family, key, rel);
}

static int ozaki(int argc, char **argv, float t_ref) {
  if (argc > 4) {
    for (int i = 4; i < argc; ++i) {
      int M = 0, Ncols = 0, K = 0, end = 0;
      if (sscanf(argv[i], "%dx%dx%d%n", &M, &Ncols, &K, &end) == 3 &&
          !argv[i][end] && M > 0 && Ncols > 0 && K > 0) {
        full_row(M, Ncols, K, t_ref);
        continue;
      }
      if (sscanf(argv[i], "%dx%d%n", &M, &K, &end) != 2 || argv[i][end] ||
          M <= 0 || K <= 0) {
        fprintf(stderr,
                "[dispatch_calibrate] '%s' is not a shape MxK or MxNxK\n",
                argv[i]);
        return 1;
      }
      rect_rows(M, K, t_ref);
    }
    return 0;
  }
  // Every modulus count the dispatch candidate generator can propose, measured
  // in one session so the rows are mutually consistent.
  char key[16];
  for (unsigned nm : kOzIIModuliCounts) {
    ARG = nm;
    snprintf(key, sizeof(key), "nm%u", nm);
    row("ozaki_dispatch_rel", key, run_oz, t_ref);
  }
  // The native DGEMM through the same dispatch entry (nm = 0), what the native
  // DGEMM candidate runs; the warmup call amortizes the handle creation.
  ARG = 0;
  row("ozaki_dispatch_rel", "dgemm", run_oz, t_ref);
  return 0;
}

static int direct(float t_ref) {
  // The modes of __poseidon_direct_dgemm, in the class spelling of the rows.
  static const struct {
    int Mode;
    const char *Class;
  } Classes[] = {{1, "f16_f32"},
                 {2, "bf16_f32"},
#ifndef __HIP__
                 {3, "tf32_f32"},
#endif
                 {4, "f32_f32"}};
  for (const auto &C : Classes) {
    ARG = C.Mode;
    row("direct_dispatch_rel", C.Class, run_dr, t_ref);
  }
  return 0;
}

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
  N = atoi(argv[2]);
  REPS = atoi(argv[3]);
  size_t sz = (size_t)N * N;
  srand(1);
  dA = rand_dev(sz);
  dB = rand_dev(sz);
  CK(cudaMalloc(&dC, sz * 8));

  float t_ref = timeit(run_ref, REPS);
  fprintf(stderr,
          "[dispatch_calibrate] N=%d reps=%d  scalar-FP64 ref = %.4f ms\n", N,
          REPS, t_ref);
  if (!strcmp(family, "ozaki"))
    return ozaki(argc, argv, t_ref);
  if (!strcmp(family, "direct"))
    return direct(t_ref);
#if POSEIDON_TCEC
  row("tcec_dispatch_rel", "fp16tcec", run_tc, t_ref);
#endif
  return 0;
}

// tcec_rt.cu: Poseidon host-dispatch runtime for tensor-core error
// correction (TCEC), the F32-accuracy-class sibling of ozaki_rt.cu. The
// product is cuMpSGEMM's, the published implementation of the scheme
// (Ootomo and Yokota): its error correction accumulates outside the tensor
// cores with round-to-nearest, which a GEMM library accumulating inside them
// does not reproduce. It is built only with POSEIDON_TCEC_USE_CUMPSGEMM.
//
// The operands are narrowed to FP32 for the library, which runs with
// alpha=1,beta=0 into scratch; the FP64 alpha/beta are applied in the widen
// kernel, so C = alpha*(A@B) + beta*C keeps FP64 scalar semantics.
//
// TCEC is not exact: the FP64->FP32 narrowing (~6e-8 relative) composes with
// the FP16+correction product error, giving F32-class results (~2e-7 relL2 at
// N=2048). Callers needing better must use the Ozaki-II entry.
//
// mode mirrors the Ozaki entry's num_moduli slot so the two entries share one
// calling convention:
//   0 = native cuBLAS DGEMM (same escape hatch as __poseidon_ozaki_dgemm nm==0)
//   1 = FP16TCEC (default)
//   2 = TF32TCEC
//   3 = FP16TC  (no correction, ablation only)
//   4 = TF32TC  (no correction)
// Out-of-range values abort rather than silently substituting a precision.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cumpsgemm/cumpsgemm.hpp>

#include "tcec_scale.h"

#define PTC_CK(x)                                                              \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "poseidon_tcec_rt FATAL CUDA %s:%d %s\n", __FILE__,      \
              __LINE__, cudaGetErrorString(e_));                               \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define PTC_CB(x)                                                              \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "poseidon_tcec_rt FATAL cuBLAS %s:%d status %d\n",       \
              __FILE__, __LINE__, (int)s_);                                    \
      abort();                                                                 \
    }                                                                          \
  } while (0)

namespace {

cublasHandle_t ptc_cublas() {
  static cublasHandle_t h = nullptr;
  if (!h)
    PTC_CB(cublasCreate(&h));
  return h;
}
cuMpSGEMM_handle_t ptc_cumps() {
  static cuMpSGEMM_handle_t h = nullptr;
  static bool init = false;
  if (!init) {
    cumpsgemm::create(h);
    init = true;
  }
  return h;
}

struct PtcScratch {
  size_t capA = 0, capB = 0, capC = 0;
  float *A = nullptr, *B = nullptr, *C = nullptr;
  // Exponents of the power-of-two scales: [0] of A, [1] of B, [2] of the
  // result, which undoes both.
  int *scaleExp = nullptr;
  unsigned long long *absmax = nullptr; // [0]=max|A| bits, [1]=max|B| bits
};
PtcScratch g;

void ptc_reserve(float **p, size_t *cap, size_t need) {
  if (*cap >= need)
    return;
  if (*p)
    PTC_CK(cudaFree(*p));
  PTC_CK(cudaMalloc(p, need * sizeof(float)));
  *cap = need;
}

// FP16 spans only ~4 decades, so an FP32 operand outside that window splits
// into Inf/NaN limbs. One power-of-two scale per operand suffices (exact in
// both directions) because the correction is applied to the whole product.
__device__ __forceinline__ unsigned long long ptc_absBits(double v) {
  // monotone over non-negative doubles
  return (unsigned long long)__double_as_longlong(fabs(v));
}

__device__ __forceinline__ unsigned long long
ptc_warpMax(unsigned long long m) {
  for (int off = 16; off; off >>= 1) {
    unsigned long long o = __shfl_xor_sync(0xffffffffu, m, off);
    m = o > m ? o : m;
  }
  return m;
}

// Layout-aware, so a caller passing a strided sub-matrix view (ld larger than
// the packed extent) is measured over exactly the elements the GEMM will read,
// not over whatever happens to lie in the gaps. Only magnitudes matter, so the
// operand is walked in storage order: `inner` elements are contiguous and the
// `outer` runs are ld apart. Each thread folds PTC_AM_RUN runs and each block
// publishes one atomic: a 64-bit atomic per warp on the one slot costs more
// than reading the operand.
#define PTC_AM_THREADS 256
#define PTC_AM_RUN 16
__global__ void ptc_absmax(const double *__restrict__ X,
                           unsigned long long *out, int slot, int rows,
                           int cols, int ld, int colMajor) {
  int inner = colMajor ? rows : cols, outer = colMajor ? cols : rows;
  int i = blockIdx.x * PTC_AM_THREADS + threadIdx.x;
  unsigned long long m = 0;
  if (i < inner)
    for (int o0 = blockIdx.y * PTC_AM_RUN; o0 < outer;
         o0 += gridDim.y * PTC_AM_RUN) {
      int e = min(outer, o0 + PTC_AM_RUN);
      for (int o = o0; o < e; ++o) {
        unsigned long long v = ptc_absBits(X[(size_t)o * ld + i]);
        m = v > m ? v : m;
      }
    }
  __shared__ unsigned long long warpMax[PTC_AM_THREADS / 32];
  m = ptc_warpMax(m);
  if ((threadIdx.x & 31) == 0)
    warpMax[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = ptc_warpMax(threadIdx.x < PTC_AM_THREADS / 32 ? warpMax[threadIdx.x]
                                                      : 0ull);
    if (threadIdx.x == 0 && m)
      atomicMax(&out[slot], m);
  }
}

void ptc_launchAbsmax(const double *X, unsigned long long *out, int slot,
                      int rows, int cols, int ld, int colMajor,
                      cudaStream_t stream) {
  int inner = colMajor ? rows : cols, outer = colMajor ? cols : rows;
  int runs = (outer + PTC_AM_RUN - 1) / PTC_AM_RUN;
  dim3 grid((inner + PTC_AM_THREADS - 1) / PTC_AM_THREADS,
            runs < 65535 ? runs : 65535);
  ptc_absmax<<<grid, PTC_AM_THREADS, 0, stream>>>(X, out, slot, rows, cols, ld,
                                                  colMajor);
}

__global__ void ptc_scales(const unsigned long long *absmax, int *scaleExp) {
  int ea = ptc_scaleExponent(__longlong_as_double((long long)absmax[0]));
  int eb = ptc_scaleExponent(__longlong_as_double((long long)absmax[1]));
  scaleExp[0] = ea;
  scaleExp[1] = eb;
  scaleExp[2] = -(ea + eb);
}

// Narrow a layout-aware FP64 source into a contiguous column-major FP32 buffer
// of the given logical shape, so every {row,col}-major input combination maps
// onto a single NN library call.
//   src element (r, c) lives at c*ld + r when colMajor, else r*ld + c.
//   dst is rows x cols column-major with leading dimension rows.
__global__ void ptc_narrow(float *__restrict__ dst, const double *__restrict__ src,
                           int rows, int cols, int ld, int colMajor,
                           const int *__restrict__ scaleExp, int which) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols)
    return;
  size_t s = colMajor ? (size_t)c * ld + r : (size_t)r * ld + c;
  dst[(size_t)c * rows + r] = (float)ptc_ldexp(src[s], scaleExp[which]);
}

// Widen the column-major FP32 result back into the layout-aware FP64
// destination, applying the FP64 alpha/beta the library never saw. The
// epilogue's one fused multiply-add is written out so that its rounding does
// not depend on the compiler's contraction.
__global__ void ptc_widen(double *__restrict__ C, const float *__restrict__ src,
                          int rows, int cols, int ldc, int cColMajor,
                          double alpha, double beta,
                          const int *__restrict__ scaleExp) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols)
    return;
  size_t o = (size_t)c * rows + r;
  size_t d = cColMajor ? (size_t)c * ldc + r : (size_t)r * ldc + c;
  double v = ptc_ldexp((double)src[o], scaleExp[2]);
  C[d] = (beta == 0.0) ? alpha * v : fma(alpha, v, beta * C[d]);
}

cuMpSGEMM_compute_mode_t ptc_mode(int mode) {
  switch (mode) {
  case 1: return CUMPSGEMM_FP16TCEC;
  case 2: return CUMPSGEMM_TF32TCEC;
  case 3: return CUMPSGEMM_FP16TC;
  case 4: return CUMPSGEMM_TF32TC;
  }
  fprintf(stderr,
          "poseidon_tcec_rt FATAL: unsupported mode %d (0=dgemm, 1=FP16TCEC, "
          "2=TF32TCEC, 3=FP16TC, 4=TF32TC). Refusing to substitute a "
          "different precision.\n",
          mode);
  abort();
}

// Layout-aware native DGEMM (mode == 0): every {a,b,c}ColMajor combination is
// one call.
void ptc_native_dgemm(double *C, const double *A, const double *B, int M,
                      int Ncols, int K, int lda, int ldb, int ldc,
                      int aColMajor, int bColMajor, int cColMajor, double alpha,
                      double beta, cudaStream_t stream) {
  cublasHandle_t h = ptc_cublas();
  PTC_CB(cublasSetStream(h, stream));
  if (cColMajor) {
    PTC_CB(cublasDgemm(h, aColMajor ? CUBLAS_OP_N : CUBLAS_OP_T,
                       bColMajor ? CUBLAS_OP_N : CUBLAS_OP_T, M, Ncols, K,
                       &alpha, A, lda, B, ldb, &beta, C, ldc));
  } else {
    PTC_CB(cublasDgemm(h, bColMajor ? CUBLAS_OP_T : CUBLAS_OP_N,
                       aColMajor ? CUBLAS_OP_T : CUBLAS_OP_N, Ncols, M, K,
                       &alpha, B, ldb, A, lda, &beta, C, ldc));
  }
}

// The one real path: narrow -> cuMpSGEMM (alpha=1,beta=0) -> widen.
// Operands are packed column-major, so the library call is a plain NN GEMM.
void ptc_run(double *C, const double *A, const double *B, int M, int Ncols,
             int K, int lda, int ldb, int ldc, int aColMajor, int bColMajor,
             int cColMajor, double alpha, double beta, cudaStream_t stream,
             int mode) {
  ptc_reserve(&g.A, &g.capA, (size_t)M * K);
  ptc_reserve(&g.B, &g.capB, (size_t)K * Ncols);
  ptc_reserve(&g.C, &g.capC, (size_t)M * Ncols);
  if (!g.scaleExp) {
    PTC_CK(cudaMalloc(&g.scaleExp, 3 * sizeof(int)));
    PTC_CK(cudaMalloc(&g.absmax, 2 * sizeof(unsigned long long)));
  }

  // Range normalization must see the operands as laid out, but only their
  // magnitudes, so it can stream them flat regardless of layout.
  PTC_CK(cudaMemsetAsync(g.absmax, 0, 2 * sizeof(unsigned long long), stream));
  ptc_launchAbsmax(A, g.absmax, 0, M, K, lda, aColMajor, stream);
  ptc_launchAbsmax(B, g.absmax, 1, K, Ncols, ldb, bColMajor, stream);
  dim3 b(16, 16);
  ptc_scales<<<1, 1, 0, stream>>>(g.absmax, g.scaleExp);

  ptc_narrow<<<dim3((K + 15) / 16, (M + 15) / 16), b, 0, stream>>>(
      g.A, A, M, K, lda, aColMajor, g.scaleExp, 0);
  ptc_narrow<<<dim3((Ncols + 15) / 16, (K + 15) / 16), b, 0, stream>>>(
      g.B, B, K, Ncols, ldb, bColMajor, g.scaleExp, 1);

  const float one = 1.0f, zero = 0.0f;
  cuMpSGEMM_handle_t h = ptc_cumps();
  cumpsgemm::set_stream(h, stream);
  cumpsgemm::gemm<float>(h, CUBLAS_OP_N, CUBLAS_OP_N, (uint64_t)M,
                         (uint64_t)Ncols, (uint64_t)K, &one, g.A, (uint64_t)M,
                         g.B, (uint64_t)K, &zero, g.C, (uint64_t)M,
                         ptc_mode(mode));

  ptc_widen<<<dim3((Ncols + 15) / 16, (M + 15) / 16), b, 0, stream>>>(
      C, g.C, M, Ncols, ldc, cColMajor, alpha, beta, g.scaleExp);
}

} // namespace

// Square entry: same ABI and same accepted-shape set as
// __poseidon_ozaki_dgemm (row-major, leading dim N, no transpose).
extern "C" void __poseidon_tcec_dgemm(double *C, const double *A,
                                      const double *B, int N, int lda, int ldb,
                                      int ldc, int transA, int transB,
                                      double alpha, double beta,
                                      cudaStream_t stream, int mode) {
  if (N <= 0)
    return;
  if (transA || transB || (lda && lda != N) || (ldb && ldb != N) ||
      (ldc && ldc != N)) {
    fprintf(stderr,
            "poseidon_tcec_rt FATAL: unsupported square GEMM shape (transA=%d "
            "transB=%d lda=%d ldb=%d ldc=%d N=%d)\n",
            transA, transB, lda, ldb, ldc, N);
    abort();
  }
  if (mode == 0) {
    ptc_native_dgemm(C, A, B, N, N, N, N, N, N, /*aColMajor=*/0,
                     /*bColMajor=*/0, /*cColMajor=*/0, alpha, beta, stream);
    return;
  }
  ptc_run(C, A, B, N, N, N, N, N, N, /*aColMajor=*/0, /*bColMajor=*/0,
          /*cColMajor=*/0, alpha, beta, stream, mode);
}

// General M x Ncols x K entry: operands are repacked column-major during the
// narrow, so any shape and layout combination is one library call, without
// the Ozaki _ex path's square padding.
extern "C" void __poseidon_tcec_dgemm_ex(double *C, const double *A,
                                         const double *B, int M, int Ncols,
                                         int K, int lda, int ldb, int ldc,
                                         int aColMajor, int bColMajor,
                                         int cColMajor, double alpha,
                                         double beta, cudaStream_t stream,
                                         int mode) {
  if (M <= 0 || Ncols <= 0 || K <= 0)
    return;
  if (mode == 0) {
    ptc_native_dgemm(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor, bColMajor,
                     cColMajor, alpha, beta, stream);
    return;
  }
  ptc_run(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor, bColMajor, cColMajor,
          alpha, beta, stream, mode);
}

// tcec_rt.cu: Poseidon host-dispatch runtime for tensor-core error
// correction (TCEC), the F32-accuracy-class sibling of ozaki_rt.cu.
// -DPOSEIDON_TCEC_USE_CUMPSGEMM delegates to cuMpSGEMM instead of the built-in
// three-GEMM cuBLAS backend.
//
// The split, on operands already narrowed to FP32:
//   A = A_hi + A_lo/S,  A_hi = fp16(A),  A_lo = fp16((A - A_hi)*S),  S = 2^11
//   A*B ~ A_hi*B_hi + (A_hi*B_lo + A_lo*B_hi)/S
// with A_lo*B_lo dropped (O(S^-2), below the F32 target). The two correction
// products accumulate into their own FP32 buffer: added directly they would be
// absorbed by the much larger main term. The GEMM runs with alpha=1,beta=0
// into scratch and the FP64 alpha/beta are applied in the widen kernel, so
// C = alpha*(A@B) + beta*C keeps FP64 scalar semantics.
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef POSEIDON_TCEC_USE_CUMPSGEMM
#include <cumpsgemm/cumpsgemm.hpp>
#endif

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
#ifdef POSEIDON_TCEC_USE_CUMPSGEMM
cuMpSGEMM_handle_t ptc_cumps() {
  static cuMpSGEMM_handle_t h = nullptr;
  static bool init = false;
  if (!init) {
    cumpsgemm::create(h);
    init = true;
  }
  return h;
}
#endif

struct PtcScratch {
  size_t capA = 0, capB = 0, capC = 0;
  float *A = nullptr, *B = nullptr, *C = nullptr;
  // Split-correction working set (own-backend path only).
  size_t capAh = 0, capBh = 0, capCc = 0;
  __half *Ah = nullptr, *Al = nullptr, *Bh = nullptr, *Bl = nullptr;
  float *Ccorr = nullptr;
  // [0]=scaleA [1]=scaleB [2]=1/(scaleA*scaleB); all exact powers of two.
  float *scale = nullptr;
  unsigned *absmax = nullptr; // [0]=max|A| bits, [1]=max|B| bits
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

// Reserve a paired FP16 hi/lo buffer of `need` elements each.
void ptc_reserve_half2(__half **hi, __half **lo, size_t *cap, size_t need) {
  if (*cap >= need)
    return;
  if (*hi) { PTC_CK(cudaFree(*hi)); PTC_CK(cudaFree(*lo)); }
  PTC_CK(cudaMalloc(hi, need * sizeof(__half)));
  PTC_CK(cudaMalloc(lo, need * sizeof(__half)));
  *cap = need;
}

// Ootomo's split scale: the residual A - fp16(A) is ~2^-11 of A, so lifting it
// by 2^11 puts it back in FP16's normal range instead of its subnormals.
#define PTC_SPLIT_S 2048.0f

__global__ void ptc_split(const float *__restrict__ src, __half *__restrict__ hi,
                          __half *__restrict__ lo, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = src[i];
  __half h = __float2half(v);
  hi[i] = h;
  lo[i] = __float2half((v - __half2float(h)) * PTC_SPLIT_S);
}

// FP16 spans only ~4 decades, so an FP32 operand outside that window splits
// into Inf/NaN limbs. One power-of-two scale per operand suffices (exact in
// both directions) because the correction is applied to the whole product.
__device__ __forceinline__ unsigned ptc_absBits(double v) {
  float f = (float)fabs(v);
  return __float_as_uint(f); // monotone over non-negative floats
}

// Layout-aware, so a caller passing a strided sub-matrix view (ld larger than
// the packed extent) is measured over exactly the elements the GEMM will read,
// not over whatever happens to lie in the gaps.
__global__ void ptc_absmax(const double *__restrict__ X, unsigned *out, int slot,
                           int rows, int cols, int ld, int colMajor) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned m = 0;
  if (r < rows && c < cols) {
    size_t s = colMajor ? (size_t)c * ld + r : (size_t)r * ld + c;
    m = ptc_absBits(X[s]);
  }
  for (int off = 16; off; off >>= 1) {
    unsigned o = __shfl_xor_sync(0xffffffffu, m, off);
    m = o > m ? o : m;
  }
  if ((threadIdx.x & 31) == 0)
    atomicMax(&out[slot], m);
}

// Target magnitude 2^10: comfortably inside FP16's 65504 with headroom for the
// high limb of the split, and far above the subnormal floor.
__global__ void ptc_scales(const unsigned *absmax, float *scale) {
  float ma = __uint_as_float(absmax[0]);
  float mb = __uint_as_float(absmax[1]);
  float sa = (ma > 0.f && isfinite(ma)) ? exp2f((float)(10 - ilogbf(ma))) : 1.f;
  float sb = (mb > 0.f && isfinite(mb)) ? exp2f((float)(10 - ilogbf(mb))) : 1.f;
  scale[0] = sa;
  scale[1] = sb;
  scale[2] = 1.f / (sa * sb); // exact: both are powers of two
}

// Narrow a layout-aware FP64 source into a contiguous column-major FP32 buffer
// of the given logical shape, so every {row,col}-major input combination maps
// onto a single NN library call.
//   src element (r, c) lives at c*ld + r when colMajor, else r*ld + c.
//   dst is rows x cols column-major with leading dimension rows.
__global__ void ptc_narrow(float *__restrict__ dst, const double *__restrict__ src,
                           int rows, int cols, int ld, int colMajor,
                           const float *__restrict__ scale, int which) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols)
    return;
  size_t s = colMajor ? (size_t)c * ld + r : (size_t)r * ld + c;
  dst[(size_t)c * rows + r] = (float)(src[s] * (double)scale[which]);
}

// Widen the column-major FP32 result back into the layout-aware FP64
// destination, applying the FP64 alpha/beta the library never saw. `corr` is
// the separate correction accumulator (null when the backend already folded
// the correction in).
__global__ void ptc_widen(double *__restrict__ C, const float *__restrict__ src,
                          const float *__restrict__ corr, int rows, int cols,
                          int ldc, int cColMajor, double alpha, double beta,
                          const float *__restrict__ scale) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || c >= cols)
    return;
  size_t o = (size_t)c * rows + r;
  size_t d = cColMajor ? (size_t)c * ldc + r : (size_t)r * ldc + c;
  double v = (double)src[o];
  if (corr)
    v += (double)corr[o] / (double)PTC_SPLIT_S;
  v *= (double)scale[2];
  C[d] = (beta == 0.0) ? alpha * v : alpha * v + beta * C[d];
}

#ifdef POSEIDON_TCEC_USE_CUMPSGEMM
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
#endif

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

// The one real path: narrow -> library TCEC SGEMM (alpha=1,beta=0) -> widen.
// Operands are packed column-major, so the library call is a plain NN GEMM.
void ptc_run(double *C, const double *A, const double *B, int M, int Ncols,
             int K, int lda, int ldb, int ldc, int aColMajor, int bColMajor,
             int cColMajor, double alpha, double beta, cudaStream_t stream,
             int mode) {
  ptc_reserve(&g.A, &g.capA, (size_t)M * K);
  ptc_reserve(&g.B, &g.capB, (size_t)K * Ncols);
  ptc_reserve(&g.C, &g.capC, (size_t)M * Ncols);
  if (!g.scale) {
    PTC_CK(cudaMalloc(&g.scale, 3 * sizeof(float)));
    PTC_CK(cudaMalloc(&g.absmax, 2 * sizeof(unsigned)));
  }

  // Range normalization must see the operands as laid out, but only their
  // magnitudes, so it can stream them flat regardless of layout.
  PTC_CK(cudaMemsetAsync(g.absmax, 0, 2 * sizeof(unsigned), stream));
  dim3 b(16, 16);
  ptc_absmax<<<dim3((K + 15) / 16, (M + 15) / 16), b, 0, stream>>>(
      A, g.absmax, 0, M, K, lda, aColMajor);
  ptc_absmax<<<dim3((Ncols + 15) / 16, (K + 15) / 16), b, 0, stream>>>(
      B, g.absmax, 1, K, Ncols, ldb, bColMajor);
  ptc_scales<<<1, 1, 0, stream>>>(g.absmax, g.scale);

  ptc_narrow<<<dim3((K + 15) / 16, (M + 15) / 16), b, 0, stream>>>(
      g.A, A, M, K, lda, aColMajor, g.scale, 0);
  ptc_narrow<<<dim3((Ncols + 15) / 16, (K + 15) / 16), b, 0, stream>>>(
      g.B, B, K, Ncols, ldb, bColMajor, g.scale, 1);

  const float one = 1.0f, zero = 0.0f;
  const float *corr = nullptr;
#ifdef POSEIDON_TCEC_USE_CUMPSGEMM
  {
    cuMpSGEMM_handle_t h = ptc_cumps();
    cumpsgemm::set_stream(h, stream);
    cumpsgemm::gemm<float>(h, CUBLAS_OP_N, CUBLAS_OP_N, (uint64_t)M,
                           (uint64_t)Ncols, (uint64_t)K, &one, g.A, (uint64_t)M,
                           g.B, (uint64_t)K, &zero, g.C, (uint64_t)M,
                           ptc_mode(mode));
  }
#else
  if (mode != 1) {
    fprintf(stderr,
            "poseidon_tcec_rt FATAL: mode %d is only available from the "
            "cuMpSGEMM reference backend (-DPOSEIDON_TCEC_USE_CUMPSGEMM); the "
            "self-contained backend implements FP16TCEC (mode 1) only. "
            "Refusing to substitute a different precision.\n", mode);
    abort();
  }
  ptc_reserve_half2(&g.Ah, &g.Al, &g.capAh, (size_t)M * K);
  ptc_reserve_half2(&g.Bh, &g.Bl, &g.capBh, (size_t)K * Ncols);
  ptc_reserve(&g.Ccorr, &g.capCc, (size_t)M * Ncols);
  {
    size_t nA = (size_t)M * K, nB = (size_t)K * Ncols;
    ptc_split<<<(unsigned)((nA + 255) / 256), 256, 0, stream>>>(g.A, g.Ah, g.Al, nA);
    ptc_split<<<(unsigned)((nB + 255) / 256), 256, 0, stream>>>(g.B, g.Bh, g.Bl, nB);
    cublasHandle_t h = ptc_cublas();
    PTC_CB(cublasSetStream(h, stream));
    // main term
    PTC_CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, Ncols, K, &one, g.Ah,
                        CUDA_R_16F, M, g.Bh, CUDA_R_16F, K, &zero, g.C,
                        CUDA_R_32F, M, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    // corrections, into their own accumulator
    PTC_CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, Ncols, K, &one, g.Ah,
                        CUDA_R_16F, M, g.Bl, CUDA_R_16F, K, &zero, g.Ccorr,
                        CUDA_R_32F, M, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    PTC_CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, Ncols, K, &one, g.Al,
                        CUDA_R_16F, M, g.Bh, CUDA_R_16F, K, &one, g.Ccorr,
                        CUDA_R_32F, M, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    corr = g.Ccorr;
  }
#endif

  ptc_widen<<<dim3((Ncols + 15) / 16, (M + 15) / 16), b, 0, stream>>>(
      C, g.C, corr, M, Ncols, ldc, cColMajor, alpha, beta, g.scale);
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

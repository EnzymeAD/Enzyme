// direct_rt.cu: Poseidon host-dispatch runtime for the direct
// reduced-precision GEMM, the library realization of the in-kernel direct
// tensor-core raise.
//
// This runtime performs no operand scaling: each FP64 operand is rounded once
// to the target input format, the product runs as one cublasGemmEx with an
// FP32 accumulator, and the result is widened back to FP64. That is bit for
// bit the numerics the in-kernel direct raise models (including FP16's
// subnormal floor and 65504 ceiling), which is why the candidate borrows that
// raise's accuracy evaluation; a range normalization here would make this
// entry more accurate than the model that prices it. Operands that leave the
// format's range belong to TCEC or Ozaki-II.
//
// The GEMM runs with alpha=1, beta=0 into scratch and the FP64 alpha/beta are
// applied in the widen kernel, so C = alpha*(A@B) + beta*C keeps FP64 scalar
// semantics.
//
// mode mirrors the Ozaki entry's num_moduli slot so all three dispatch entries
// share one calling convention:
//   0 = native cuBLAS DGEMM (same escape hatch as __poseidon_ozaki_dgemm nm==0
//       and __poseidon_tcec_dgemm mode==0)
//   1 = FP16 inputs, FP32 accumulator   (CUDA_R_16F,  CUBLAS_COMPUTE_32F)
//   2 = BF16 inputs, FP32 accumulator   (CUDA_R_16BF, CUBLAS_COMPUTE_32F)
//   3 = TF32 inputs, FP32 accumulator   (CUDA_R_32F,  CUBLAS_COMPUTE_32F_FAST_TF32)
//   4 = FP32 inputs, FP32 accumulator   (CUDA_R_32F,  CUBLAS_COMPUTE_32F)
// Mode 4 is the one entry that does not run on tensor cores: CUBLAS_COMPUTE_32F
// over CUDA_R_32F operands is cuBLAS's SGEMM, kept beside the tensor-core modes
// because the operand narrow, the FP32 accumulator and the widen-back are
// identical and only the compute type differs.
// Out-of-range values abort rather than silently substituting a precision.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define PDR_CK(x)                                                              \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "poseidon_direct_rt FATAL CUDA %s:%d %s\n", __FILE__,    \
              __LINE__, cudaGetErrorString(e_));                               \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define PDR_CB(x)                                                              \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "poseidon_direct_rt FATAL cuBLAS %s:%d status %d\n",     \
              __FILE__, __LINE__, (int)s_);                                    \
      abort();                                                                 \
    }                                                                          \
  } while (0)

namespace {

cublasHandle_t pdr_cublas() {
  static cublasHandle_t h = nullptr;
  if (!h)
    PDR_CB(cublasCreate(&h));
  return h;
}

// Widest element the operand buffers ever hold is a float (the TF32 and FP32
// modes), so one byte-capacity pair per operand serves every mode.
struct PdrScratch {
  size_t capA = 0, capB = 0, capC = 0; // capA/capB in BYTES, capC in floats
  void *A = nullptr, *B = nullptr;
  float *C = nullptr;
};
PdrScratch g;

void pdr_reserve_bytes(void **p, size_t *cap, size_t need) {
  if (*cap >= need)
    return;
  if (*p)
    PDR_CK(cudaFree(*p));
  PDR_CK(cudaMalloc(p, need));
  *cap = need;
}

void pdr_reserve_f32(float **p, size_t *cap, size_t need) {
  if (*cap >= need)
    return;
  if (*p)
    PDR_CK(cudaFree(*p));
  PDR_CK(cudaMalloc(p, need * sizeof(float)));
  *cap = need;
}

// One rounding step, FP64 -> the tensor-core input format. Rounding through an
// intermediate FP32 would double-round; these are the same primitives the
// in-kernel raise emits.
__device__ __forceinline__ void pdr_round(__half *d, double v) {
  *d = __double2half(v);
}
__device__ __forceinline__ void pdr_round(__nv_bfloat16 *d, double v) {
  *d = __double2bfloat16(v);
}
__device__ __forceinline__ void pdr_round(float *d, double v) {
  *d = (float)v; // mode 4 stops here; mode 3's TF32 truncation is in the core
}

// Narrow a layout-aware FP64 source into a contiguous buffer that keeps the
// source's own layout (packed to the tight leading dimension), so both the
// read and the write are coalesced and the transpose becomes the GEMM's op
// flag.
//   colMajor source: element (r,c) at c*ld + r  ->  dst[c*rows + r]
//   row-major source: element (r,c) at r*ld + c ->  dst[r*cols + c]
template <typename T>
__global__ void pdr_narrow(T *__restrict__ dst, const double *__restrict__ src,
                           int rows, int cols, int ld, int colMajor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // fastest-varying axis
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (colMajor) {
    if (i >= rows || j >= cols)
      return;
    pdr_round(&dst[(size_t)j * rows + i], src[(size_t)j * ld + i]);
  } else {
    if (i >= cols || j >= rows)
      return;
    pdr_round(&dst[(size_t)j * cols + i], src[(size_t)j * ld + i]);
  }
}

// Packed fast path: when the leading dimension equals the packed extent the
// narrow is a flat elementwise pass; the general kernel above stays for
// strided sub-matrix views.
template <typename T>
__global__ void pdr_narrow_flat(T *__restrict__ dst,
                                const double *__restrict__ src, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    pdr_round(&dst[i], src[i]);
}

// Widen the FP32 result back into the layout-aware FP64 destination, applying
// the FP64 alpha/beta the library never saw. The scratch is column-major
// M x Ncols (ld M) when C is column-major and column-major Ncols x M (ld Ncols)
// when C is row-major (the operands are swapped), so in both cases the scratch
// index is (slow index)*(fast extent) + (fast index) in C's axis order.
//
// ACC is a template parameter rather than a runtime `beta == 0` test: written
// as a ternary the accumulate arm's `C[d]` read is emitted as an unconditional
// load, so the beta = 0 case pays a full extra read of the FP64 output buffer
// (measured 1.82x the non-accumulating kernel on ozp at N = 2048).
template <bool ACC>
__global__ void pdr_widen(double *__restrict__ C, const float *__restrict__ src,
                          int rows, int cols, int ldc, int cColMajor,
                          double alpha, double beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  size_t o, d;
  if (cColMajor) {
    if (i >= rows || j >= cols)
      return;
    o = (size_t)j * rows + i;
    d = (size_t)j * ldc + i;
  } else {
    if (i >= cols || j >= rows)
      return;
    o = (size_t)j * cols + i;
    d = (size_t)j * ldc + i;
  }
  double v = alpha * (double)src[o];
  if (ACC)
    C[d] = v + beta * C[d];
  else
    C[d] = v;
}

// Packed fast path for the widen: when C's leading dimension equals its packed
// extent the scratch index and the destination index coincide.
template <bool ACC>
__global__ void pdr_widen_flat(double *__restrict__ C,
                               const float *__restrict__ src, size_t n,
                               double alpha, double beta) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double v = alpha * (double)src[i];
  if (ACC)
    C[i] = v + beta * C[i];
  else
    C[i] = v;
}

struct PdrFormat {
  cudaDataType_t dt;
  cublasComputeType_t ct;
  size_t elemBytes;
};

PdrFormat pdr_format(int mode) {
  switch (mode) {
  case 1:
    return {CUDA_R_16F, CUBLAS_COMPUTE_32F, sizeof(__half)};
  case 2:
    return {CUDA_R_16BF, CUBLAS_COMPUTE_32F, sizeof(__nv_bfloat16)};
  case 3:
    return {CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, sizeof(float)};
  case 4:
    return {CUDA_R_32F, CUBLAS_COMPUTE_32F, sizeof(float)};
  }
  fprintf(stderr,
          "poseidon_direct_rt FATAL: unsupported mode %d (0=dgemm, 1=FP16/F32, "
          "2=BF16/F32, 3=TF32/F32, 4=FP32/F32). Refusing to substitute a "
          "different precision.\n",
          mode);
  abort();
}

// Layout-aware native DGEMM (mode == 0): every {a,b,c}ColMajor combination is
// one call.
void pdr_native_dgemm(double *C, const double *A, const double *B, int M,
                      int Ncols, int K, int lda, int ldb, int ldc,
                      int aColMajor, int bColMajor, int cColMajor, double alpha,
                      double beta, cudaStream_t stream) {
  cublasHandle_t h = pdr_cublas();
  PDR_CB(cublasSetStream(h, stream));
  if (cColMajor) {
    PDR_CB(cublasDgemm(h, aColMajor ? CUBLAS_OP_N : CUBLAS_OP_T,
                       bColMajor ? CUBLAS_OP_N : CUBLAS_OP_T, M, Ncols, K,
                       &alpha, A, lda, B, ldb, &beta, C, ldc));
  } else {
    PDR_CB(cublasDgemm(h, bColMajor ? CUBLAS_OP_T : CUBLAS_OP_N,
                       aColMajor ? CUBLAS_OP_T : CUBLAS_OP_N, Ncols, M, K,
                       &alpha, B, ldb, A, lda, &beta, C, ldc));
  }
}

// The one real path: narrow -> one cublasGemmEx -> widen.
void pdr_run(double *C, const double *A, const double *B, int M, int Ncols,
             int K, int lda, int ldb, int ldc, int aColMajor, int bColMajor,
             int cColMajor, double alpha, double beta, cudaStream_t stream,
             int mode) {
  const PdrFormat f = pdr_format(mode);
  pdr_reserve_bytes(&g.A, &g.capA, (size_t)M * K * f.elemBytes);
  pdr_reserve_bytes(&g.B, &g.capB, (size_t)K * Ncols * f.elemBytes);
  pdr_reserve_f32(&g.C, &g.capC, (size_t)M * Ncols);

  dim3 b(16, 16);
  auto launchNarrow = [&](void *dst, const double *src, int rows, int cols,
                          int ld, int colMajor) {
    if (ld == (colMajor ? rows : cols)) {
      const size_t n = (size_t)rows * cols;
      const unsigned g = (unsigned)((n + 255) / 256);
      switch (mode) {
      case 1:
        pdr_narrow_flat<<<g, 256, 0, stream>>>((__half *)dst, src, n);
        break;
      case 2:
        pdr_narrow_flat<<<g, 256, 0, stream>>>((__nv_bfloat16 *)dst, src, n);
        break;
      default:
        pdr_narrow_flat<<<g, 256, 0, stream>>>((float *)dst, src, n);
        break;
      }
      return;
    }
    dim3 grid(colMajor ? (rows + 15) / 16 : (cols + 15) / 16,
              colMajor ? (cols + 15) / 16 : (rows + 15) / 16);
    switch (mode) {
    case 1:
      pdr_narrow<<<grid, b, 0, stream>>>((__half *)dst, src, rows, cols, ld,
                                         colMajor);
      break;
    case 2:
      pdr_narrow<<<grid, b, 0, stream>>>((__nv_bfloat16 *)dst, src, rows, cols,
                                         ld, colMajor);
      break;
    default:
      pdr_narrow<<<grid, b, 0, stream>>>((float *)dst, src, rows, cols, ld,
                                         colMajor);
      break;
    }
  };
  launchNarrow(g.A, A, M, K, lda, aColMajor);
  launchNarrow(g.B, B, K, Ncols, ldb, bColMajor);

  // The packed A buffer is a column-major M x K matrix (ld = M) when the source
  // was column-major, and a column-major K x M matrix (ld = K) otherwise, so
  // the op flag that recovers the logical M x K operand is N or T respectively;
  // likewise for B. A row-major C is produced by computing its transpose,
  // (A@B)^T = B^T @ A^T, which swaps the operands and flips both op flags.
  const cublasOperation_t opA = aColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t opB = bColMajor ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int packLdA = aColMajor ? M : K;
  const int packLdB = bColMajor ? K : Ncols;
  auto flip = [](cublasOperation_t o) {
    return o == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
  };

  const float one = 1.0f, zero = 0.0f;
  cublasHandle_t h = pdr_cublas();
  PDR_CB(cublasSetStream(h, stream));
  if (cColMajor) {
    PDR_CB(cublasGemmEx(h, opA, opB, M, Ncols, K, &one, g.A, f.dt, packLdA, g.B,
                        f.dt, packLdB, &zero, g.C, CUDA_R_32F, M, f.ct,
                        CUBLAS_GEMM_DEFAULT));
  } else {
    PDR_CB(cublasGemmEx(h, flip(opB), flip(opA), Ncols, M, K, &one, g.B, f.dt,
                        packLdB, g.A, f.dt, packLdA, &zero, g.C, CUDA_R_32F,
                        Ncols, f.ct, CUBLAS_GEMM_DEFAULT));
  }

  const bool acc = (beta != 0.0);
  if (ldc == (cColMajor ? M : Ncols)) {
    const size_t n = (size_t)M * Ncols;
    const unsigned wg = (unsigned)((n + 255) / 256);
    if (acc)
      pdr_widen_flat<true><<<wg, 256, 0, stream>>>(C, g.C, n, alpha, beta);
    else
      pdr_widen_flat<false><<<wg, 256, 0, stream>>>(C, g.C, n, alpha, beta);
  } else {
    dim3 wgrid(cColMajor ? (M + 15) / 16 : (Ncols + 15) / 16,
               cColMajor ? (Ncols + 15) / 16 : (M + 15) / 16);
    if (acc)
      pdr_widen<true><<<wgrid, b, 0, stream>>>(C, g.C, M, Ncols, ldc, cColMajor,
                                               alpha, beta);
    else
      pdr_widen<false><<<wgrid, b, 0, stream>>>(C, g.C, M, Ncols, ldc,
                                                cColMajor, alpha, beta);
  }
}

} // namespace

// Square entry: same ABI and same accepted-shape set as
// __poseidon_ozaki_dgemm and __poseidon_tcec_dgemm (row-major, leading dim N,
// no transpose).
extern "C" void __poseidon_direct_dgemm(double *C, const double *A,
                                        const double *B, int N, int lda,
                                        int ldb, int ldc, int transA,
                                        int transB, double alpha, double beta,
                                        cudaStream_t stream, int mode) {
  if (N <= 0)
    return;
  if (transA || transB || (lda && lda != N) || (ldb && ldb != N) ||
      (ldc && ldc != N)) {
    fprintf(stderr,
            "poseidon_direct_rt FATAL: unsupported square GEMM shape (transA=%d "
            "transB=%d lda=%d ldb=%d ldc=%d N=%d)\n",
            transA, transB, lda, ldb, ldc, N);
    abort();
  }
  if (mode == 0) {
    pdr_native_dgemm(C, A, B, N, N, N, N, N, N, /*aColMajor=*/0,
                     /*bColMajor=*/0, /*cColMajor=*/0, alpha, beta, stream);
    return;
  }
  pdr_run(C, A, B, N, N, N, N, N, N, /*aColMajor=*/0, /*bColMajor=*/0,
          /*cColMajor=*/0, alpha, beta, stream, mode);
}

// General M x Ncols x K entry: the operands are repacked during the narrow, so
// any shape and layout combination is one library call, without the Ozaki _ex
// path's square padding.
extern "C" void __poseidon_direct_dgemm_ex(double *C, const double *A,
                                           const double *B, int M, int Ncols,
                                           int K, int lda, int ldb, int ldc,
                                           int aColMajor, int bColMajor,
                                           int cColMajor, double alpha,
                                           double beta, cudaStream_t stream,
                                           int mode) {
  if (M <= 0 || Ncols <= 0 || K <= 0)
    return;
  if (mode == 0) {
    pdr_native_dgemm(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor, bColMajor,
                     cColMajor, alpha, beta, stream);
    return;
  }
  pdr_run(C, A, B, M, Ncols, K, lda, ldb, ldc, aColMajor, bColMajor, cColMajor,
          alpha, beta, stream, mode);
}

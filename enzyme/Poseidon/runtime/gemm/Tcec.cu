//===- Tcec.cu - error-corrected GEMM runtime -----------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tensor-core error correction (TCEC), the F32-accuracy-class sibling of
// ozaki_rt.cu. The product is cuMpSGEMM's, linked, not included (OzakiII.h):
// its error correction accumulates outside the tensor cores with
// round-to-nearest, which a GEMM library accumulating inside them does not
// reproduce. It is built only with POSEIDON_CUMPSGEMM_ROOT.
//
// FP16 spans only about four decades, so an FP32 operand outside that window
// splits into Inf or NaN limbs. Each operand is scaled by one power of two
// (exact in both directions, tcecScaleExponent) taken from its largest
// magnitude, narrowed to FP32 and packed column-major, so every layout is one
// NN call of the library with alpha = 1 and beta = 0; the product is scaled
// back and widened with the FP64 alpha and beta applied.
//
// TCEC is not exact: the FP64 to FP32 narrowing (about 6e-8 relative)
// composes with the FP16 and correction product error, giving F32-class
// results.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cumpsgemm/cumpsgemm.hpp>

#include "GemmRuntime.h"
#include "Schemes.h"

using namespace poseidon::gemm;

namespace {

struct TcecDevice : Device {
  cuMpSGEMM_handle_t cumpsgemm = nullptr;
  ScratchBuffer A, B, P;
  // The exponents of the scales: [0] of A, [1] of B, [2] of the product, which
  // undoes both.
  int *scaleExp = nullptr;
  // The bits of max|A| and max|B|.
  unsigned long long *maxBits = nullptr;
};

// The bits of |v|, ordered as the magnitudes are.
__device__ __forceinline__ unsigned long long magnitudeBits(double v) {
  return (unsigned long long)__double_as_longlong(fabs(v));
}

__device__ __forceinline__ unsigned long long warpMax(unsigned long long m) {
  for (int off = 16; off; off >>= 1) {
    const unsigned long long o = __shfl_xor_sync(0xffffffffu, m, off);
    m = o > m ? o : m;
  }
  return m;
}

// The magnitude of an operand is read over exactly the elements the product
// reads (a strided view included), in storage order: `inner` elements are
// contiguous and the `outer` runs are ld apart. Each thread folds kMaxRun
// runs and each block publishes one atomic: a 64-bit atomic per warp on the
// one slot costs more than reading the operand.
constexpr int kMaxThreads = 256;
constexpr int kMaxRun = 16;

__global__ void maxMagnitudeKernel(const double *__restrict__ X,
                                   unsigned long long *out, int rows, int cols,
                                   int ld, int colMajor) {
  const int inner = contiguousExtent(rows, cols, colMajor);
  const int outer = stridedExtent(rows, cols, colMajor);
  const int i = blockIdx.x * kMaxThreads + threadIdx.x;
  unsigned long long m = 0;
  if (i < inner)
    for (int o0 = blockIdx.y * kMaxRun; o0 < outer; o0 += gridDim.y * kMaxRun)
      for (int o = o0, e = min(outer, o0 + kMaxRun); o < e; ++o) {
        const unsigned long long v = magnitudeBits(X[(size_t)o * ld + i]);
        m = v > m ? v : m;
      }
  __shared__ unsigned long long perWarp[kMaxThreads / 32];
  m = warpMax(m);
  if ((threadIdx.x & 31) == 0)
    perWarp[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = warpMax(threadIdx.x < kMaxThreads / 32 ? perWarp[threadIdx.x] : 0ull);
    if (threadIdx.x == 0 && m)
      atomicMax(out, m);
  }
}

void maxMagnitude(const double *X, unsigned long long *out, int rows, int cols,
                  int ld, int colMajor, cudaStream_t stream) {
  const int inner = contiguousExtent(rows, cols, colMajor);
  const int runs =
      (stridedExtent(rows, cols, colMajor) + kMaxRun - 1) / kMaxRun;
  const dim3 grid((inner + kMaxThreads - 1) / kMaxThreads,
                  std::min<unsigned>(runs, kMaxGridY));
  maxMagnitudeKernel<<<grid, kMaxThreads, 0, stream>>>(X, out, rows, cols, ld,
                                                       colMajor);
}

__global__ void scaleExponentsKernel(const unsigned long long *maxBits,
                                     int *scaleExp) {
  const int ea = tcecScaleExponent(__longlong_as_double((long long)maxBits[0]));
  const int eb = tcecScaleExponent(__longlong_as_double((long long)maxBits[1]));
  scaleExp[0] = ea;
  scaleExp[1] = eb;
  scaleExp[2] = -(ea + eb);
}

// The scaled FP32 operand, and the product scaled back; scaleExp[which] is
// read on the device.
struct NarrowScaled {
  const int *scaleExp;
  int which;
  __device__ float operator()(double v) const {
    return (float)tcecScale(v, scaleExp[which]);
  }
};
struct WidenScaled {
  const int *scaleExp;
  __device__ double operator()(float p) const {
    return tcecScale((double)p, scaleExp[2]);
  }
};

cuMpSGEMM_compute_mode_t computeModeOf(int mode) {
  switch (mode) {
  case kTcecFp16Tcec:
    return CUMPSGEMM_FP16TCEC;
  case kTcecTf32Tcec:
    return CUMPSGEMM_TF32TCEC;
  case kTcecFp16Tc:
    return CUMPSGEMM_FP16TC;
  case kTcecTf32Tc:
    return CUMPSGEMM_TF32TC;
  }
  fail("__poseidon_tcec_dgemm with mode %d: it takes a PoseidonTcecMode, 1 to "
       "4",
       mode);
}

// The scale of each operand from its largest magnitude.
void scaleExponents(TcecDevice &dev, const double *A, const double *B, int M,
                    int Ncols, int K, int lda, int ldb, int aColMajor,
                    int bColMajor, cudaStream_t stream) {
  if (!dev.scaleExp) {
    POSEIDON_GEMM_CK(cudaMalloc(&dev.scaleExp, 3 * sizeof(int)));
    POSEIDON_GEMM_CK(cudaMalloc(&dev.maxBits, 2 * sizeof(unsigned long long)));
  }
  POSEIDON_GEMM_CK(
      cudaMemsetAsync(dev.maxBits, 0, 2 * sizeof(unsigned long long), stream));
  maxMagnitude(A, dev.maxBits, M, K, lda, aColMajor, stream);
  maxMagnitude(B, dev.maxBits + 1, K, Ncols, ldb, bColMajor, stream);
  scaleExponentsKernel<<<1, 1, 0, stream>>>(dev.maxBits, dev.scaleExp);
}

// Scale and pack, one NN call of cuMpSGEMM with alpha = 1 and beta = 0 into
// the column-major P, unpack.
void multiply(Call<TcecDevice> &call, cuMpSGEMM_compute_mode_t mode, double *C,
              const double *A, const double *B, int M, int Ncols, int K,
              int lda, int ldb, int ldc, int aColMajor, int bColMajor,
              int cColMajor, double alpha, double beta) {
  TcecDevice &dev = call.dev;
  const cudaStream_t stream = call.stream;
  dev.A.reserve(dev, (size_t)M * K * sizeof(float));
  dev.B.reserve(dev, (size_t)K * Ncols * sizeof(float));
  dev.P.reserve(dev, (size_t)M * Ncols * sizeof(float));
  scaleExponents(dev, A, B, M, Ncols, K, lda, ldb, aColMajor, bColMajor,
                 stream);
  pack(dev.A.as<float>(), A, M, K, lda, aColMajor, /*dstColMajor=*/1,
       NarrowScaled{dev.scaleExp, 0}, stream);
  pack(dev.B.as<float>(), B, K, Ncols, ldb, bColMajor, /*dstColMajor=*/1,
       NarrowScaled{dev.scaleExp, 1}, stream);

  if (!dev.cumpsgemm)
    POSEIDON_GEMM_CB(cuMpSGEMM_create(&dev.cumpsgemm));
  POSEIDON_GEMM_CB(cuMpSGEMM_set_stream(dev.cumpsgemm, stream));
  const float one = 1.0f, zero = 0.0f;
  POSEIDON_GEMM_CB(cumpsgemm::gemm<float>(
      dev.cumpsgemm, CUBLAS_OP_N, CUBLAS_OP_N, (uint64_t)M, (uint64_t)Ncols,
      (uint64_t)K, &one, dev.A.as<float>(), (uint64_t)M, dev.B.as<float>(),
      (uint64_t)K, &zero, dev.P.as<float>(), (uint64_t)M, mode));

  unpack(C, dev.P.as<float>(), M, Ncols, ldc, cColMajor, /*pColMajor=*/1, alpha,
         beta, WidenScaled{dev.scaleExp}, stream);
}

} // namespace

extern "C" void __poseidon_tcec_dgemm(double *C, const double *A,
                                      const double *B, int M, int Ncols, int K,
                                      int lda, int ldb, int ldc, int aColMajor,
                                      int bColMajor, int cColMajor,
                                      double alpha, double beta,
                                      cudaStream_t stream, int mode) {
  const cuMpSGEMM_compute_mode_t computeMode = computeModeOf(mode);
  runEntry<TcecDevice>(
      "__poseidon_tcec_dgemm", C, M, Ncols, K, ldc, cColMajor, beta, stream,
      /*scratch=*/true, [&](Call<TcecDevice> &call) {
        multiply(call, computeMode, C, A, B, M, Ncols, K, lda, ldb, ldc,
                 aColMajor, bColMajor, cColMajor, alpha, beta);
      });
}

// The direct-dispatch runtime's FP32 mode (mode 4): FP32 operands, FP32
// accumulator, which is cuBLAS's SGEMM rather than a tensor-core product. The
// pass condition is bit exactness against a reference cublasSgemm over the same
// narrowed operands, so the candidate the solver prices as "24-bit significand,
// FP32 accumulation" is what the library actually runs. The square entry and
// the general M x Ncols x K entry are both covered, with a non-unit alpha/beta
// pair so the FP64 epilogue in the widen kernel is checked too.
//
// RUN: %clang -x cuda --cuda-gpu-arch=%gpu_arch -O2 %s -x none \
// RUN:     %poseidon_rt_lib -L/usr/local/cuda/lib64 -lcublas -lcudart \
// RUN:     -lstdc++ -lm -o %t.exe
// RUN: %t.exe | FileCheck %s
//
// REQUIRES: poseidon, cuda-runtime

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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
    cudaError_t e = (x);                                                       \
    if (e) {                                                                   \
      printf("CUDA %s:%d %s\n", __FILE__, __LINE__,                           \
             cudaGetErrorString(e));                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define CB(x)                                                                  \
  do {                                                                         \
    cublasStatus_t s = (x);                                                    \
    if (s != CUBLAS_STATUS_SUCCESS) {                                          \
      printf("cuBLAS %s:%d status %d\n", __FILE__, __LINE__, (int)s);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void narrow(float *dst, const double *src, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = (float)src[i];
}
__global__ void widen(double *C, const float *src, size_t n, double alpha,
                      double beta) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    C[i] = alpha * (double)src[i] + (beta == 0.0 ? 0.0 : beta * C[i]);
}

// The reference: narrow both row-major operands to FP32, one cublasSgemm in the
// operand order a row-major C needs ((A@B)^T = B^T@A^T with both packed
// operands already transposed), then widen with the FP64 alpha/beta.
static void refSgemm(double *C, const double *A, const double *B, int M,
                     int Ncols, int K, double alpha, double beta) {
  static cublasHandle_t h = nullptr;
  if (!h)
    CB(cublasCreate(&h));
  float *Af, *Bf, *Cf;
  CK(cudaMalloc(&Af, (size_t)M * K * sizeof(float)));
  CK(cudaMalloc(&Bf, (size_t)K * Ncols * sizeof(float)));
  CK(cudaMalloc(&Cf, (size_t)M * Ncols * sizeof(float)));
  narrow<<<((size_t)M * K + 255) / 256, 256>>>(Af, A, (size_t)M * K);
  narrow<<<((size_t)K * Ncols + 255) / 256, 256>>>(Bf, B, (size_t)K * Ncols);
  const float one = 1.0f, zero = 0.0f;
  CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, Ncols, M, K, &one, Bf, Ncols, Af,
                 K, &zero, Cf, Ncols));
  widen<<<((size_t)M * Ncols + 255) / 256, 256>>>(C, Cf, (size_t)M * Ncols,
                                                  alpha, beta);
  CK(cudaDeviceSynchronize());
  CK(cudaFree(Af));
  CK(cudaFree(Bf));
  CK(cudaFree(Cf));
}

static void fill(double *h, size_t n, unsigned seed) {
  srand(seed);
  for (size_t i = 0; i < n; ++i)
    h[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Bit exactness, not a tolerance: the two paths must run the same kernel on the
// same bits.
static int sameBits(const double *a, const double *b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    unsigned long long x, y;
    memcpy(&x, &a[i], 8);
    memcpy(&y, &b[i], 8);
    if (x != y) {
      printf("  first difference at %zu: %.17g vs %.17g\n", i, a[i], b[i]);
      return 0;
    }
  }
  return 1;
}

int main() {
  const int M = 97, Ncols = 64, K = 129, N = 128;

  double *hA = (double *)malloc((size_t)M * K * 8);
  double *hB = (double *)malloc((size_t)K * Ncols * 8);
  double *hC = (double *)malloc((size_t)M * Ncols * 8);
  double *hR = (double *)malloc((size_t)M * Ncols * 8);
  double *dA, *dB, *dC, *dR;

  // ---- general entry, row-major, alpha/beta not 1/0 ------------------------
  fill(hA, (size_t)M * K, 1);
  fill(hB, (size_t)K * Ncols, 2);
  fill(hC, (size_t)M * Ncols, 3);
  CK(cudaMalloc(&dA, (size_t)M * K * 8));
  CK(cudaMalloc(&dB, (size_t)K * Ncols * 8));
  CK(cudaMalloc(&dC, (size_t)M * Ncols * 8));
  CK(cudaMalloc(&dR, (size_t)M * Ncols * 8));
  CK(cudaMemcpy(dA, hA, (size_t)M * K * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, (size_t)K * Ncols * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, hC, (size_t)M * Ncols * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dR, hC, (size_t)M * Ncols * 8, cudaMemcpyHostToDevice));
  __poseidon_direct_dgemm_ex(dC, dA, dB, M, Ncols, K, K, Ncols, Ncols, 0, 0, 0,
                             2.5, 0.5, 0, /*mode=*/4);
  CK(cudaDeviceSynchronize());
  refSgemm(dR, dA, dB, M, Ncols, K, 2.5, 0.5);
  CK(cudaMemcpy(hC, dC, (size_t)M * Ncols * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hR, dR, (size_t)M * Ncols * 8, cudaMemcpyDeviceToHost));
  printf("ex %dx%dx%d alpha=2.5 beta=0.5: %s\n", M, Ncols, K,
         sameBits(hC, hR, (size_t)M * Ncols) ? "BITEXACT" : "MISMATCH");
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
  CK(cudaFree(dR));
  free(hA);
  free(hB);
  free(hC);
  free(hR);

  // ---- square entry --------------------------------------------------------
  hA = (double *)malloc((size_t)N * N * 8);
  hB = (double *)malloc((size_t)N * N * 8);
  hC = (double *)malloc((size_t)N * N * 8);
  hR = (double *)malloc((size_t)N * N * 8);
  fill(hA, (size_t)N * N, 4);
  fill(hB, (size_t)N * N, 5);
  CK(cudaMalloc(&dA, (size_t)N * N * 8));
  CK(cudaMalloc(&dB, (size_t)N * N * 8));
  CK(cudaMalloc(&dC, (size_t)N * N * 8));
  CK(cudaMalloc(&dR, (size_t)N * N * 8));
  CK(cudaMemcpy(dA, hA, (size_t)N * N * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, (size_t)N * N * 8, cudaMemcpyHostToDevice));
  __poseidon_direct_dgemm(dC, dA, dB, N, N, N, N, 0, 0, 1.0, 0.0, 0,
                          /*mode=*/4);
  CK(cudaDeviceSynchronize());
  refSgemm(dR, dA, dB, N, N, N, 1.0, 0.0);
  CK(cudaMemcpy(hC, dC, (size_t)N * N * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hR, dR, (size_t)N * N * 8, cudaMemcpyDeviceToHost));
  printf("square %d alpha=1 beta=0: %s\n", N,
         sameBits(hC, hR, (size_t)N * N) ? "BITEXACT" : "MISMATCH");

  // Mode 4 must not be mode 3: TF32 truncates the operands to 10 explicit
  // significand bits, so the two modes cannot agree on these inputs.
  __poseidon_direct_dgemm(dR, dA, dB, N, N, N, N, 0, 0, 1.0, 0.0, 0,
                          /*mode=*/3);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hR, dR, (size_t)N * N * 8, cudaMemcpyDeviceToHost));
  printf("square %d vs TF32: %s\n", N,
         sameBits(hC, hR, (size_t)N * N) ? "SAME" : "DIFFERENT");
  return 0;
}

// CHECK: ex 97x64x129 alpha=2.5 beta=0.5: BITEXACT
// CHECK: square 128 alpha=1 beta=0: BITEXACT
// CHECK: square 128 vs TF32: DIFFERENT

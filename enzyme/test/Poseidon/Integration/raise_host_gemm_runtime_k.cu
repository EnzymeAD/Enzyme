// A dense GEMM written the way a finite-element partial-assembly reduce writes
// it: runtime M / Ncols / K, the row index behind a strided
// MFEM_FOREACH_THREAD loop, the column index a thread axis fused with the CTA
// index, the contraction split across an unrolled outer level, and an
// accumulating epilogue. findScalarLoopMatmuls rejects it (no constant trip);
// findHostGemmLoopNests must recognize it and report every dimension
// symbolically, and the host dispatch must evaluate those dimensions from the
// launch arguments.
//
// Both phases go through poseidon-clang++ so the body is canonicalized the same
// way in each; a device-only -emit-llvm + opt use phase canonicalizes this loop
// nest differently from the profile-generate build and the slot indices then do
// not name the same instructions.
//
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate %s -o %t.exe
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
//
// RUN: rm -rf %t.cache && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t.profile -poseidon-cache=%t.cache \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-herbie=0 -poseidon-enable-pt=0 -poseidon-raise-wmma \
// RUN:     %s -o %t.opt.exe 2>&1 | FileCheck %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_const;
__device__ int enzyme_dup;

// y(i,q,e) += sum_{m<3} sum_{p<numPoints} Q(p,m,q,e) * G(p,m,i)
extern "C" __device__ __attribute__((noinline)) void
rt_gemm_body(int numPoints, int numEls, int nDofs, double *Q, double *G,
             double *y) {
  const int d = 3;
  const int k = blockIdx.x * blockDim.z + threadIdx.z;
  if (k >= numEls)
    return;
  const int e = k;
  for (int i = threadIdx.y; i < nDofs; i += blockDim.y) {
    for (int q = threadIdx.x; q < d; q += blockDim.x) {
      double sum = 0.;
      for (int m = 0; m < d; m++)
        for (int p = 0; p < numPoints; p++)
          sum += Q[p + numPoints * (m + d * (q + d * e))] *
                 G[p + numPoints * (m + d * i)];
      y[i + nDofs * (q + d * e)] += sum;
    }
  }
}

__global__ void rt_gemm_kernel(int numPoints, int numEls, int nDofs, double *Q,
                               double *Q_sh, double *G, double *G_sh, double *y,
                               double *y_sh) {
  __poseidon_fp_optimize<void>((void *)rt_gemm_body, enzyme_const, numPoints,
                               enzyme_const, numEls, enzyme_const, nDofs,
                               enzyme_dup, Q, Q_sh, enzyme_dup, G, G_sh,
                               enzyme_dup, y, y_sh);
}

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main(void) {
  const int numPoints = 4, numEls = 2, nDofs = 3, d = 3;
  const size_t szQ = (size_t)numPoints * d * d * numEls;
  const size_t szG = (size_t)numPoints * d * nDofs;
  const size_t szY = (size_t)nDofs * d * numEls;
  double *hQ = (double *)std::malloc(szQ * 8);
  double *hG = (double *)std::malloc(szG * 8);
  for (size_t i = 0; i < szQ; ++i)
    hQ[i] = 0.5 + 0.125 * (double)(i % 7);
  for (size_t i = 0; i < szG; ++i)
    hG[i] = 1.0 + 0.25 * (double)(i % 5);

  double *Q, *G, *Y, *Qs, *Gs, *Ys;
  CK(cudaMalloc(&Q, szQ * 8));
  CK(cudaMalloc(&G, szG * 8));
  CK(cudaMalloc(&Y, szY * 8));
  CK(cudaMalloc(&Qs, szQ * 8));
  CK(cudaMalloc(&Gs, szG * 8));
  CK(cudaMalloc(&Ys, szY * 8));
  CK(cudaMemcpy(Q, hQ, szQ * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(G, hG, szG * 8, cudaMemcpyHostToDevice));
  CK(cudaMemset(Y, 0, szY * 8));
  CK(cudaMemset(Qs, 0, szQ * 8));
  CK(cudaMemset(Gs, 0, szG * 8));
  // A zero output shadow degenerates every gradient and with it the whole
  // accuracy cost.
  {
    double *ones = (double *)std::malloc(szY * 8);
    for (size_t i = 0; i < szY; ++i)
      ones[i] = 1.0;
    CK(cudaMemcpy(Ys, ones, szY * 8, cudaMemcpyHostToDevice));
    std::free(ones);
  }
  rt_gemm_kernel<<<dim3(numEls), dim3(d, nDofs, 1)>>>(numPoints, numEls, nDofs,
                                                      Q, Qs, G, Gs, Y, Ys);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  double *hY = (double *)std::malloc(szY * 8);
  CK(cudaMemcpy(hY, Y, szY * 8, cudaMemcpyDeviceToHost));
  int nbad = 0;
  for (int e = 0; e < numEls; ++e)
    for (int q = 0; q < d; ++q)
      for (int i = 0; i < nDofs; ++i) {
        double expected = 0.0;
        for (int m = 0; m < d; ++m)
          for (int p = 0; p < numPoints; ++p)
            expected += hQ[p + numPoints * (m + d * (q + d * e))] *
                        hG[p + numPoints * (m + d * i)];
        if (std::fabs(hY[i + nDofs * (q + d * e)] - expected) > 1e-12)
          ++nbad;
      }
  if (nbad) {
    fprintf(stderr, "RTGEMM-FAIL: %d mismatches\n", nbad);
    return 1;
  }
  printf("RTGEMM-PASS\n");
  return 0;
}

// PRIMAL: RTGEMM-PASS

// The dimensions are EXPRESSIONS over the body's parameters, not integers:
// M = nDofs (arg2), Ncols = 3*numEls (arg1), K = 3*numPoints (arg0).
// CHECK: [hostgemm] {{.*}}: recognized GEMM from 3 reduction loop(s)
// CHECK-NEXT: [hostgemm]   C = alpha*A^T*B + beta*C  alpha=1{{.*}} beta=1{{.*}}
// CHECK-NEXT: [hostgemm]   M=arg2  Ncols=3*arg1  K=3*arg0
// CHECK-NEXT: [hostgemm]   lda=3*arg0  ldb=3*arg0  ldc=arg2
// CHECK-NEXT: [hostgemm]   aColMajor=0 bColMajor=1 cColMajor=1  C=param5 A=param4 B=param3
// CHECK: origin=HostGemmLoopNest

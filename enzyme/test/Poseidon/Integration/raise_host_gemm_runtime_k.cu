// A dense GEMM written the way a finite-element partial-assembly reduce writes
// it: runtime M / Ncols / K, the row index behind a strided
// MFEM_FOREACH_THREAD loop, the column index a thread axis fused with the CTA
// index, the contraction split across an unrolled outer level, and an
// accumulating epilogue. findScalarLoopMatmuls rejects it (no constant trip);
// findHostGemmLoopNests must recognize it and report every dimension
// symbolically, and the host dispatch must evaluate those dimensions from the
// launch arguments. With no target the solve takes the most accurate candidate
// that costs no more than the loop nest, cuBLAS's DGEMM, which replaces the
// launch.
//
// Both phases go through poseidon-clang++ so the body is canonicalized the same
// way in each; a device-only -emit-llvm + opt use phase canonicalizes this loop
// nest differently from the profile-generate build and the slot indices then do
// not name the same instructions.
//
// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/exe
// RUN: env POSEIDON_PROFILE_DIR=%t/profile %t/exe \
// RUN:   | FileCheck --check-prefix=PRIMAL %s
// A GPU program has no host site, so its profiling run records no allocation.
// RUN: llvm-nm --defined-only %t/exe | not grep __poseidon_host_sites_present
//
// RUN: %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-use=%t/profile -poseidon-cache=%t/cache \
// RUN:     -poseidon-cost-model=%gpu_cost_model -poseidon-print \
// RUN:     -poseidon-enable-herbie=0 -poseidon-enable-pt=0 -poseidon-raise-wmma \
// RUN:     %s -o %t/opt.exe 2>&1 | FileCheck %s
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: %t/opt.exe | FileCheck --check-prefix=PRIMAL %s

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

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

// The dimensions are expressions over the body's parameters, not integers, as
// the descriptor records them: M = nDofs (arg2), Ncols = 3*numEls (arg1),
// K = 3*numPoints (arg0).
// CHECK: [poseidon] Optimizing [[BODY:preprocess_rt_gemm_body]] with relative error tolerance: 0.000000e+00
// CHECK: [poseidon] Found 1 AbstractMatmul(s) for [[BODY]]
// CHECK-NEXT: Matmul[0]: 3x3x12 type=f64 origin=HostGemmLoopNest
// CHECK: Starting accuracy DP solver with cost budget: 0.000000e+00 of the site's profiled cost
// CHECK: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#{{[0-9]+}})
// CHECK: [poseidon] pin: rt_gemm_body:M0=native-dgemm cublas f64/f64{{$}}
// CHECK: [ozaki-host-dispatch] wrote descriptor for _Z14rt_gemm_kerneliiiPdS_S_S_S_S_: C=arg7 A=arg5 B=arg3 arg2x3*arg1x3*arg0 nm=0 (standalone){{$}}
// CHECK-NEXT: [poseidon] {{.*}}raise_host_gemm_runtime_k.cu: 1 function(s) a rewrite reached; every other function as written
// CHECK: [ozaki-host-dispatch] replaced the launch in stub _Z29__device_stub__rt_gemm_kerneliiiPdS_S_S_S_S_ with __poseidon_native_dgemm (C=arg7 A=arg5 B=arg3){{$}}

// The descriptor the stub is built from: C, A, B are launch arguments 7, 5 and
// 3, and the rd block gives M, Ncols, K, lda, ldb and ldc each as (factor
// count, argument and offset of each factor, multiplier, addend): nDofs,
// 3 numEls, 3 numPoints for K, lda and ldb, and nDofs; beta is 1.
// DESC: "host dispatch": [
// DESC-NEXT: "_Z14rt_gemm_kerneliiiPdS_S_S_S_S_ 7 5 3 1 0 0 1 1 3 rd 1 2 -1 1 0 1 1 -1 3 0 1 0 -1 3 0 1 0 -1 3 0 1 0 -1 3 0 1 2 -1 1 0 1.000000e+00"{{$}}
// DESC-NEXT: ],

// The stage-3 reduce of MFEM's elasticity apply, profiled with the launch MFEM
// makes (one 343-thread block per element) and nothing in the application
// prepared for the profile. The reverse pass caches the products' operands in
// the device heap, far more than CUDA's default 8 MiB; the profiler sizes the
// heap before the first kernel runs.
//
// RUN: rm -rf %t.cache && %poseidon_clangxx -x cuda --cuda-gpu-arch=%gpu_arch -O2 \
// RUN:     -poseidon-profile-generate -poseidon-cache=%t.cache %s -o %t.exe
// RUN: rm -rf %t.profile && POSEIDON_PROFILE_DIR=%t.profile %t.exe | FileCheck --check-prefix=PRIMAL %s
// RUN: cat %t.profile/*.fpprofile | FileCheck %s
//
// REQUIRES: poseidon, enzyme, cuda-runtime

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "poseidon/poseidon.h"

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e));                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  template <typename... I> __host__ __device__ double &operator()(I... i) const {
    const int idx[D] = {i...};
    int r = idx[D - 1];
    for (int k = D - 2; k >= 0; --k)
      r = idx[k] + sizes[k] * r;
    return data[r];
  }
};

template <typename... I> DeviceTensor<sizeof...(I)> Reshape(double *p, I... n) {
  DeviceTensor<sizeof...(I)> t{1, p, {n...}};
  for (int k = 0; k < (int)sizeof...(I); ++k)
    t.capacity *= t.sizes[k];
  return t;
}

template <typename BODY> __global__ static void CuKernel2D(const int N, BODY body) {
  const int k = blockIdx.x * blockDim.z + threadIdx.z;
  if (k >= N)
    return;
  body(k);
}

template <typename BODY> void forall_2D(int N, int X, int Y, BODY &&body) {
  CuKernel2D<<<N, dim3(X, Y, 1)>>>(N, body);
  CK(cudaGetLastError());
}

// p = 6: 343 dofs and 512 quadrature points per element.
int main() {
  constexpr int d = 3, np = 512, nd = 343, ne = 64;
  const size_t szQ = (size_t)np * d * ne, szG = (size_t)np * d * nd;
  const size_t szY = (size_t)nd * ne;
  double *hQ = (double *)std::malloc(szQ * 8), *hG = (double *)std::malloc(szG * 8);
  double *hY = (double *)std::malloc(szY * 8);
  for (size_t k = 0; k < szQ; ++k)
    hQ[k] = std::sin(0.37 * (double)k);
  for (size_t k = 0; k < szG; ++k)
    hG[k] = std::cos(0.11 * (double)k);
  double *Q, *Gp, *y;
  CK(cudaMalloc(&Q, szQ * 8));
  CK(cudaMalloc(&Gp, szG * 8));
  CK(cudaMalloc(&y, szY * 8));
  CK(cudaMemcpy(Q, hQ, szQ * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(Gp, hG, szG * 8, cudaMemcpyHostToDevice));
  CK(cudaMemset(y, 0, szY * 8));

  const auto QRead = Reshape(Q, np, d, 1, ne);
  const auto G = Reshape(Gp, np, d, nd);
  auto yDev = Reshape(y, nd, 1, ne);
  forall_2D(ne, 1, nd, [=] __host__ __device__(int e) POSEIDON_OPTIMIZE {
    for (int i = threadIdx.y; i < nd; i += blockDim.y) {
      double sum = 0.;
      for (int m = 0; m < d; m++)
        for (int p = 0; p < np; p++)
          sum += QRead(p, m, 0, e) * G(p, m, i);
      yDev(i, 0, e) += sum;
    }
  });
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hY, y, szY * 8, cudaMemcpyDeviceToHost));

  double num = 0.0, den = 0.0;
  for (int e = 0; e < ne; ++e)
    for (int i = 0; i < nd; ++i) {
      double r = 0.0;
      for (int m = 0; m < d; ++m)
        for (int p = 0; p < np; ++p)
          r += hQ[p + np * (m + d * e)] * hG[p + np * (m + d * i)];
      num += (hY[i + nd * e] - r) * (hY[i + nd * e] - r);
      den += r * r;
    }
  printf("rel err %s\n", std::sqrt(num / den) <= 1e-14 ? "ok" : "too large");
  return 0;
}

// PRIMAL: rel err ok

// CHECK: MaxBlockDims = 1 343 1
// CHECK: MaxGridDims = 64 1 1

// REQUIRES: amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t/prof
// RUN: sed "s/^CanonicalHash = .*/CanonicalHash = 35f30366bcfd6f97/" %S/../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile > %t/prof/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~643adf2c2c7efe1e.fpprofile
// RUN: not %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%t/prof -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-unit=gfx942 -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-herbie=0 %s -o %t/k.co 2>&1 | FileCheck --implicit-check-not='recognized GEMM' %s

// host_gemm_guard_nvptx.cu for gfx942: the reduce of
// annotated_lambda_host_gemm_amdgpu.hip with its store kept to the lower
// triangle, i <= e, is not recognized. The compile then stops for want of a
// gfx942 cost model, as that test's does.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

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

template <typename BODY> __global__ static void HipKernel2D(const int N, BODY body) {
  const int k = blockIdx.x * blockDim.z + threadIdx.z;
  if (k >= N)
    return;
  body(k);
}

template <typename BODY> void forall_2D(int N, int X, int Y, BODY &&body) {
  HipKernel2D<<<N, dim3(X, Y, 1)>>>(N, body);
}

// y(i,e) += sum_{m<3} sum_{p<np} Q(p,m,0,e) G(p,m,i), with Q stored with
// leading extent ldq >= np.
void reduce(int np, int ldq, int ne, int nd, double *Q, double *Gp, double *y,
            int blockY) {
  constexpr int d = 3, qSize = 1;
  const auto QRead = Reshape(Q, ldq, d, qSize, ne);
  const auto G = Reshape(Gp, np, d, nd);
  auto yDev = Reshape(y, nd, qSize, ne);
  forall_2D(ne, qSize, blockY,
            [=] __host__ __device__ POSEIDON_OPTIMIZE_TAU(1e-12)(int e) {
              for (int i = threadIdx.y; i < nd; i += blockDim.y) {
                double sum = 0.;
                for (int m = 0; m < d; m++)
                  for (int p = 0; p < np; p++)
                    sum += QRead(p, m, 0, e) * G(p, m, i);
                if (i <= e)
                  yDev(i, 0, e) += sum;
              }
            });
}

// CHECK: [poseidon] [[BODY:preprocess__ZL11HipKernel2D[^ ]+_poseidon_body]]: reduction chain of 3 loop(s) is not a dispatchable runtime-shape GEMM: the output store is gated by a condition that is not a bound of the product's indices

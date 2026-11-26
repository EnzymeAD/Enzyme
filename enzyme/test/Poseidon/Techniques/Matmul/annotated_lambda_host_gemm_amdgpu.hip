// REQUIRES: amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: not %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/../../Inputs/host_gemm_hip -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx942 -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 %s -o %t/k.co 2>&1 | FileCheck -DARCH=gfx942 %s
// RUN: not %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/../../Inputs/host_gemm_hip -Xclang -mllvm -Xclang -poseidon-cache=%t/c90a -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx90a -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 %s -o %t/k90a.co 2>&1 | FileCheck -DARCH=gfx90a %s
// RUN: not %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/../../Inputs/host_gemm_hip -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx942-foreign -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 %s -o %t/k.co 2>&1 | FileCheck --check-prefix=FOREIGN -DARCH=gfx942 %s
// RUN: not %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/../../Inputs/host_gemm_hip -Xclang -mllvm -Xclang -poseidon-cache=%t/c90a -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx90a-foreign -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 %s -o %t/k90a.co 2>&1 | FileCheck --check-prefix=FOREIGN -DARCH=gfx90a %s

// The device half of Integration/annotated_lambda_host_gemm.hip for gfx942,
// with the profile its CUDA counterpart records on an RTX 5090 (Inputs/host_gemm_hip):
// the gfx942 canonical form hashes like the NVPTX one, and the reduce is
// recognized from HIP's work-item and work-group ids and the work-group size in
// the implicit kernel arguments as the product the NVPTX compile finds. With no
// gfx942 cost model the compile refuses, and it refuses the RTX 5090's model
// for the processor the triple names. The same for gfx90a.

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
                yDev(i, 0, e) += sum;
              }
            });
}

// CHECK: [poseidon] site void HipKernel2D<reduce(int, int, int, int, double*, double*, double*, int)::'lambda'(int)>(int, reduce(int, int, int, int, double*, double*, double*, int)::'lambda'(int)): accuracy target 1e-12
// CHECK-NEXT: [poseidon] Optimizing [[BODY:preprocess__ZL11HipKernel2D[^ ]+_poseidon_body]] with relative error tolerance: 1.000000e-12
// CHECK-NEXT: [poseidon] GPU mode active for [[BODY]]
// CHECK-NEXT: [poseidon] canonical form hash of [[BODY]] is ec8a91ad004ef65f
// CHECK-NEXT: [poseidon] profile header for [[BODY]]: MaxBlockDims=(1,64,1) LaunchCount=192
// CHECK-NEXT: [poseidon] [[BODY]]: 3 candidate reduction loop(s)
// CHECK-NEXT: [poseidon] [[BODY]]: recognized GEMM from 3 reduction loop(s)
// CHECK-NEXT: [poseidon]   C = A^T*B + beta*C  beta=1.000000e+00
// CHECK-NEXT: [poseidon]   M=arg1[+0]  Ncols=arg0  K=3*arg1[+4]
// CHECK-NEXT: [poseidon]   lda=arg1[+56]*arg1[+60]  ldb=arg1[+24]*arg1[+28]*arg1[+32]  ldc=arg1[+88]*arg1[+92]
// CHECK-NEXT: [poseidon]   aColMajor=0 bColMajor=1 cColMajor=1  C=param1[+80] A=param1[+48] B=param1[+16]
// CHECK-NEXT: [poseidon]   dense only if param1[+4] == param1[+24] (checked at the launch)
// CHECK-NEXT: [poseidon]   dense only if param1[+4] == param1[+56] (checked at the launch)
// CHECK-NEXT: [poseidon]   profile shape M=192 Ncols=192 K=192 (per-CTA tile 192x1, launches=1 CTAs/launch=192 MACs=7077888)
// CHECK-NEXT: [poseidon] Found 1 matrix product(s) in [[BODY]]
// CHECK-NEXT: Matmul[0]: 192x1x192 type=f64 origin=RuntimeShape
// CHECK-NEXT: error: Poseidon: no cost model for target-cpu '[[ARCH]]' in {{.*}}/poseidon; measure the GPU with 'poseidon-calibrate --gpu <id>', or name a model with -poseidon-cost-model=<csv> or POSEIDON_COST_MODEL

// FOREIGN: error: Poseidon: cost model {{.*}}/cm_sm_120_synthetic.csv was measured on native_arch 'sm_120' but {{preprocess__ZL11HipKernel2D[^ ]+_poseidon_body}} compiles for target-cpu '[[ARCH]]'{{$}}

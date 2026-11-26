// From LLVM 22 an sm_120 compile needs the PTX version of a CUDA installation.
// DEFINE: %{compile} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-print -mllvm -poseidon-enable-rewrite=0 -mllvm -poseidon-enable-precision=0 -mllvm -poseidon-enable-matmul %s -o /dev/null
// DEFINE: %{prof} = %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile
// DEFINE: %{kernel} = _ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~0cd909cd70b69825.fpprofile
// DEFINE: %{hipkernel} = _ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~caa5c2a240f1bcf9.fpprofile
// RUN: rm -rf %t && mkdir -p %t/cols %t/guard %t/param %t/field %t/hip %t/hipguard
// RUN: %if cuda-toolkit %{ sed -e 's/^MaxGridDims = 192 1 1$/MaxGridDims = 1536 1 1/' -e 's/^LaunchCount = 192$/LaunchCount = 1536/' %{prof} > %t/cols/%{kernel} %}
// RUN: %if cuda-toolkit %{ cp %cost_models/cm_sm_120_synthetic_library.csv %t/cm.csv && echo 'library_ozaki_rect_rel,dgemm_m24n1536k192,0.05' >> %t/cm.csv %}
// RUN: %if cuda-toolkit %{ %{compile} -mllvm -poseidon-profile-use=%t/cols -mllvm -poseidon-early-prune -mllvm -poseidon-cache=%t/c -mllvm -poseidon-cost-model=%t/cm.csv 2>&1 | FileCheck --check-prefix=COLS --implicit-check-not='the native DGEMM runs at' %s %}
// RUN: %if cuda-toolkit %{ sed "s/^CanonicalHash = .*/CanonicalHash = 35f30366bcfd6f97/" %{prof} > %t/guard/%{kernel} %}
// RUN: %if cuda-toolkit %{ %{compile} -DGUARD -mllvm -poseidon-profile-use=%t/guard -mllvm -poseidon-cache=%t/cguard -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv 2>&1 | FileCheck --check-prefix=GUARD --implicit-check-not='wrote descriptor' --implicit-check-not='recognized GEMM' %s %}
// RUN: %if cuda-toolkit %{ cp %{prof} %t/param/_ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EvsT_~2beb9b9ed00c38f0.fpprofile %}
// RUN: %if cuda-toolkit %{ %{compile} -DNT=short -mllvm -poseidon-profile-use=%t/param -mllvm -poseidon-cache=%t/cparam -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv 2>&1 | FileCheck --check-prefix=PARAM --implicit-check-not='wrote descriptor' --implicit-check-not='library-call' %s %}
// RUN: %if cuda-toolkit %{ cp %{prof} %t/field/_ZL10CuKernel2DIZ6reduceiiisPdS0_S0_iEUliE_EviT_~111ca53155b4cfe8.fpprofile %}
// RUN: %if cuda-toolkit %{ %{compile} -DDT=short -mllvm -poseidon-profile-use=%t/field -mllvm -poseidon-cache=%t/cfield -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv 2>&1 | FileCheck --check-prefix=FIELD --implicit-check-not='wrote descriptor' --implicit-check-not='library-call' %s %}
// RUN: %if amdgpu-registered-target && rocm %{ cp %{prof} %t/hip/%{hipkernel} %}
// RUN: %if amdgpu-registered-target && rocm %{ not %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%t/hip -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx942 -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 %s -o %t/k.co 2>&1 | FileCheck --check-prefix=HIP -DARCH=gfx942 %s %}
// RUN: %if amdgpu-registered-target && rocm %{ not %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%t/hip -Xclang -mllvm -Xclang -poseidon-cache=%t/c90a -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx90a -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 %s -o %t/k90a.co 2>&1 | FileCheck --check-prefix=HIP -DARCH=gfx90a %s %}
// RUN: %if amdgpu-registered-target && rocm %{ not %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%t/hip -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx942-foreign -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 %s -o %t/k.co 2>&1 | FileCheck --check-prefix=HIP-FOREIGN -DARCH=gfx942 %s %}
// RUN: %if amdgpu-registered-target && rocm %{ not %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%t/hip -Xclang -mllvm -Xclang -poseidon-cache=%t/c90a -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx90a-foreign -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 %s -o %t/k90a.co 2>&1 | FileCheck --check-prefix=HIP-FOREIGN -DARCH=gfx90a %s %}
// RUN: %if amdgpu-registered-target && rocm %{ sed "s/^CanonicalHash = .*/CanonicalHash = 35f30366bcfd6f97/" %{prof} > %t/hipguard/%{hipkernel} %}
// RUN: %if amdgpu-registered-target && rocm %{ not %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%t/hipguard -Xclang -mllvm -Xclang -poseidon-cache=%t/cguard -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx942 -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 -DGUARD %s -o %t/kguard.co 2>&1 | FileCheck --check-prefix=HIP-GUARD --implicit-check-not='recognized GEMM' %s %}

// The reduce of Integration/annotated_lambda_host_gemm.hip, a product of
// run-time extents, compiled as CUDA and as HIP with the CUDA build's profile
// (Inputs/host_gemm_hip): its pricing (COLS), its AMDGPU recognition (HIP), and
// the stores and 16-bit dimensions a library call refuses.

#include "poseidon/poseidon.h"

#ifdef __HIP__
#include <hip/hip_runtime.h>
#define KERNEL2D HipKernel2D
#define TID(a) threadIdx.a
#define NTID(a) blockDim.a
#define CTAID(a) blockIdx.a
#else
#define KERNEL2D CuKernel2D
#define TID(a) __nvvm_read_ptx_sreg_tid_##a()
#define NTID(a) __nvvm_read_ptx_sreg_ntid_##a()
#define CTAID(a) __nvvm_read_ptx_sreg_ctaid_##a()
#endif
#define HD __attribute__((host, device))
#ifndef NT
#define NT int
#endif
#ifndef DT
#define DT int
#endif

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  template <typename... I> HD double &operator()(I... i) const {
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

template <typename BODY>
__attribute__((global)) static void KERNEL2D(const NT N, BODY body) {
  const int k = CTAID(x) * NTID(z) + TID(z);
  if (k >= N)
    return;
  body(k);
}

void reduce(int np, int ldq, int ne, DT nd, double *Q, double *Gp, double *y,
            int blockY) {
  constexpr int d = 3, qSize = 1;
  const auto QRead = Reshape(Q, ldq, d, qSize, ne);
  const auto G = Reshape(Gp, np, d, nd);
  auto yDev = Reshape(y, nd, qSize, ne);
  auto body = [=] HD POSEIDON_OPTIMIZE_TAU(1e-12)(int e) {
    for (int i = TID(y); i < nd; i += NTID(y)) {
      double sum = 0.;
      for (int m = 0; m < d; m++)
        for (int p = 0; p < np; p++)
          sum += QRead(p, m, 0, e) * G(p, m, i);
#ifdef GUARD
      if (i <= e)
#endif
        yDev(i, 0, e) += sum;
    }
  };
#ifdef __HIP__
  KERNEL2D<<<ne, dim3(qSize, blockY, 1)>>>(ne, body);
#else
  void *k = (void *)KERNEL2D<decltype(body)>;
  (void)k;
#endif
}

// COLS: [poseidon]   profile shape M=24 Ncols=1536 K=192 (per-CTA tile 24x1, launches=1 CTAs/launch=1536 MACs=7077888)
// COLS: Poseidon: matrix product 0, 24x1x192: ozaki-ii library call at nm=8, 9, 10, 11, 12, 13, 14 is priced as the square product it embeds in (the square row times 512.00): this device's cost model carries no measured 'library_ozaki_rect_rel,nm<nm>_m24k192' row; poseidon-calibrate --gpu <id> --ozaki-rect 24x192 measures it{{$}}
// COLS: [poseidon] matmul[0] pricing: baseline/MAC=5.146667e+02 executions=7077888
// COLS: [poseidon]   #10 native-dgemm cublas f64/f64  compCost/MAC=2.573333e+02  rel=5.000000e-01  domainError=5.102214e-16
// COLS: tau=1.000000e-12 {{.*}} rel=5.102214e-16{{$}}
// COLS: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#10)

// GUARD: [poseidon] [[BODY:preprocess__ZL10CuKernel2D[^ ]+_poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: the output store is gated by a condition that is not a bound of the product's indices

// PARAM: [poseidon] [[BODY:preprocess__ZL10CuKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EvsT__poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: a GEMM dimension depends on param0, an i16, but the host reads each launch value as an i32{{$}}
// PARAM: [poseidon] no rewrite applied for [[BODY]];
// FIELD: [poseidon] [[BODY:preprocess__ZL10CuKernel2DIZ6reduceiiisPdS0_S0_iEUliE_EviT__poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: a GEMM dimension depends on param1[+0], an i16, but the host reads each launch value as an i32{{$}}
// FIELD: [poseidon] no rewrite applied for [[BODY]];

// HIP: [poseidon] site void HipKernel2D<reduce(int, int, int, int, double*, double*, double*, int)::'lambda'(int)>(int, reduce(int, int, int, int, double*, double*, double*, int)::'lambda'(int)): accuracy target 1e-12
// HIP-NEXT: [poseidon] Optimizing [[BODY:preprocess__ZL11HipKernel2D[^ ]+_poseidon_body]] with relative error tolerance: 1.000000e-12
// HIP-NEXT: [poseidon] GPU mode active for [[BODY]]
// HIP-NEXT: [poseidon] canonical form hash of [[BODY]] is ec8a91ad004ef65f
// HIP-NEXT: [poseidon] profile header for [[BODY]]: MaxBlockDims=(1,64,1) LaunchCount=192
// HIP-NEXT: [poseidon] [[BODY]]: 3 candidate reduction loop(s)
// HIP-NEXT: [poseidon] [[BODY]]: recognized GEMM from 3 reduction loop(s)
// HIP-NEXT: [poseidon]   C = A^T*B + beta*C  beta=1.000000e+00
// HIP-NEXT: [poseidon]   M=arg1[+0]  Ncols=arg0  K=3*arg1[+4]
// HIP-NEXT: [poseidon]   lda=arg1[+56]*arg1[+60]  ldb=arg1[+24]*arg1[+28]*arg1[+32]  ldc=arg1[+88]*arg1[+92]
// HIP-NEXT: [poseidon]   aColMajor=0 bColMajor=1 cColMajor=1  C=param1[+80] A=param1[+48] B=param1[+16]
// HIP-NEXT: [poseidon]   dense only if param1[+4] == param1[+24] (checked at the launch)
// HIP-NEXT: [poseidon]   dense only if param1[+4] == param1[+56] (checked at the launch)
// HIP-NEXT: [poseidon]   profile shape M=192 Ncols=192 K=192 (per-CTA tile 192x1, launches=1 CTAs/launch=192 MACs=7077888)
// HIP-NEXT: [poseidon] Found 1 matrix product(s) in [[BODY]]
// HIP-NEXT: Matmul[0]: 192x1x192 type=f64 origin=RuntimeShape
// HIP-NEXT: error: Poseidon: no cost model for target-cpu '[[ARCH]]' in {{.*}}/poseidon; measure the GPU with 'poseidon-calibrate --gpu <id>', or name a model with -poseidon-cost-model=<csv> or POSEIDON_COST_MODEL

// HIP-FOREIGN: error: Poseidon: cost model {{.*}}/cm_sm_120_synthetic.csv was measured on native_arch 'sm_120' but {{preprocess__ZL11HipKernel2D[^ ]+_poseidon_body}} compiles for target-cpu '[[ARCH]]'{{$}}

// HIP-GUARD: [poseidon] [[BODY:preprocess__ZL11HipKernel2D[^ ]+_poseidon_body]]: reduction chain of 3 loop(s) is not a library-call runtime-shape product: the output store is gated by a condition that is not a bound of the product's indices

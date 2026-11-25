// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print %s -o %t/use.ll 2>&1 | FileCheck --check-prefix=USE %s
// RUN: not %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/wrapper -DNOT_A_WRAPPER %s -o /dev/null 2>&1 | FileCheck --check-prefix=WRAPPER %s
// RUN: not %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -DNOT_A_WRAPPER %s -o /dev/null 2>&1 | FileCheck --check-prefix=WRAPPER %s

// annotated_lambda_nvptx.cu for HIP: POSEIDON_OPTIMIZE on the lambda an
// mfem::forall body is written as makes every AMDGPU kernel that calls it a
// site, with the lambda's target, and each data pointer of the by-value
// aggregate the lambda reaches the kernel as is shadowed.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#define HD __host__ __device__
#define FOREACH_Y(i, N) for (int i = threadIdx.y; i < N; i += blockDim.y)

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  HD double &operator()(int a, int b) const { return data[a + sizes[0] * b]; }
};

template <typename BODY> __global__ static void HipKernel2D(int N, BODY body) {
  int k = blockIdx.x * blockDim.z + threadIdx.z;
  if (k >= N)
    return;
  body(k);
}

// A kernel that computes more than the lambda cannot be its site.
// WRAPPER: annotated_lambda_amdgpu.hip:[[@LINE+4]]:{{[0-9]+}}: error: Poseidon: void Scaled<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, double*, reduce(int, int, int, double*, double*, double*)::'lambda'(int)) calls reduce(int, int, int, double*, double*, double*)::'lambda'(int)::operator()(int) const, which carry POSEIDON_OPTIMIZE, and does floating-point work outside them (in void Scaled<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, double*, reduce(int, int, int, double*, double*, double*)::'lambda'(int))); a kernel is the site of an annotated function only when it is a wrapper around the call. Move that work out of the kernel, or annotate the kernel instead.
// WRAPPER-NOT: error:
// WRAPPER: {{[12] warnings? and 1 error}} generated when compiling for {{.*}}
template <typename BODY>
__global__ static void Scaled(int N, double *s, BODY body) {
  int k = blockIdx.x;
  if (k >= N)
    return;
  s[k] = s[k] * 2.0;
  body(k);
}

void reduce(int nq, int ne, int nd, double *Q, double *G, double *y) {
  DeviceTensor<2> QR{nq * ne, Q, {nq, ne}};
  DeviceTensor<2> GR{nq * nd, G, {nq, nd}};
  DeviceTensor<2> Y{nd * ne, y, {nd, ne}};
  auto body = [=] HD POSEIDON_OPTIMIZE_TAU(1e-9) (int e) {
    FOREACH_Y(i, nd) {
      double sum = 0.0;
      for (int p = 0; p < nq; ++p)
        sum += QR(p, e) * GR(p, i);
      Y(i, e) += sum;
    }
  };
  void *k1 = (void *)HipKernel2D<decltype(body)>;
  (void)k1;
#ifdef NOT_A_WRAPPER
  void *k2 = (void *)Scaled<decltype(body)>;
  (void)k2;
#endif
}

// An annotated device function no kernel calls is not a site.
POSEIDON_OPTIMIZE __device__ double unused(double x) { return x * x; }

// GEN: annotated_lambda_amdgpu.hip:[[@LINE-2]]:{{[0-9]+}}: warning: Poseidon: unused(double) is a device function no kernel of this translation unit calls; on the device, POSEIDON_OPTIMIZE marks a kernel or a function a kernel calls. Emitting the original computation.
// GEN: [poseidon] block-geometry probe placed in 1 profiled function(s)
// GEN-NEXT: [poseidon-profgen] _ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_: site 0, 3 shadow buffer(s), output(s) 1[+64]
// GEN-NOT: [poseidon-profgen] {{.*}}Scaled
// GEN: 1 warning generated when compiling for {{.*}}

// The kernel, its two launch arguments, its three pointers (all in argument
// 1), the eight-byte seed of the one written, and where each sits in the
// lambda: the layout is NVPTX's.
// DESC: "host dispatch": [],
// DESC-NEXT: "profile generation": [
// DESC-NEXT: "_ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_ 0 2 3 1 1 1 0 0 8 off 16 40 64"
// DESC-NEXT: ],
// DESC-NEXT: "profile headers": {
// DESC-NEXT: "_ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_": {
// DESC-NEXT: "header": "SiteId = 0\nCanonicalHash = {{[0-9a-f]+}}\n",
// DESC-NEXT: "profile": "preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body"

// The kernel keeps its calling convention and gains the three shadows, in the
// order of the pointers they shadow: the adjoints of Q and G accumulate into
// the first two, and the one of y is read from the third, cleared, and logged
// as the seed of both operations.
// GENIR: @poseidon_site_preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body = private unnamed_addr addrspace(1) constant [73 x i8] c"preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body\00"
// GENIR: define {{.*}}amdgpu_kernel void @_ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(i32 {{.*}}, ptr addrspace(4) {{.*}}byref(%class.anon){{.*}}, ptr {{.*}}[[DQ:%[0-9]+]], ptr {{.*}}[[DG:%[0-9]+]], ptr {{.*}}[[DY:%[0-9]+]])
// GENIR-DAG: [[DQG:%[0-9]+]] = addrspacecast ptr [[DQ]] to ptr addrspace(1)
// GENIR-DAG: [[DGG:%[0-9]+]] = addrspacecast ptr [[DG]] to ptr addrspace(1)
// GENIR-DAG: [[DYG:%[0-9]+]] = addrspacecast ptr [[DY]] to ptr addrspace(1)
// GENIR-DAG: call void @poseidonProfileStaticCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), ptr addrspacecast (ptr addrspace(1) @fpprofile_static_preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr))
// GENIR-DAG: [[DQI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DQG]], i64 {{%[0-9]+}}
// GENIR-DAG: atomicrmw fadd ptr addrspace(1) [[DQI]], double {{%[0-9]+}} monotonic
// GENIR-DAG: [[DGI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DGG]], i64 {{%[0-9]+}}
// GENIR-DAG: atomicrmw fadd ptr addrspace(1) [[DGI]], double {{%[0-9]+}} monotonic
// GENIR-DAG: [[DYI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DYG]], i64 {{%[0-9]+}}
// GENIR-DAG: [[ADJ:%[0-9]+]] = load double, ptr addrspace(1) [[DYI]]
// GENIR-DAG: store double 0.000000e+00, ptr addrspace(1) [[DYI]]
// GENIR-DAG: call void @poseidonLogGradCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), i64 0, double {{%[0-9]+}}, double [[ADJ]])
// GENIR-DAG: call void @poseidonLogGradCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), i64 1, double {{%[0-9]+}}, double [[ADJ]])
// GENIR: {{^}}}

// USE: [poseidon] site void HipKernel2D<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, reduce(int, int, int, double*, double*, double*)::'lambda'(int)): accuracy target 1e-09
// USE-NEXT: annotated_lambda_amdgpu.hip:[[@LINE-83]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body.fpprofile: the profiled program did not contain this site (it was annotated after the profiling run, is in a translation unit the profiled program did not link, or was renamed); leaving the site unchanged
// USE: [poseidon] _ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body: the profiled program did not contain this site (no profile at {{.*}}/Inputs/other_sites/preprocess__ZL11HipKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body.fpprofile); left unchanged
// USE-NEXT: [poseidon] every site of {{.*}}annotated_lambda_amdgpu.hip was left alone; emitting the module as written
// USE-NEXT: 2 warnings generated when compiling for {{.*}}

// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// DEFINE: %{nvptx} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc
// DEFINE: %{gen} = %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate
// DEFINE: %{use} = %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile
// RUN: rm -rf %t && mkdir %t
// RUN: %if nvptx-device-profiler %{ %{nvptx} %{gen} -mllvm -poseidon-cache=%t/nvptx -mllvm -poseidon-print %s -o %t/nvptx.ll 2>&1 | FileCheck --check-prefix=GEN %s %}
// RUN: %if nvptx-device-profiler %{ cat %t/nvptx/launches_*.json | FileCheck --check-prefix=DESC %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=GENIR-NVPTX %s < %t/nvptx.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} %{use} -mllvm -poseidon-print %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s %}
// RUN: %if nvptx-device-profiler %{ not %{nvptx} %{gen} -mllvm -poseidon-cache=%t/nvptx-wrapper -DNOT_A_WRAPPER %s -o /dev/null 2>&1 | FileCheck --check-prefix=WRAPPER %s %}
// RUN: %if nvptx-device-profiler %{ not %{nvptx} %{use} -DNOT_A_WRAPPER %s -o /dev/null 2>&1 | FileCheck --check-prefix=WRAPPER %s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %{gen} -mllvm -poseidon-cache=%t/amdgpu -mllvm -poseidon-print %s -o %t/amdgpu.ll 2>&1 | FileCheck --check-prefix=GEN %s %}
// RUN: %if amdgpu-registered-target && rocm %{ cat %t/amdgpu/launches_*.json | FileCheck --check-prefix=DESC %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=GENIR-AMDGPU %s < %t/amdgpu.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %{use} -mllvm -poseidon-print %s -o /dev/null 2>&1 | FileCheck --check-prefix=USE %s %}
// RUN: %if amdgpu-registered-target && rocm %{ not %{amdgpu} %{gen} -mllvm -poseidon-cache=%t/amdgpu-wrapper -DNOT_A_WRAPPER %s -o /dev/null 2>&1 | FileCheck --check-prefix=WRAPPER %s %}
// RUN: %if amdgpu-registered-target && rocm %{ not %{amdgpu} %{use} -DNOT_A_WRAPPER %s -o /dev/null 2>&1 | FileCheck --check-prefix=WRAPPER %s %}

// POSEIDON_OPTIMIZE on a forall lambda makes every kernel that calls it a site
// with the lambda's target; profile generation shadows each data pointer field
// of the by-value lambda and seeds the one written.

#include "poseidon/poseidon.h"

#ifdef __NVPTX__
#define THREAD_Y __nvvm_read_ptx_sreg_tid_y()
#define THREAD_Z __nvvm_read_ptx_sreg_tid_z()
#define BLOCK_DIM_Y __nvvm_read_ptx_sreg_ntid_y()
#define BLOCK_DIM_Z __nvvm_read_ptx_sreg_ntid_z()
#define BLOCK_ID_X __nvvm_read_ptx_sreg_ctaid_x()
#else
#define THREAD_Y __builtin_amdgcn_workitem_id_y()
#define THREAD_Z __builtin_amdgcn_workitem_id_z()
#define BLOCK_DIM_Y __builtin_amdgcn_workgroup_size_y()
#define BLOCK_DIM_Z __builtin_amdgcn_workgroup_size_z()
#define BLOCK_ID_X __builtin_amdgcn_workgroup_id_x()
#endif

#define HD __attribute__((host, device))
#define FOREACH_Y(i, N) for (int i = THREAD_Y; i < N; i += BLOCK_DIM_Y)

template <int D> struct DeviceTensor {
  int capacity;
  double *data;
  int sizes[D];
  HD double &operator()(int a, int b) const { return data[a + sizes[0] * b]; }
};

template <typename BODY>
__attribute__((global)) static void Kernel2D(int N, BODY body) {
  int k = BLOCK_ID_X * BLOCK_DIM_Z + THREAD_Z;
  if (k >= N)
    return;
  body(k);
}

// A kernel that computes more than the lambda cannot be its site.
// WRAPPER: annotated_lambda.cu:[[@LINE+4]]:{{[0-9]+}}: error: Poseidon: void Scaled<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, double*, reduce(int, int, int, double*, double*, double*)::'lambda'(int)) calls reduce(int, int, int, double*, double*, double*)::'lambda'(int)::operator()(int) const, which carry POSEIDON_OPTIMIZE, and does floating-point work outside them (in void Scaled<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, double*, reduce(int, int, int, double*, double*, double*)::'lambda'(int))), so it is no wrapper around the call and not their site; move that work out of the kernel, or annotate the kernel instead
// WRAPPER-NOT: error:
// WRAPPER: {{^}}1 error generated when compiling for {{.*}}
template <typename BODY>
__attribute__((global)) static void Scaled(int N, double *s, BODY body) {
  int k = BLOCK_ID_X;
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
  void *k1 = (void *)Kernel2D<decltype(body)>;
  (void)k1;
#ifdef NOT_A_WRAPPER
  void *k2 = (void *)Scaled<decltype(body)>;
  (void)k2;
#endif
}

// An annotated device function no kernel calls is not a site.
POSEIDON_OPTIMIZE __attribute__((device)) double unused(double x) {
  return x * x;
}

// GEN: annotated_lambda.cu:[[@LINE-4]]:{{[0-9]+}}: warning: Poseidon: unused(double) is a device function no kernel of this translation unit calls, so it is left as written; on the device, put POSEIDON_OPTIMIZE on a kernel or on a function a kernel calls
// GEN: [poseidon] block-geometry probe placed in 1 profiled function(s)
// GEN-NEXT: [poseidon] _ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_: site 0, 3 shadow buffer(s), output(s) 1[+64]
// GEN-NOT: [poseidon] {{.*}}Scaled
// GEN: 1 warning generated when compiling for {{.*}}

// The descriptor: three pointers in argument 1, the seed of the one written,
// their offsets in the lambda and their traffic records, alike on both targets.
// DESC: "library calls": [],
// DESC-NEXT: "profile generation": [
// DESC-NEXT: "_ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_ 0 2 3 1 1 1 0 0 8 off 16 40 64"
// DESC-NEXT: ],
// DESC-NEXT: "profile headers": {
// DESC-NEXT: "_ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_": {
// DESC-NEXT: "header": "SiteId = 0\nCanonicalHash = {{[0-9a-f]+}}\nTrafficArray = 0 1 64\nTrafficArray = 1 1 16\nTrafficArray = 2 1 40\n",
// DESC-NEXT: "profile": "_ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_~473a3bf0e0637c14"

// The kernel gains the three shadows and passes the body a copy of the lambda
// with each data pointer replaced by its shadow: the adjoints of Q and G
// accumulate, the one of y is read, cleared and logged as both seeds.
// GENIR-NVPTX: @poseidon.site.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body = private unnamed_addr addrspace(1) constant [61 x i8] c"_ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_~473a3bf0e0637c14\00"
// GENIR-NVPTX: define ptx_kernel void @_ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(i32 [[N:%[0-9]+]], ptr {{.*}}byval(%class.anon){{.*}} [[L:%[0-9]+]], ptr [[DQ:%[0-9]+]], ptr [[DG:%[0-9]+]], ptr [[DY:%[0-9]+]])
// GENIR-NVPTX-NEXT: [[SL:%[0-9]+]] = alloca %class.anon
// GENIR-NVPTX-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[SL]], ptr {{.*}}[[L]], i64 80, i1 false)
// GENIR-NVPTX-NEXT: [[FQ:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SL]], i64 16
// GENIR-NVPTX-NEXT: store ptr [[DQ]], ptr [[FQ]]
// GENIR-NVPTX-NEXT: [[FG:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SL]], i64 40
// GENIR-NVPTX-NEXT: store ptr [[DG]], ptr [[FG]]
// GENIR-NVPTX-NEXT: [[FY:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SL]], i64 64
// GENIR-NVPTX-NEXT: store ptr [[DY]], ptr [[FY]]
// GENIR-NVPTX-NEXT: call fastcc void @diffepreprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body(i32 [[N]], ptr {{.*}}[[L]], ptr [[SL]])
// GENIR-NVPTX-NEXT: ret void
// GENIR-NVPTX: define internal fastcc void @diffepreprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body(i32 {{.*}}%{{[0-9]+}}, ptr {{.*}}%{{[0-9]+}}, ptr {{.*}}[[S:%[0-9]+]])
// GENIR-NVPTX-DAG: call void @__poseidon_device_publish_static(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), ptr addrspacecast (ptr addrspace(1) @poseidon.static.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr))
// GENIR-NVPTX-DAG: [[SQP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[S]], i64 16
// GENIR-NVPTX-DAG: [[SQ:%[0-9]+]] = load ptr, ptr [[SQP]]
// GENIR-NVPTX-DAG: [[SGP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[S]], i64 40
// GENIR-NVPTX-DAG: [[SG:%[0-9]+]] = load ptr, ptr [[SGP]]
// GENIR-NVPTX-DAG: [[SYP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[S]], i64 64
// GENIR-NVPTX-DAG: [[SY:%[0-9]+]] = load ptr, ptr [[SYP]]
// GENIR-NVPTX-DAG: [[DQI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SQ]], i64 {{%[0-9]+}}
// GENIR-NVPTX-DAG: atomicrmw fadd ptr [[DQI]], double {{%[0-9]+}} monotonic
// GENIR-NVPTX-DAG: [[DGI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SG]], i64 {{%[0-9]+}}
// GENIR-NVPTX-DAG: atomicrmw fadd ptr [[DGI]], double {{%[0-9]+}} monotonic
// GENIR-NVPTX-DAG: [[DYI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[SY]], i64 {{%[0-9]+}}
// GENIR-NVPTX-DAG: [[ADJ:%[0-9]+]] = load double, ptr [[DYI]]
// GENIR-NVPTX-DAG: store double 0.000000e+00, ptr [[DYI]]
// GENIR-NVPTX-DAG: call void @__poseidon_device_log_grad(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), i64 0, double {{%[0-9]+}}, double [[ADJ]])
// GENIR-NVPTX-DAG: atomicrmw fadd ptr [[DYI]], double [[ADJ]] monotonic
// GENIR-NVPTX-DAG: call void @__poseidon_device_log_grad(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), i64 1, double {{%[0-9]+}}, double [[ADJ]])
// GENIR-NVPTX: {{^}}}

// The same on AMDGPU, where the kernel keeps its calling convention.
// GENIR-AMDGPU: @poseidon.site.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body = private unnamed_addr addrspace(1) constant [61 x i8] c"_ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_~473a3bf0e0637c14\00"
// GENIR-AMDGPU: define {{.*}}amdgpu_kernel void @_ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(i32 {{.*}}, ptr addrspace(4) {{.*}}byref(%class.anon){{.*}}, ptr {{.*}}[[DQ:%[0-9]+]], ptr {{.*}}[[DG:%[0-9]+]], ptr {{.*}}[[DY:%[0-9]+]])
// GENIR-AMDGPU-DAG: [[DQG:%[0-9]+]] = addrspacecast ptr [[DQ]] to ptr addrspace(1)
// GENIR-AMDGPU-DAG: [[DGG:%[0-9]+]] = addrspacecast ptr [[DG]] to ptr addrspace(1)
// GENIR-AMDGPU-DAG: [[DYG:%[0-9]+]] = addrspacecast ptr [[DY]] to ptr addrspace(1)
// GENIR-AMDGPU-DAG: call void @__poseidon_device_publish_static(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), ptr addrspacecast (ptr addrspace(1) @poseidon.static.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr))
// GENIR-AMDGPU-DAG: [[DQI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DQG]], i64 {{%[0-9]+}}
// GENIR-AMDGPU-DAG: atomicrmw fadd ptr addrspace(1) [[DQI]], double {{%[0-9]+}} monotonic
// GENIR-AMDGPU-DAG: [[DGI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DGG]], i64 {{%[0-9]+}}
// GENIR-AMDGPU-DAG: atomicrmw fadd ptr addrspace(1) [[DGI]], double {{%[0-9]+}} monotonic
// GENIR-AMDGPU-DAG: [[DYI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DYG]], i64 {{%[0-9]+}}
// GENIR-AMDGPU-DAG: [[ADJ:%[0-9]+]] = load double, ptr addrspace(1) [[DYI]]
// GENIR-AMDGPU-DAG: store double 0.000000e+00, ptr addrspace(1) [[DYI]]
// GENIR-AMDGPU-DAG: call void @__poseidon_device_log_grad(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), i64 0, double {{%[0-9]+}}, double [[ADJ]])
// GENIR-AMDGPU-DAG: atomicrmw fadd ptr addrspace(1) [[DYI]], double [[ADJ]] monotonic
// GENIR-AMDGPU-DAG: call void @__poseidon_device_log_grad(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT__poseidon_body to ptr), i64 1, double {{%[0-9]+}}, double [[ADJ]])
// GENIR-AMDGPU: {{^}}}

// USE: [poseidon] site void Kernel2D<reduce(int, int, int, double*, double*, double*)::'lambda'(int)>(int, reduce(int, int, int, double*, double*, double*)::'lambda'(int)): accuracy target 1e-09
// USE-NEXT: annotated_lambda.cu:[[@LINE-117]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_ZL8Kernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_~473a3bf0e0637c14.fpprofile:
// USE: [poseidon] every site of {{.*}}annotated_lambda.cu was left alone; emitting the module as written
// USE-NEXT: 2 warnings generated when compiling for {{.*}}

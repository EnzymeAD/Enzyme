// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// DEFINE: %{nvptx} = %clang -x cuda --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %s
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -cuid=0 -O1 -S -emit-llvm -I%poseidon_profile_inc %s
// RUN: rm -rf %t && mkdir %t
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-device-only %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/nvptx-c -o %t/nvptx-gen.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=GEN,GEN-NVPTX --implicit-check-not=__poseidon_site_explicit %s < %t/nvptx-gen.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=NOMARKER-NVPTX --implicit-check-not=__poseidon_site_explicit %s < %t/nvptx-gen.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-device-only %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-print -o %t/nvptx-use.ll 2>&1 | FileCheck --check-prefix=USE %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=IR,IR-NVPTX --implicit-check-not=__poseidon_site_explicit %s < %t/nvptx-use.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-device-only -o - | tee %t/nvptx-plain.ll | FileCheck --check-prefixes=PLAIN,PLAIN-NVPTX %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=DECL --implicit-check-not='define {{.*}}@__poseidon_site_explicit' %s < %t/nvptx-plain.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} --cuda-host-only %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -o %t/nvptx-host.ll 2>&1 | FileCheck --allow-empty --check-prefix=QUIET %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=HOST --implicit-check-not=__poseidon_site_explicit --implicit-check-not=__poseidon_fp_optimize %s < %t/nvptx-host.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --cuda-device-only %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/amdgpu-c -o %t/amdgpu-gen.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=GEN,GEN-AMDGPU --implicit-check-not=__poseidon_site_explicit %s < %t/amdgpu-gen.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=NOMARKER-AMDGPU --implicit-check-not=__poseidon_site_explicit %s < %t/amdgpu-gen.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --cuda-device-only %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-print -o %t/amdgpu-use.ll 2>&1 | FileCheck --check-prefix=USE %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=IR,IR-AMDGPU --implicit-check-not=__poseidon_site_explicit %s < %t/amdgpu-use.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --cuda-device-only -o - | tee %t/amdgpu-plain.ll | FileCheck --check-prefixes=PLAIN,PLAIN-AMDGPU %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=DECL --implicit-check-not='define {{.*}}@__poseidon_site_explicit' %s < %t/amdgpu-plain.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --cuda-host-only %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -o %t/amdgpu-host.ll 2>&1 | FileCheck --allow-empty --check-prefix=QUIET %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=HOST --implicit-check-not=__poseidon_site_explicit --implicit-check-not=__poseidon_fp_optimize %s < %t/amdgpu-host.ll %}

// In device code an explicit list through __poseidon_optimize makes the call
// a site over the function itself, with the shadows the caller gives: here
// sum reaches no constant range of memory through x, which the pass could not
// make the shadow of. __poseidon_optimize_tau with the function's own
// arguments is no explicit list: that call is a per-call site. Without the
// plugin the call goes through an identity nothing defines. The host half of a
// host-device function holds no site: there the call runs the function with
// the primal values, silently.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))
// Device functions are internal in AMDGPU device code: twice and third, which
// the kernel inlines, are kept as on NVPTX.
#define KEEP __attribute__((used))

// The runtime declarations a kernel stub needs.
struct dim3 {
  unsigned x, y, z;
};
#ifdef __HIP__
typedef struct ihipStream_t *hipStream_t;
extern "C" int hipLaunchKernel(const void *, dim3, dim3, void **,
                               unsigned long, hipStream_t);
#else
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
__attribute__((device)) int enzyme_dup, enzyme_const, poseidon_tau;
#else
int enzyme_dup, enzyme_const, poseidon_tau;
#endif

__attribute__((device)) double sum(const double *x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += x[i] * x[i];
  return s;
}

__attribute__((device)) void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = x[i] * x[i] + 1.0;
}

HD __attribute__((noinline)) double norm2(const double *x) {
  return x[0] * x[0] + x[1] * x[1];
}

HD KEEP double twice(const double *x, double *dx) {
  return 2.0 * __poseidon_optimize(norm2, enzyme_dup, x, dx);
}

HD __attribute__((noinline)) double spread(const double *x) {
  return x[0] * x[1] - x[1] * x[1];
}

HD KEEP double third(const double *x) {
  return __poseidon_optimize_tau(1e-7, spread, x);
}

__attribute__((global)) void k(const double *x, double *dx, double *y,
                               double *dy, int n) {
  y[0] = __poseidon_optimize(sum, enzyme_dup, x, dx, n);
  y[3] = twice(x, dx);
  __poseidon_optimize_tau(1e-6, scale, enzyme_const, x, enzyme_dup, y + 1, dy,
                          n);
  y[2] = __poseidon_optimize_tau(1e-9, sum, enzyme_const, x, n);
  y[4] = third(x);
}

// Profile generation leaves no call of the identity anywhere in the module.
// NOMARKER-NVPTX: target triple = "nvptx64-nvidia-cuda"
// NOMARKER-AMDGPU: target triple = "{{(amdgcn|amdgpu)[^"]*}}-amd-amdhsa"

// GEN-DAG: @poseidon.site.preprocess__Z3sumPKdi = private unnamed_addr addrspace(1) constant [11 x i8] c"_Z3sumPKdi\00"
// GEN-DAG: @poseidon.site.preprocess__Z5scalePKdPdi = private unnamed_addr addrspace(1) constant [15 x i8] c"_Z5scalePKdPdi\00"
// GEN-DAG: @poseidon.site.preprocess__Z5norm2PKd = private unnamed_addr addrspace(1) constant [12 x i8] c"_Z5norm2PKd\00"
// GEN-DAG: @poseidon.site.preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0 = private unnamed_addr addrspace(1) constant [36 x i8] c"_Z6spreadPKd.poseidon._Z5thirdPKd.0\00"

// Each site's adjoint reaches the shadow the call gives it: dx for sum and
// norm2, dy for scale, and a zeroed local the pass makes for spread's x. The
// kernel holds norm2's call on both of its paths around the loop of sum.
// GEN: define {{.*}} @_Z5twicePKdPd(ptr {{.*}}[[TX:%[0-9]+]], ptr {{.*}}[[TDX:%[0-9]+]])
// GEN-NEXT: [[TR:%[0-9]+]] = {{.*}}call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr [[TX]], ptr [[TDX]])
// GEN-NEXT: [[TT:%[0-9]+]] = fmul {{.*}}double [[TR]], 2.000000e+00
// GEN-NEXT: ret double [[TT]]
// GEN-NVPTX: define {{.*}} @_Z5thirdPKd(ptr {{.*}}[[HX:%[0-9]+]])
// GEN-NVPTX-NEXT: [[HSH:%[0-9]+]] = alloca [16 x i8]
// GEN-NVPTX-NEXT: call void @llvm.memset.p0.i64(ptr {{.*}}[[HSH]], i8 0, i64 16, i1 false)
// GEN-NVPTX-NEXT: [[HR:%[0-9]+]] = {{.*}}call {{.*}}double @diffepreprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0(ptr [[HX]], ptr [[HSH]])
// GEN-NVPTX-NEXT: ret double [[HR]]
// GEN-AMDGPU: define {{.*}} @_Z5thirdPKd(ptr {{.*}}[[HX:%[0-9]+]])
// GEN-AMDGPU-NEXT: [[HSH:%[0-9]+]] = alloca [16 x i8], align 8, addrspace(5)
// GEN-AMDGPU-NEXT: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}[[HSH]], i8 0, i64 16, i1 false)
// GEN-AMDGPU-NEXT: [[HSHG:%[0-9]+]] = addrspacecast ptr addrspace(5) [[HSH]] to ptr
// GEN-AMDGPU-NEXT: [[HR:%[0-9]+]] = {{.*}}call {{.*}}double @diffepreprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0(ptr [[HX]], ptr [[HSHG]])
// GEN-AMDGPU-NEXT: ret double [[HR]]
// GEN-NVPTX: define {{.*}}ptx_kernel void @_Z1kPKdPdS1_S1_i(ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}[[DX:%[0-9]+]], ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// GEN-NVPTX-DAG: call void @__poseidon_device_publish_static(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z3sumPKdi to ptr), ptr addrspacecast (ptr addrspace(1) @poseidon.static.preprocess__Z3sumPKdi to ptr))
// GEN-NVPTX-DAG: [[DXI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[DX]], i64 {{%[0-9]+}}
// GEN-NVPTX-DAG: atomicrmw fadd ptr [[DXI]], double {{%[0-9]+}} monotonic
// GEN-NVPTX-DAG: call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr [[X]], ptr [[DX]])
// GEN-NVPTX-DAG: call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr [[X]], ptr [[DX]])
// GEN-NVPTX-DAG: call void @__poseidon_device_publish_static(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z5scalePKdPdi to ptr), ptr addrspacecast (ptr addrspace(1) @poseidon.static.preprocess__Z5scalePKdPdi to ptr))
// GEN-NVPTX-DAG: [[DYI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[DY]], i64 {{%[0-9]+}}
// GEN-NVPTX-DAG: [[DYV:%[0-9]+]] = load double, ptr [[DYI]]
// GEN-NVPTX-DAG: store double 0.000000e+00, ptr [[DYI]]
// GEN-NVPTX-DAG: call void @__poseidon_device_log_grad(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z5scalePKdPdi to ptr), i64 0, double {{%[0-9]+}}, double [[DYV]])
// GEN-NVPTX-DAG: call void @llvm.memset.p0.i64(ptr {{.*}}[[SHA:%[0-9]+]], i8 0, i64 16, i1 false)
// GEN-NVPTX-DAG: [[SHA]] = alloca [16 x i8]
// GEN-NVPTX-DAG: call {{.*}}double @diffepreprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0(ptr [[X]], ptr [[SHA]])
// GEN-NVPTX: {{^}}}

// The kernel on AMDGPU, whose pointer parameters are in the global address
// space.
// GEN-AMDGPU: define protected amdgpu_kernel void @_Z1kPKdPdS1_S1_i(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}[[DX:%[0-9]+]], ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// GEN-AMDGPU-DAG: call void @__poseidon_device_publish_static(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z3sumPKdi to ptr), ptr addrspacecast (ptr addrspace(1) @poseidon.static.preprocess__Z3sumPKdi to ptr))
// GEN-AMDGPU-DAG: [[DXI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DX]], i64 {{%[0-9]+}}
// GEN-AMDGPU-DAG: atomicrmw fadd ptr addrspace(1) [[DXI]], double {{%[0-9]+}} monotonic
// GEN-AMDGPU-DAG: call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr {{.*}}[[XN:%[0-9]+]], ptr [[DXN:%[0-9]+]])
// GEN-AMDGPU-DAG: [[XN]] = addrspacecast ptr addrspace(1) [[X]] to ptr
// GEN-AMDGPU-DAG: [[DXN]] = addrspacecast ptr addrspace(1) [[DX]] to ptr
// GEN-AMDGPU-DAG: call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr {{.*}}[[XN2:%[0-9]+]], ptr [[DXN2:%[0-9]+]])
// GEN-AMDGPU-DAG: [[XN2]] = addrspacecast ptr addrspace(1) [[X]] to ptr
// GEN-AMDGPU-DAG: [[DXN2]] = addrspacecast ptr addrspace(1) [[DX]] to ptr
// GEN-AMDGPU-DAG: call void @__poseidon_device_publish_static(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z5scalePKdPdi to ptr), ptr addrspacecast (ptr addrspace(1) @poseidon.static.preprocess__Z5scalePKdPdi to ptr))
// GEN-AMDGPU-DAG: [[DYI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DY]], i64 {{%[0-9]+}}
// GEN-AMDGPU-DAG: [[DYV:%[0-9]+]] = load double, ptr addrspace(1) [[DYI]]
// GEN-AMDGPU-DAG: store double 0.000000e+00, ptr addrspace(1) [[DYI]]
// GEN-AMDGPU-DAG: call void @__poseidon_device_log_grad(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z5scalePKdPdi to ptr), i64 0, double {{%[0-9]+}}, double [[DYV]])
// GEN-AMDGPU-DAG: call {{.*}}double @diffepreprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0(ptr [[XS:%[0-9]+]], ptr [[SH:%[0-9]+]])
// GEN-AMDGPU-DAG: [[XS]] = addrspacecast ptr addrspace(1) [[X]] to ptr
// GEN-AMDGPU-DAG: [[SH]] = addrspacecast ptr addrspace(5) [[SHA:%[0-9]+]] to ptr
// GEN-AMDGPU-DAG: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}[[SHA]], i8 0, i64 16, i1 false)
// GEN-AMDGPU: {{^}}}

// USE: [poseidon] site spread(double const*) at its call in third(double const*): accuracy target 1e-07
// USE-DAG: per_call_explicit.cu:[[@LINE-89]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z5norm2PKd.fpprofile:
// USE-DAG: per_call_explicit.cu:[[@LINE-82]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z6spreadPKd.poseidon._Z5thirdPKd.0.fpprofile:
// USE-DAG: per_call_explicit.cu:[[@LINE-79]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z3sumPKdi.fpprofile:
// USE-DAG: per_call_explicit.cu:[[@LINE-80]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z3sumPKdi.fpprofile:
// USE-DAG: per_call_explicit.cu:[[@LINE-81]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z5norm2PKd.fpprofile:
// USE-DAG: per_call_explicit.cu:[[@LINE-82]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z5scalePKdPdi.fpprofile:
// USE-DAG: per_call_explicit.cu:[[@LINE-83]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z6spreadPKd.poseidon._Z5thirdPKd.0.fpprofile:
// USE: [poseidon] every site of {{.*}}per_call_explicit.cu was left alone; emitting the module as written
// USE-NEXT: 7 warnings generated when compiling for {{.*}}

// Each call left alone calls the function itself with the primal arguments.
// IR: define {{.*}} @_Z5twicePKdPd(ptr {{.*}}[[TX:%[0-9]+]], ptr {{.*}}%{{[0-9]+}})
// IR-NEXT: [[TR:%[0-9]+]] = {{.*}}call double @_Z5norm2PKd(ptr [[TX]])
// IR-NEXT: [[TT:%[0-9]+]] = fmul {{.*}}double [[TR]], 2.000000e+00
// IR-NEXT: ret double [[TT]]
// IR: define {{.*}} @_Z5thirdPKd(ptr {{.*}}[[HX:%[0-9]+]])
// IR-NEXT: [[HR:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[HX]])
// IR-NEXT: ret double [[HR]]
// IR-NVPTX: define {{.*}}ptx_kernel void @_Z1kPKdPdS1_S1_i(ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}%{{[0-9]+}}, ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}%{{[0-9]+}}, i32 {{.*}}[[N:%[0-9]+]])
// IR-NVPTX-NEXT: [[S0:%[0-9]+]] = {{.*}}call double @_Z3sumPKdi(ptr [[X]], i32 [[N]])
// IR-NVPTX-NEXT: store double [[S0]], ptr [[Y]]
// IR-NVPTX-NEXT: [[NR:%[0-9]+]] = {{.*}}call double @_Z5norm2PKd(ptr [[X]])
// IR-NVPTX-NEXT: [[TW:%[0-9]+]] = fmul {{.*}}double [[NR]], 2.000000e+00
// IR-NVPTX-NEXT: [[Y3:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 24
// IR-NVPTX-NEXT: store double [[TW]], ptr [[Y3]]
// IR-NVPTX-NEXT: [[Y1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 8
// IR-NVPTX-NEXT: {{.*}}call void @_Z5scalePKdPdi(ptr [[X]], ptr {{.*}}[[Y1]], i32 [[N]])
// IR-NVPTX-NEXT: [[S1:%[0-9]+]] = {{.*}}call double @_Z3sumPKdi(ptr [[X]], i32 [[N]])
// IR-NVPTX-NEXT: [[Y2:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 16
// IR-NVPTX-NEXT: store double [[S1]], ptr [[Y2]]
// IR-NVPTX-NEXT: [[SP:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[X]])
// IR-NVPTX-NEXT: [[Y4:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 32
// IR-NVPTX-NEXT: store double [[SP]], ptr [[Y4]]
// IR-NVPTX-NEXT: ret void

// The kernel on AMDGPU, whose pointer parameters are in the global address
// space.
// IR-AMDGPU: define protected amdgpu_kernel void @_Z1kPKdPdS1_S1_i(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}%{{[0-9]+}}, ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}%{{[0-9]+}}, i32 {{.*}}[[N:%[0-9]+]])
// IR-AMDGPU-NEXT: [[XI:%[0-9]+]] = ptrtoint ptr addrspace(1) [[X]] to i64
// IR-AMDGPU-NEXT: [[XP:%[0-9]+]] = inttoptr i64 [[XI]] to ptr
// IR-AMDGPU-NEXT: [[YG:%[0-9]+]] = addrspacecast ptr addrspace(1) [[Y]] to ptr
// IR-AMDGPU-NEXT: [[S0:%[0-9]+]] = {{.*}}call double @_Z3sumPKdi(ptr [[XP]], i32 [[N]])
// IR-AMDGPU-NEXT: store double [[S0]], ptr addrspace(1) [[Y]]
// IR-AMDGPU-NEXT: [[NR:%[0-9]+]] = {{.*}}call double @_Z5norm2PKd(ptr [[XP]])
// IR-AMDGPU-NEXT: [[TW:%[0-9]+]] = fmul {{.*}}double [[NR]], 2.000000e+00
// IR-AMDGPU-NEXT: [[Y3:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 24
// IR-AMDGPU-NEXT: store double [[TW]], ptr addrspace(1) [[Y3]]
// IR-AMDGPU-NEXT: [[Y1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[YG]], i64 8
// IR-AMDGPU-NEXT: {{.*}}call void @_Z5scalePKdPdi(ptr [[XP]], ptr {{.*}}[[Y1]], i32 [[N]])
// IR-AMDGPU-NEXT: [[S1:%[0-9]+]] = {{.*}}call double @_Z3sumPKdi(ptr [[XP]], i32 [[N]])
// IR-AMDGPU-NEXT: [[Y2:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 16
// IR-AMDGPU-NEXT: store double [[S1]], ptr addrspace(1) [[Y2]]
// IR-AMDGPU-NEXT: [[SP:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[XP]])
// IR-AMDGPU-NEXT: [[Y4:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 32
// IR-AMDGPU-NEXT: store double [[SP]], ptr addrspace(1) [[Y4]]
// IR-AMDGPU-NEXT: ret void

// Without the plugin each call goes through the identity with the call's
// arguments as written.
// PLAIN: define {{.*}} @_Z5twicePKdPd(ptr {{.*}}[[TX:%[0-9]+]], ptr {{.*}}[[TDX:%[0-9]+]])
// PLAIN-NEXT: [[TF:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z5norm2PKd)
// PLAIN-NVPTX-NEXT: [[TDUP:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)
// PLAIN-AMDGPU-NEXT: [[TDUP:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_dup
// PLAIN-NEXT: [[TR:%[0-9]+]] = {{.*}}call {{.*}}double [[TF]](i32 {{.*}}[[TDUP]], ptr {{.*}}[[TX]], ptr {{.*}}[[TDX]])
// PLAIN-NEXT: [[TT:%[0-9]+]] = fmul {{.*}}double [[TR]], 2.000000e+00
// PLAIN-NEXT: ret double [[TT]]
// PLAIN: define {{.*}} @_Z5thirdPKd(ptr {{.*}}[[HX:%[0-9]+]])
// PLAIN-NEXT: [[HR:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[HX]])
// PLAIN-NEXT: ret double [[HR]]
// PLAIN-NVPTX: define {{.*}}ptx_kernel void @_Z1kPKdPdS1_S1_i(ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}[[DX:%[0-9]+]], ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// PLAIN-NVPTX-NEXT: [[F0:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z3sumPKdi)
// PLAIN-NVPTX-NEXT: [[DUP0:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)
// PLAIN-NVPTX-NEXT: [[R0:%[0-9]+]] = {{.*}}call {{.*}}double [[F0]](i32 {{.*}}[[DUP0]], ptr {{.*}}[[X]], ptr {{.*}}[[DX]], i32 {{.*}}[[N]])
// PLAIN-NVPTX-NEXT: store double [[R0]], ptr [[Y]]
// PLAIN-NVPTX-NEXT: [[F1:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z5norm2PKd)
// PLAIN-NVPTX-NEXT: [[DUP1:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)
// PLAIN-NVPTX-NEXT: [[R1:%[0-9]+]] = {{.*}}call {{.*}}double [[F1]](i32 {{.*}}[[DUP1]], ptr {{.*}}[[X]], ptr {{.*}}[[DX]])
// PLAIN-NVPTX-NEXT: [[T1:%[0-9]+]] = fmul {{.*}}double [[R1]], 2.000000e+00
// PLAIN-NVPTX-NEXT: [[Y3:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 24
// PLAIN-NVPTX-NEXT: store double [[T1]], ptr [[Y3]]
// PLAIN-NVPTX-NEXT: [[F2:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z5scalePKdPdi, double {{.*}}{{f?}}0x3EB0C6F7A0B5ED8D)
// PLAIN-NVPTX-NEXT: [[CONST2:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr)
// PLAIN-NVPTX-NEXT: [[DUP2:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)
// PLAIN-NVPTX-NEXT: [[Y1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 8
// PLAIN-NVPTX-NEXT: {{.*}}call void [[F2]](i32 {{.*}}[[CONST2]], ptr {{.*}}[[X]], i32 {{.*}}[[DUP2]], ptr {{.*}}[[Y1]], ptr {{.*}}[[DY]], i32 {{.*}}[[N]])
// PLAIN-NVPTX-NEXT: [[F3:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z3sumPKdi, double {{.*}}1.000000e-09)
// PLAIN-NVPTX-NEXT: [[CONST3:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr)
// PLAIN-NVPTX-NEXT: [[R3:%[0-9]+]] = {{.*}}call {{.*}}double [[F3]](i32 {{.*}}[[CONST3]], ptr {{.*}}[[X]], i32 {{.*}}[[N]])
// PLAIN-NVPTX-NEXT: [[Y2:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 16
// PLAIN-NVPTX-NEXT: store double [[R3]], ptr [[Y2]]
// PLAIN-NVPTX-NEXT: [[R4:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[X]])
// PLAIN-NVPTX-NEXT: [[Y4:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 32
// PLAIN-NVPTX-NEXT: store double [[R4]], ptr [[Y4]]
// PLAIN-NVPTX-NEXT: ret void

// The kernel on AMDGPU, whose pointer parameters are in the global address
// space.
// PLAIN-AMDGPU: define protected amdgpu_kernel void @_Z1kPKdPdS1_S1_i(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}[[DX:%[0-9]+]], ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// PLAIN-AMDGPU-NEXT: [[XI:%[0-9]+]] = ptrtoint ptr addrspace(1) [[X]] to i64
// PLAIN-AMDGPU-NEXT: [[XP:%[0-9]+]] = inttoptr i64 [[XI]] to ptr
// PLAIN-AMDGPU-NEXT: [[DXI:%[0-9]+]] = ptrtoint ptr addrspace(1) [[DX]] to i64
// PLAIN-AMDGPU-NEXT: [[DXP:%[0-9]+]] = inttoptr i64 [[DXI]] to ptr
// PLAIN-AMDGPU-NEXT: [[YG:%[0-9]+]] = addrspacecast ptr addrspace(1) [[Y]] to ptr
// PLAIN-AMDGPU-NEXT: [[DYI:%[0-9]+]] = ptrtoint ptr addrspace(1) [[DY]] to i64
// PLAIN-AMDGPU-NEXT: [[DYP:%[0-9]+]] = inttoptr i64 [[DYI]] to ptr
// PLAIN-AMDGPU-NEXT: [[F0:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z3sumPKdi)
// PLAIN-AMDGPU-NEXT: [[DUP0:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_dup
// PLAIN-AMDGPU-NEXT: [[R0:%[0-9]+]] = {{.*}}call {{.*}}double [[F0]](i32 {{.*}}[[DUP0]], ptr {{.*}}[[XP]], ptr {{.*}}[[DXP]], i32 {{.*}}[[N]])
// PLAIN-AMDGPU-NEXT: store double [[R0]], ptr addrspace(1) [[Y]]
// PLAIN-AMDGPU-NEXT: [[F1:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z5norm2PKd)
// PLAIN-AMDGPU-NEXT: [[DUP1:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_dup
// PLAIN-AMDGPU-NEXT: [[R1:%[0-9]+]] = {{.*}}call {{.*}}double [[F1]](i32 {{.*}}[[DUP1]], ptr {{.*}}[[XP]], ptr {{.*}}[[DXP]])
// PLAIN-AMDGPU-NEXT: [[T1:%[0-9]+]] = fmul {{.*}}double [[R1]], 2.000000e+00
// PLAIN-AMDGPU-NEXT: [[Y3:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 24
// PLAIN-AMDGPU-NEXT: store double [[T1]], ptr addrspace(1) [[Y3]]
// PLAIN-AMDGPU-NEXT: [[F2:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z5scalePKdPdi, double {{.*}}{{f?}}0x3EB0C6F7A0B5ED8D)
// PLAIN-AMDGPU-NEXT: [[CONST2:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_const
// PLAIN-AMDGPU-NEXT: [[DUP2:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_dup
// PLAIN-AMDGPU-NEXT: [[Y1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[YG]], i64 8
// PLAIN-AMDGPU-NEXT: {{.*}}call void [[F2]](i32 {{.*}}[[CONST2]], ptr {{.*}}[[XP]], i32 {{.*}}[[DUP2]], ptr {{.*}}[[Y1]], ptr {{.*}}[[DYP]], i32 {{.*}}[[N]])
// PLAIN-AMDGPU-NEXT: [[F3:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z3sumPKdi, double {{.*}}1.000000e-09)
// PLAIN-AMDGPU-NEXT: [[CONST3:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_const
// PLAIN-AMDGPU-NEXT: [[R3:%[0-9]+]] = {{.*}}call {{.*}}double [[F3]](i32 {{.*}}[[CONST3]], ptr {{.*}}[[XP]], i32 {{.*}}[[N]])
// PLAIN-AMDGPU-NEXT: [[Y2:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 16
// PLAIN-AMDGPU-NEXT: store double [[R3]], ptr addrspace(1) [[Y2]]
// PLAIN-AMDGPU-NEXT: [[R4:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[XP]])
// PLAIN-AMDGPU-NEXT: [[Y4:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 32
// PLAIN-AMDGPU-NEXT: store double [[R4]], ptr addrspace(1) [[Y4]]
// PLAIN-AMDGPU-NEXT: ret void

// DECL: declare {{.*}}ptr @__poseidon_site_explicit(ptr{{[^,)]*}})
// DECL: declare {{.*}}ptr @__poseidon_site_explicit_tau(ptr{{[^,)]*}}, double{{[^,)]*}})

// QUIET-NOT: {{.}}
// HOST-LABEL: define {{.*}} @_Z5twicePKdPd(
// HOST-SAME: ptr {{.*}}[[TX:%[0-9]+]], ptr {{.*}}%{{[0-9]+}})
// HOST-NEXT: [[TR:%[0-9]+]] = {{.*}}call {{.*}}double @_Z5norm2PKd(ptr [[TX]])
// HOST-NEXT: [[TT:%[0-9]+]] = fmul {{.*}}double [[TR]], 2.000000e+00
// HOST-NEXT: ret double [[TT]]
// HOST-LABEL: define {{.*}} @_Z5thirdPKd(
// HOST-SAME: ptr {{.*}}[[HX:%[0-9]+]])
// HOST-NEXT: [[HR:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[HX]])
// HOST-NEXT: ret double [[HR]]

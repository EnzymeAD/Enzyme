// REQUIRES: (amdgpu-registered-target && rocm) || (enzyme && cuda-toolkit && nvptx-registered-target)
// DEFINE: %{hip} = %clang -x hip --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S
// DEFINE: %{cuda} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S
// RUN: rm -rf %t && mkdir -p %t/a %t/n
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx942 -DAS_WRITTEN -DNO_SITE %s -o %t/a/as_written.s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx942 -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/a/plain.s %}
// RUN: %if amdgpu-registered-target && rocm %{ diff %t/a/as_written.s %t/a/plain.s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx942 -DAS_WRITTEN %s -o %t/a/as_written.site.s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx942 -I%poseidon_profile_inc %s -o %t/a/header.s %}
// RUN: %if amdgpu-registered-target && rocm %{ diff %t/a/as_written.site.s %t/a/header.s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx942 -DNO_SITE -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile %s -o %t/a/use.nosite.s %}
// RUN: %if amdgpu-registered-target && rocm %{ diff %t/a/as_written.s %t/a/use.nosite.s %}
// RUN: %if amdgpu-registered-target && rocm && enzyme %{ %{hip} --offload-arch=%amdgpu_arch -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/a/cache %s -o %t/a/gen.ll %}
// RUN: %if amdgpu-registered-target && rocm && enzyme %{ FileCheck --check-prefix=CALLS-AMDGPU %s < %t/a/gen.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx942 -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile %s -o %t/a/use.s %}
// RUN: %if amdgpu-registered-target && rocm %{ diff %t/a/as_written.s %t/a/use.s %}
// RUN: %if amdgpu-registered-target && rocm && enzyme %{ mkdir -p %t/a/prof %}
// RUN: %if amdgpu-registered-target && rocm && enzyme %{ sed -n 's/^@poseidon.static.preprocess__Z4wavePdPKdi_poseidon_body = .*c"\(.*\)\\00".*/\1Executed = 0\\0A/p' %t/a/gen.ll | sed 's/\\0A/\n/g' > %t/a/prof/_Z4wavePdPKdi.fpprofile %}
// RUN: %if amdgpu-registered-target && rocm && enzyme %{ %{hip} --offload-arch=%amdgpu_arch -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/a/prof -mllvm -poseidon-print %s -o %t/a/use.profiled.s 2>&1 | FileCheck --check-prefix=PROFILED-AMDGPU %s %}
// RUN: %if amdgpu-registered-target && rocm && enzyme %{ %{hip} --offload-arch=%amdgpu_arch -DAS_WRITTEN -DNO_SITE %s -o %t/a/as_written.arch.s %}
// RUN: %if amdgpu-registered-target && rocm && enzyme %{ diff %t/a/as_written.arch.s %t/a/use.profiled.s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx90a -DAS_WRITTEN -DNO_SITE %s -o %t/a/as_written.90a.s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx90a -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/a/plain.90a.s %}
// RUN: %if amdgpu-registered-target && rocm %{ diff %t/a/as_written.90a.s %t/a/plain.90a.s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{hip} --offload-arch=gfx90a -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile %s -o %t/a/use.90a.s %}
// RUN: %if amdgpu-registered-target && rocm %{ diff %t/a/as_written.90a.s %t/a/use.90a.s %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{cuda} -DAS_WRITTEN -DNO_SITE %s -o %t/n/as_written.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{cuda} -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/n/plain.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ diff %t/n/as_written.ptx %t/n/plain.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{cuda} -DAS_WRITTEN %s -o %t/n/as_written.site.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{cuda} -I%poseidon_profile_inc %s -o %t/n/header.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ diff %t/n/as_written.site.ptx %t/n/header.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{cuda} -DNO_SITE -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile %s -o %t/n/use.nosite.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ diff %t/n/as_written.ptx %t/n/use.nosite.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{cuda} -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/n/cache %s -o %t/n/gen.ll %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ FileCheck --check-prefix=CALLS-NVPTX %s < %t/n/gen.ll %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{cuda} -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile %s -o %t/n/use.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ diff %t/n/as_written.ptx %t/n/use.ptx %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ mkdir -p %t/n/prof %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ sed -n 's/^@poseidon.static.preprocess__Z4wavePfPKfi_poseidon_body = .*c"\(.*\)\\00".*/\1Executed = 0\\0A/p' %t/n/gen.ll | sed 's/\\0A/\n/g' > %t/n/prof/_Z4wavePfPKfi.fpprofile %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ %{cuda} -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/n/prof -mllvm -poseidon-print %s -o %t/n/use.profiled.ptx 2>&1 | FileCheck --check-prefix=PROFILED-NVPTX %s %}
// RUN: %if enzyme && cuda-toolkit && nvptx-registered-target %{ diff %t/n/as_written.ptx %t/n/use.profiled.ptx %}

// Profiling does not depend on Enzyme's plugin being loaded: device math
// (ocml, libdevice) stays a call Enzyme differentiates by name, and a
// never-executed profile (the generating compile's static header) is accepted
// by Poseidon's plugin alone. A compile that changes nothing emits the source
// as written (gfx942, gfx90a, NVPTX), as does including poseidon.h without the
// plugin.

#ifdef __HIP__
#include <hip/hip_runtime.h>
#endif

#ifdef AS_WRITTEN
#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon")))
#else
#include "poseidon/poseidon.h"
#endif
#ifdef NO_SITE
#undef POSEIDON_OPTIMIZE
#define POSEIDON_OPTIMIZE
#endif

#ifdef __HIP__
POSEIDON_OPTIMIZE __global__ void wave(double *o, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    o[i] = sqrt(x) * exp(-x) + (1.0 - cos(x)) / (x * x) + pow(x, 1.5) * sin(x);
  }
}
#else
POSEIDON_OPTIMIZE __global__ void wave(float *o, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i];
    o[i] = sqrtf(x) * expf(-x) + (1.0f - cosf(x)) / (x * x) +
           powf(x, 1.5f) * sinf(x);
  }
}
#endif

// llvm.exp.f64 becomes __ocml_exp_f64 once poseidon.h links it; each call is
// profiled and differentiated by name, adding into the shadow of a.
// CALLS-AMDGPU-LABEL: define amdgpu_kernel void @_Z4wavePdPKdi(
// CALLS-AMDGPU-SAME: ptr addrspace(1) {{[^,]*}}%{{[0-9]+}}, ptr addrspace(1) {{[^,]*}}%[[A:[0-9]+]], i32 {{[^,]*}}%{{[0-9]+}}, ptr {{[^,]*}}%{{[0-9]+}}, ptr {{[^,]*}}%[[DA:[0-9]+]])
// CALLS-AMDGPU: %[[DAG:[0-9]+]] = addrspacecast ptr %[[DA]] to ptr addrspace(1)
// CALLS-AMDGPU: %[[XP:[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) %[[A]], i64 %[[I:[0-9]+]]
// CALLS-AMDGPU-NEXT: %[[X:[0-9]+]] = load double, ptr addrspace(1) %[[XP]],
// CALLS-AMDGPU: %[[NEG:[0-9]+]] = fneg double %[[X]],
// CALLS-AMDGPU-NEXT: %[[NEGP:[0-9]+]] = call double @__poseidon_device_perturb(i32 0, double %[[NEG]])
// CALLS-AMDGPU: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 1, double %[[NEGP]], i64 1,
// CALLS-AMDGPU: %[[EXP:[0-9]+]] = call {{.*}}double @__ocml_exp_f64(double %[[NEGP]])
// CALLS-AMDGPU-NEXT: %[[EXPP:[0-9]+]] = call double @__poseidon_device_perturb(i32 0, double %[[EXP]])
// CALLS-AMDGPU: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 2, double %[[EXPP]], i64 1,
// CALLS-AMDGPU: %[[COS:[0-9]+]] = call {{.*}}double @__ocml_cos_f64(double {{(noundef )?}}%[[X]])
// CALLS-AMDGPU-NEXT: %[[COSP:[0-9]+]] = call double @__poseidon_device_perturb(i32 0, double %[[COS]])
// CALLS-AMDGPU: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 3, double %[[COSP]], i64 1,
// CALLS-AMDGPU: %[[POW:[0-9]+]] = call {{.*}}double @__ocml_pow_f64(double {{(noundef )?}}%[[X]], double {{(noundef )?}}1.500000e+00)
// CALLS-AMDGPU-NEXT: %[[POWP:[0-9]+]] = call double @__poseidon_device_perturb(i32 0, double %[[POW]])
// CALLS-AMDGPU: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 8, double %[[POWP]], i64 2,
// CALLS-AMDGPU: %[[SIN:[0-9]+]] = call {{.*}}double @__ocml_sin_f64(double {{(noundef )?}}%[[X]])
// CALLS-AMDGPU-NEXT: %[[SINP:[0-9]+]] = call double @__poseidon_device_perturb(i32 0, double %[[SIN]])
// CALLS-AMDGPU: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 9, double %[[SINP]], i64 1,
// CALLS-AMDGPU: call {{.*}}double @__ocml_exp_f64(double %[[NEG]])
// CALLS-AMDGPU: call {{.*}}double @__ocml_cos_f64(double %[[X]])
// CALLS-AMDGPU: call {{.*}}double @__ocml_pow_f64(double %[[X]], double 5.000000e-01)
// CALLS-AMDGPU: call {{.*}}double @__ocml_sin_f64(double %[[X]])
// CALLS-AMDGPU: %[[DXP:[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) %[[DAG]], i64 %[[I]]
// CALLS-AMDGPU-NEXT: atomicrmw fadd ptr addrspace(1) %[[DXP]], double %{{[0-9]+}} monotonic
// CALLS-AMDGPU: {{^}}}
// PROFILED-AMDGPU: {{^}}[poseidon] _Z4wavePdPKdi_poseidon_body: the profiling run instrumented this site and never executed it ({{.*}}/prof/_Z4wavePdPKdi.fpprofile); left unchanged{{$}}
// PROFILED-AMDGPU: {{^}}[poseidon] every site of {{.*}}device_math.cu was left alone; emitting the module as written{{$}}

// Each libdevice call is profiled and differentiated by name, adding into the
// shadow of a.
// CALLS-NVPTX-LABEL: define ptx_kernel void @_Z4wavePfPKfi(
// CALLS-NVPTX-SAME: ptr {{[^,]*}}[[O:%[0-9]+]], ptr {{[^,]*}}[[A:%[0-9]+]], i32 {{[^,]*}}[[N:%[0-9]+]], ptr {{[^,]*}}[[DO:%[0-9]+]], ptr {{[^,]*}}[[DA:%[0-9]+]])
// CALLS-NVPTX-NEXT: call {{.*}}void @diffepreprocess__Z4wavePfPKfi_poseidon_body(ptr [[O]], ptr [[DO]], ptr [[A]], ptr [[DA]], i32 [[N]])
// CALLS-NVPTX-NEXT: ret void
// CALLS-NVPTX-LABEL: define internal {{.*}}void @diffepreprocess__Z4wavePfPKfi_poseidon_body(
// CALLS-NVPTX-SAME: ptr {{[^,]*}}%{{[0-9]+}}, ptr {{[^,]*}}%{{[0-9]+}}, ptr {{[^,]*}}[[PA:%[0-9]+]], ptr {{[^,]*}}[[PDA:%[0-9]+]], i32
// CALLS-NVPTX: [[XP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[PA]], i64 [[I:%[0-9]+]]
// CALLS-NVPTX-NEXT: [[X:%[0-9]+]] = load float, ptr [[XP]],
// CALLS-NVPTX: [[NEG:%[0-9]+]] = fneg float [[X]],
// CALLS-NVPTX-NEXT: [[NEGP:%[0-9]+]] = call float @__poseidon_device_perturbf(i32 0, float [[NEG]])
// CALLS-NVPTX-NEXT: [[NEGD:%[0-9]+]] = fpext float [[NEGP]] to double
// CALLS-NVPTX: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 1, double [[NEGD]], i64 1,
// CALLS-NVPTX: [[EXP:%[0-9]+]] = {{(tail )?}}call {{.*}}float @__nv_expf(float {{(noundef )?}}[[NEGP]])
// CALLS-NVPTX-NEXT: [[EXPP:%[0-9]+]] = call float @__poseidon_device_perturbf(i32 0, float [[EXP]])
// CALLS-NVPTX-NEXT: [[EXPD:%[0-9]+]] = fpext float [[EXPP]] to double
// CALLS-NVPTX: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 2, double [[EXPD]], i64 1,
// CALLS-NVPTX: [[COS:%[0-9]+]] = {{(tail )?}}call {{.*}}float @__nv_cosf(float {{(noundef )?}}[[X]])
// CALLS-NVPTX-NEXT: [[COSP:%[0-9]+]] = call float @__poseidon_device_perturbf(i32 0, float [[COS]])
// CALLS-NVPTX-NEXT: [[COSD:%[0-9]+]] = fpext float [[COSP]] to double
// CALLS-NVPTX: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 3, double [[COSD]], i64 1,
// CALLS-NVPTX: [[POW:%[0-9]+]] = {{(tail )?}}call {{.*}}float @__nv_powf(float {{(noundef )?}}[[X]], float {{(noundef )?}}1.500000e+00)
// CALLS-NVPTX-NEXT: [[POWP:%[0-9]+]] = call float @__poseidon_device_perturbf(i32 0, float [[POW]])
// CALLS-NVPTX-NEXT: [[POWD:%[0-9]+]] = fpext float [[POWP]] to double
// CALLS-NVPTX: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 8, double [[POWD]], i64 2,
// CALLS-NVPTX: [[SIN:%[0-9]+]] = {{(tail )?}}call {{.*}}float @__nv_sinf(float {{(noundef )?}}[[X]])
// CALLS-NVPTX-NEXT: [[SINP:%[0-9]+]] = call float @__poseidon_device_perturbf(i32 0, float [[SIN]])
// CALLS-NVPTX-NEXT: [[SIND:%[0-9]+]] = fpext float [[SINP]] to double
// CALLS-NVPTX: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 9, double [[SIND]], i64 1,
// CALLS-NVPTX: call {{.*}}float @__nv_cosf(float [[X]])
// CALLS-NVPTX: call {{.*}}float @__nv_powf(float [[X]], float 5.000000e-01)
// CALLS-NVPTX: call {{.*}}float @__nv_sinf(float [[X]])
// CALLS-NVPTX: call {{.*}}float @__nv_expf(float [[NEG]])
// CALLS-NVPTX: [[DXP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[PDA]], i64 [[I]]
// CALLS-NVPTX-NEXT: atomicrmw fadd ptr [[DXP]], float %{{[0-9]+}} monotonic
// CALLS-NVPTX: {{^}}}
// PROFILED-NVPTX: {{^}}[poseidon] _Z4wavePfPKfi_poseidon_body: the profiling run instrumented this site and never executed it ({{.*}}/prof/_Z4wavePfPKfi.fpprofile); left unchanged{{$}}
// PROFILED-NVPTX: {{^}}[poseidon] every site of {{.*}}device_math.cu was left alone; emitting the module as written{{$}}

// REQUIRES: amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -DAS_WRITTEN -DNO_SITE %s -o %t/as_written.s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.s
// RUN: diff %t/as_written.s %t/plain.s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -DAS_WRITTEN %s -o %t/as_written.site.s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %s -o %t/header.s
// RUN: diff %t/as_written.site.s %t/header.s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -DNO_SITE -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/use.nosite.s
// RUN: diff %t/as_written.s %t/use.nosite.s
// RUN: %if enzyme %{ %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll %}
// RUN: %if enzyme %{ FileCheck --check-prefix=CALLS %s < %t/gen.ll %}
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/use.s
// RUN: diff %t/as_written.s %t/use.s
// RUN: %if enzyme %{ mkdir -p %t/prof %}
// RUN: %if enzyme %{ sed -n 's/^@fpprofile_static_preprocess__Z4wavePdPKdi_poseidon_body = .*c"\(.*\)\\00".*/\1Executed = 0\\0A/p' %t/gen.ll | sed 's/\\0A/\n/g' > %t/prof/_Z4wavePdPKdi.fpprofile %}
// RUN: %if enzyme %{ %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-print %s -o %t/use.profiled.s 2>&1 | FileCheck --check-prefix=PROFILED %s %}
// RUN: %if enzyme %{ %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -DAS_WRITTEN -DNO_SITE %s -o %t/as_written.arch.s %}
// RUN: %if enzyme %{ diff %t/as_written.arch.s %t/use.profiled.s %}
// RUN: %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -DAS_WRITTEN -DNO_SITE %s -o %t/as_written.90a.s
// RUN: %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.90a.s
// RUN: diff %t/as_written.90a.s %t/plain.90a.s
// RUN: %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/use.90a.s
// RUN: diff %t/as_written.90a.s %t/use.90a.s

// Enzyme's plugin is loaded to generate a profile and not to use one, so the
// form a site is profiled in does not depend on it: the ocml math of a site
// stays a call, which Enzyme differentiates by name, and the profile the run
// would write for a site it never executed (the static header the generating
// compile embeds) is accepted by a use compile with Poseidon's plugin alone. A
// compile that profiles or rewrites nothing emits the assembly of the source as
// written (gfx942, and gfx90a), as does one that includes poseidon.h without
// the plugin and so without POSEIDON_DEVICE_MATH.

#include <hip/hip_runtime.h>

#ifdef AS_WRITTEN
#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon")))
#else
#include "poseidon/poseidon.h"
#endif
#ifdef NO_SITE
#undef POSEIDON_OPTIMIZE
#define POSEIDON_OPTIMIZE
#endif

POSEIDON_OPTIMIZE __global__ void wave(double *o, const double *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    o[i] = sqrt(x) * exp(-x) + (1.0 - cos(x)) / (x * x) + pow(x, 1.5) * sin(x);
  }
}

// HIP writes exp(double) as llvm.exp.f64, which Enzyme takes to the ocml
// function that implements it once poseidon.h has linked that. Each call is
// profiled, and the derivative calls the derivative of each by name and adds
// into the shadow of a.
// CALLS-LABEL: define amdgpu_kernel void @_Z4wavePdPKdi(
// CALLS-SAME: ptr addrspace(1) {{[^,]*}}%{{[0-9]+}}, ptr addrspace(1) {{[^,]*}}%[[A:[0-9]+]], i32 {{[^,]*}}%{{[0-9]+}}, ptr {{[^,]*}}%{{[0-9]+}}, ptr {{[^,]*}}%[[DA:[0-9]+]])
// CALLS: %[[DAG:[0-9]+]] = addrspacecast ptr %[[DA]] to ptr addrspace(1)
// CALLS: %[[XP:[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) %[[A]], i64 %[[I:[0-9]+]]
// CALLS-NEXT: %[[X:[0-9]+]] = load double, ptr addrspace(1) %[[XP]],
// CALLS: %[[NEG:[0-9]+]] = fneg double %[[X]],
// CALLS-NEXT: %[[NEGP:[0-9]+]] = call double @poseidonProbePerturbCUDA(i32 0, double %[[NEG]])
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 1, double %[[NEGP]], i64 1,
// CALLS: %[[EXP:[0-9]+]] = call {{.*}}double @__ocml_exp_f64(double %[[NEGP]])
// CALLS-NEXT: %[[EXPP:[0-9]+]] = call double @poseidonProbePerturbCUDA(i32 0, double %[[EXP]])
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 2, double %[[EXPP]], i64 1,
// CALLS: %[[COS:[0-9]+]] = call {{.*}}double @__ocml_cos_f64(double {{(noundef )?}}%[[X]])
// CALLS-NEXT: %[[COSP:[0-9]+]] = call double @poseidonProbePerturbCUDA(i32 0, double %[[COS]])
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 3, double %[[COSP]], i64 1,
// CALLS: %[[POW:[0-9]+]] = call {{.*}}double @__ocml_pow_f64(double {{(noundef )?}}%[[X]], double {{(noundef )?}}1.500000e+00)
// CALLS-NEXT: %[[POWP:[0-9]+]] = call double @poseidonProbePerturbCUDA(i32 0, double %[[POW]])
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 8, double %[[POWP]], i64 2,
// CALLS: %[[SIN:[0-9]+]] = call {{.*}}double @__ocml_sin_f64(double {{(noundef )?}}%[[X]])
// CALLS-NEXT: %[[SINP:[0-9]+]] = call double @poseidonProbePerturbCUDA(i32 0, double %[[SIN]])
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePdPKdi_poseidon_body to ptr), i64 9, double %[[SINP]], i64 1,
// CALLS: call {{.*}}double @__ocml_exp_f64(double %[[NEG]])
// CALLS: call {{.*}}double @__ocml_cos_f64(double %[[X]])
// CALLS: call {{.*}}double @__ocml_pow_f64(double %[[X]], double 5.000000e-01)
// CALLS: call {{.*}}double @__ocml_sin_f64(double %[[X]])
// CALLS: %[[DXP:[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) %[[DAG]], i64 %[[I]]
// CALLS-NEXT: atomicrmw fadd ptr addrspace(1) %[[DXP]], double %{{[0-9]+}} monotonic
// CALLS: {{^}}}

// PROFILED: {{^}}[poseidon] _Z4wavePdPKdi_poseidon_body: the profiling run instrumented this site and never executed it ({{.*}}/prof/_Z4wavePdPKdi.fpprofile); left unchanged{{$}}
// PROFILED: {{^}}[poseidon] every site of {{.*}}device_math_amdgpu.hip was left alone; emitting the module as written{{$}}

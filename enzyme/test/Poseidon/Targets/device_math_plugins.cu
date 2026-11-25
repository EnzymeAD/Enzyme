// REQUIRES: enzyme, cuda-runtime, nvptx-registered-target
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -DAS_WRITTEN -DNO_SITE %s -o %t/as_written.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon %s -o %t/plain.ptx
// RUN: diff %t/as_written.ptx %t/plain.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -DAS_WRITTEN %s -o %t/as_written.site.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%poseidon_profile_inc %s -o %t/header.ptx
// RUN: diff %t/as_written.site.ptx %t/header.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -DNO_SITE -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/use.nosite.ptx
// RUN: diff %t/as_written.ptx %t/use.nosite.ptx
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll
// RUN: FileCheck --check-prefix=CALLS %s < %t/gen.ll
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/use.ptx
// RUN: diff %t/as_written.ptx %t/use.ptx
// RUN: mkdir -p %t/prof
// RUN: sed -n 's/^@fpprofile_static_preprocess__Z4wavePfPKfi_poseidon_body = .*c"\(.*\)\\00".*/\1Executed = 0\\0A/p' %t/gen.ll | sed 's/\\0A/\n/g' > %t/prof/preprocess__Z4wavePfPKfi_poseidon_body.fpprofile
// RUN: %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -O2 -ffp-contract=on -S -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-print %s -o %t/use.profiled.ptx 2>&1 | FileCheck --check-prefix=PROFILED %s
// RUN: diff %t/as_written.ptx %t/use.profiled.ptx

// Enzyme's plugin is loaded to generate a profile and not to use one, so the
// form a site is profiled in does not depend on it: the libdevice math stays a
// call, which Enzyme differentiates by name, and the profile the run would
// write for a site it never executed (the static header the generating compile
// embeds) is accepted by a use compile with Poseidon's plugin alone. A compile
// that profiles or rewrites nothing emits the PTX of the source as written, as
// does one that includes poseidon.h without the plugin and so without
// POSEIDON_DEVICE_MATH.

#ifdef AS_WRITTEN
#define POSEIDON_OPTIMIZE __attribute__((annotate("poseidon")))
#else
#include "poseidon/poseidon.h"
#endif
#ifdef NO_SITE
#undef POSEIDON_OPTIMIZE
#define POSEIDON_OPTIMIZE
#endif

POSEIDON_OPTIMIZE __global__ void wave(float *o, const float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i];
    o[i] = sqrtf(x) * expf(-x) + (1.0f - cosf(x)) / (x * x) +
           powf(x, 1.5f) * sinf(x);
  }
}

// The kernel runs the derivative of its body. Each libdevice call there is
// profiled, and the derivative calls the derivative of each by name and adds
// into the shadow of a.
// CALLS-LABEL: define ptx_kernel void @_Z4wavePfPKfi(
// CALLS-SAME: ptr {{[^,]*}}[[O:%[0-9]+]], ptr {{[^,]*}}[[A:%[0-9]+]], i32 {{[^,]*}}[[N:%[0-9]+]], ptr {{[^,]*}}[[DO:%[0-9]+]], ptr {{[^,]*}}[[DA:%[0-9]+]])
// CALLS-NEXT: call {{.*}}void @diffepreprocess__Z4wavePfPKfi_poseidon_body(ptr [[O]], ptr [[DO]], ptr [[A]], ptr [[DA]], i32 [[N]])
// CALLS-NEXT: ret void
// CALLS-LABEL: define internal {{.*}}void @diffepreprocess__Z4wavePfPKfi_poseidon_body(
// CALLS-SAME: ptr {{[^,]*}}%{{[0-9]+}}, ptr {{[^,]*}}%{{[0-9]+}}, ptr {{[^,]*}}[[PA:%[0-9]+]], ptr {{[^,]*}}[[PDA:%[0-9]+]], i32
// CALLS: [[XP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[PA]], i64 [[I:%[0-9]+]]
// CALLS-NEXT: [[X:%[0-9]+]] = load float, ptr [[XP]],
// CALLS: [[NEG:%[0-9]+]] = fneg float [[X]],
// CALLS-NEXT: [[NEGP:%[0-9]+]] = call {{.*}}float @__poseidon_probe_preprocess__Z4wavePfPKfi_poseidon_body_1_aug(float [[NEG]])
// CALLS-NEXT: [[NEGD:%[0-9]+]] = fpext float [[NEGP]] to double
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 1, double [[NEGD]], i64 1,
// CALLS: [[EXP:%[0-9]+]] = {{(tail )?}}call {{.*}}float @__nv_expf(float {{(noundef )?}}[[NEGP]])
// CALLS-NEXT: [[EXPP:%[0-9]+]] = call {{.*}}float @__poseidon_probe_preprocess__Z4wavePfPKfi_poseidon_body_2_aug(float [[EXP]])
// CALLS-NEXT: [[EXPD:%[0-9]+]] = fpext float [[EXPP]] to double
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 2, double [[EXPD]], i64 1,
// CALLS: [[COS:%[0-9]+]] = {{(tail )?}}call {{.*}}float @__nv_cosf(float {{(noundef )?}}[[X]])
// CALLS-NEXT: [[COSP:%[0-9]+]] = call {{.*}}float @__poseidon_probe_preprocess__Z4wavePfPKfi_poseidon_body_3_aug(float [[COS]])
// CALLS-NEXT: [[COSD:%[0-9]+]] = fpext float [[COSP]] to double
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 3, double [[COSD]], i64 1,
// CALLS: [[POW:%[0-9]+]] = {{(tail )?}}call {{.*}}float @__nv_powf(float {{(noundef )?}}[[X]], float {{(noundef )?}}1.500000e+00)
// CALLS-NEXT: [[POWP:%[0-9]+]] = call {{.*}}float @__poseidon_probe_preprocess__Z4wavePfPKfi_poseidon_body_8_aug(float [[POW]])
// CALLS-NEXT: [[POWD:%[0-9]+]] = fpext float [[POWP]] to double
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 8, double [[POWD]], i64 2,
// CALLS: [[SIN:%[0-9]+]] = {{(tail )?}}call {{.*}}float @__nv_sinf(float {{(noundef )?}}[[X]])
// CALLS-NEXT: [[SINP:%[0-9]+]] = call {{.*}}float @__poseidon_probe_preprocess__Z4wavePfPKfi_poseidon_body_9_aug(float [[SIN]])
// CALLS-NEXT: [[SIND:%[0-9]+]] = fpext float [[SINP]] to double
// CALLS: call void @poseidonLogValueCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z4wavePfPKfi_poseidon_body to ptr), i64 9, double [[SIND]], i64 1,
// CALLS: call {{.*}}float @__nv_cosf(float [[X]])
// CALLS: call {{.*}}float @__nv_powf(float [[X]], float 5.000000e-01)
// CALLS: call {{.*}}float @__nv_sinf(float [[X]])
// CALLS: call {{.*}}float @__nv_expf(float [[NEG]])
// CALLS: [[DXP:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[PDA]], i64 [[I]]
// CALLS-NEXT: atomicrmw fadd ptr [[DXP]], float %{{[0-9]+}} monotonic
// CALLS: {{^}}}

// PROFILED: {{^}}[poseidon] _Z4wavePfPKfi_poseidon_body: the profiling run instrumented this site and never executed it ({{.*}}/prof/preprocess__Z4wavePfPKfi_poseidon_body.fpprofile); left unchanged{{$}}
// PROFILED: {{^}}[poseidon] every site of {{.*}}device_math_plugins.cu was left alone; emitting the module as written{{$}}

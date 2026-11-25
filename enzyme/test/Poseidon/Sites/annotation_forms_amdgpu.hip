// REQUIRES: amdgpu-registered-target
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=gfx942 -nogpulib -nogpuinc -O2 -c -I%poseidon_profile_inc %s -o %t/plain.o
// RUN: %clang -x hip --offload-arch=gfx942 -nogpulib -nogpuinc --cuda-device-only -O2 -S -emit-llvm -I%poseidon_profile_inc %s -o - | FileCheck --check-prefix=TYPE %s
// RUN: %clang -x hip --offload-arch=gfx942 -nogpulib -nogpuinc --cuda-device-only -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/../Inputs/other_sites -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-print %s -o /dev/null 2>&1 | FileCheck --check-prefix=READ --implicit-check-not=error: %s

// A HIP unit with both annotation forms compiles and links its device code
// without the plugin: both put an entry of one type into
// llvm.global.annotations, which clang types after its first entry (on AMDGPU
// an entry without arguments has another type). The plugin reads the target
// of the one and none of the other. Without ROCm: the runtime declarations a
// kernel stub needs are written out.

#include <cstddef>

struct dim3 {
  unsigned x, y, z;
};
typedef struct ihipStream_t *hipStream_t;
extern "C" int hipLaunchKernel(const void *, dim3, dim3, void **, size_t,
                               hipStream_t);
extern "C" int __hipPushCallConfiguration(dim3, dim3, size_t, hipStream_t);
extern "C" int __hipPopCallConfiguration(dim3 *, dim3 *, size_t *,
                                         hipStream_t *);

#include "poseidon/poseidon.h"

POSEIDON_OPTIMIZE __attribute__((global)) void plain(double *x) {
  x[0] = x[0] * x[0] + 1.0;
}

POSEIDON_OPTIMIZE_TAU(1e-9) __attribute__((global)) void tight(double *x) {
  x[1] = x[1] * x[1] + 2.0;
}

// TYPE: [[ANN:@[^ ]+]] = private unnamed_addr addrspace(4) constant [9 x i8] c"poseidon\00", section "llvm.metadata"
// TYPE: [[FILE:@[^ ]+]] = private unnamed_addr addrspace(4) constant [{{[0-9]+}} x i8] c"{{.*}}annotation_forms_amdgpu.hip\00", section "llvm.metadata"
// TYPE: [[NOARGS:@[^ ]+]] = private unnamed_addr addrspace(1) constant { i32 } zeroinitializer, section "llvm.metadata"
// TYPE: [[TAU:@[^ ]+]] = private unnamed_addr addrspace(1) constant { double } { double 1.000000e-09 }, section "llvm.metadata"
// TYPE: @llvm.global.annotations = appending addrspace(1) global [2 x { ptr addrspace(1), ptr addrspace(4), ptr addrspace(4), i32, ptr addrspace(1) }] [{ ptr addrspace(1), ptr addrspace(4), ptr addrspace(4), i32, ptr addrspace(1) } { ptr addrspace(1) addrspacecast (ptr @_Z5plainPd to ptr addrspace(1)), ptr addrspace(4) [[ANN]], ptr addrspace(4) [[FILE]], i32 [[@LINE-12]], ptr addrspace(1) [[NOARGS]] }, { ptr addrspace(1), ptr addrspace(4), ptr addrspace(4), i32, ptr addrspace(1) } { ptr addrspace(1) addrspacecast (ptr @_Z5tightPd to ptr addrspace(1)), ptr addrspace(4) [[ANN]], ptr addrspace(4) [[FILE]], i32 [[@LINE-8]], ptr addrspace(1) [[TAU]] }], section "llvm.metadata"

// READ-NOT: plain(double*): accuracy target
// READ: annotation_forms_amdgpu.hip:[[@LINE-15]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z5plainPd.fpprofile:
// READ-NOT: plain(double*): accuracy target
// READ: [poseidon] site tight(double*): accuracy target 1e-09
// READ-NEXT: annotation_forms_amdgpu.hip:[[@LINE-14]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z5tightPd.fpprofile:
// READ: [poseidon] every site of {{.*}}annotation_forms_amdgpu.hip was left alone; emitting the module as written

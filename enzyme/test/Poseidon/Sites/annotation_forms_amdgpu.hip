// REQUIRES: amdgpu-registered-target
// RUN: rm -rf %t && mkdir -p %t
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

// TYPE: @llvm.global.annotations = {{.*}} global [2 x { ptr addrspace(1), ptr addrspace(4), ptr addrspace(4), i32, ptr addrspace(1) }]

// READ: no profile at {{.*}}/preprocess__Z5plainPd_poseidon_body.fpprofile
// READ-NOT: plain(double*): accuracy target
// READ: [poseidon] site tight(double*): accuracy target 1e-09

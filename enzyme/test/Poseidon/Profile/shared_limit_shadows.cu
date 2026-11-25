// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// RUN: rm -rf %t && mkdir %t
// RUN: %if nvptx-device-profiler %{ not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/nvptx %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: -DLIMIT=49152 %}
// RUN: %if amdgpu-registered-target && rocm %{ not %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/amdgpu %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: -DLIMIT=65536 %}

// Each kernel's buffer and its shadow fit the static shared memory a kernel can
// declare (48 KiB on NVPTX, the 64 KiB local data share of a gfx942 workgroup
// on AMDGPU), but the derivative of the second kernel also reaches the shadow
// of the first kernel's buffer, which takes it above the limit: the compile
// stops with an error naming all three before the backend refuses the module.

#include "poseidon/poseidon.h"

#define N 3072

#ifdef __NVPTX__
#define THREAD_X __nvvm_read_ptx_sreg_tid_x()
#define BLOCK_X __nvvm_read_ptx_sreg_ntid_x()
#else
#define THREAD_X __builtin_amdgcn_workitem_id_x()
#define BLOCK_X __builtin_amdgcn_workgroup_size_x()
#endif

template <int K>
__attribute__((global)) POSEIDON_OPTIMIZE void stage(const double *x,
                                                     double *y) {
  __attribute__((shared)) double buf[N];
  for (int i = THREAD_X; i < N; i += BLOCK_X)
    buf[i] = x[i] * x[i] + K;
  for (int i = THREAD_X; i < N; i += BLOCK_X)
    y[i] = buf[N - 1 - i] * 3.0;
}

template __attribute__((global)) void stage<1>(const double *, double *);
template __attribute__((global)) void stage<2>(const double *, double *);

// CHECK: shared_limit_shadows.cu:[[@LINE-12]]:48: error: Poseidon: void stage<2>(double const*, double*) reaches 73728 bytes of static shared memory once profiled, above the [[LIMIT]] a kernel can declare: its own buffers void stage<2>(double const*, double*)::buf (24576 bytes); the shadows of its buffers void stage<2>(double const*, double*)::buf (24576 bytes); the shadows of other kernels' buffers void stage<1>(double const*, double*)::buf (24576 bytes). Profile the kernel with smaller shared buffers, or the kernels whose shadows it reaches in another translation unit.

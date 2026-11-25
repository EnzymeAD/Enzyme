// REQUIRES: enzyme, nvptx-device-profiler
// RUN: rm -rf %t && mkdir %t
// RUN: not %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

// Each kernel's buffer and its shadow fit the static shared memory a kernel can
// declare (48 KiB), but the derivative of the second kernel also reaches the
// shadow of the first kernel's buffer, which takes it above the limit: the
// compile stops with an error naming all three before the backend refuses the
// module.

#include "poseidon/poseidon.h"

#define SHARED __attribute__((shared))

template <int K>
__attribute__((global)) POSEIDON_OPTIMIZE void stage(const double *x,
                                                     double *y) {
  SHARED double buf[2560];
  int t = __nvvm_read_ptx_sreg_tid_x();
  for (int i = t; i < 2560; i += __nvvm_read_ptx_sreg_ntid_x())
    buf[i] = x[i] * x[i] + K;
  for (int i = t; i < 2560; i += __nvvm_read_ptx_sreg_ntid_x())
    y[i] = buf[2559 - i] * 3.0;
}

template __attribute__((global)) void stage<1>(const double *, double *);
template __attribute__((global)) void stage<2>(const double *, double *);

// CHECK: shared_limit_shadows_nvptx.cu:[[@LINE-13]]:48: error: Poseidon: void stage<2>(double const*, double*) reaches 61440 bytes of static shared memory once profiled, above the 49152 a kernel can declare: its own buffers void stage<2>(double const*, double*)::buf (20480 bytes); the shadows of its buffers void stage<2>(double const*, double*)::buf (20480 bytes); the shadows of other kernels' buffers void stage<1>(double const*, double*)::buf (20480 bytes). Profile the kernel with smaller shared buffers, or the kernels whose shadows it reaches in another translation unit.

// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: not %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

// shared_limit_shadows_nvptx.cu for HIP: each kernel's buffer and its shadow
// fit the local data share a gfx942 workgroup addresses (64 KiB), but the
// derivative of the second kernel also reaches the shadow of the first
// kernel's buffer, which takes it above the limit: the compile stops with an
// error naming all three before the backend refuses the module.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

template <int K>
__global__ POSEIDON_OPTIMIZE void stage(const double *x, double *y) {
  __shared__ double buf[3072];
  for (int i = threadIdx.x; i < 3072; i += blockDim.x)
    buf[i] = x[i] * x[i] + K;
  for (int i = threadIdx.x; i < 3072; i += blockDim.x)
    y[i] = buf[3071 - i] * 3.0;
}

template __global__ void stage<1>(const double *, double *);
template __global__ void stage<2>(const double *, double *);

// CHECK: shared_limit_shadows_amdgpu.hip:[[@LINE-11]]:35: error: Poseidon: void stage<2>(double const*, double*) reaches 73728 bytes of static shared memory once profiled, above the 65536 a kernel can declare: its own buffers void stage<2>(double const*, double*)::buf (24576 bytes); the shadows of its buffers void stage<2>(double const*, double*)::buf (24576 bytes); the shadows of other kernels' buffers void stage<1>(double const*, double*)::buf (24576 bytes). Profile the kernel with smaller shared buffers, or the kernels whose shadows it reaches in another translation unit.

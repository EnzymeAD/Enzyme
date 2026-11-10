// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%FPProfileInc %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
// REQUIRES: poseidon, enzyme, amdgpu-registered-target, rocm

// shared_limit_nvptx.cu for HIP: a kernel whose shared buffers, with the
// shadow profiling keeps next to each, exceed the local data share a gfx942
// workgroup addresses (64 KiB, not CUDA's 48 KiB of static shared memory) is
// not profiled and keeps its body.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

template <int N>
__global__ POSEIDON_OPTIMIZE void stage(const double *x, double *y) {
  __shared__ double buf[N];
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    buf[i] = x[i] * x[i];
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    y[i] = buf[N - 1 - i] * 3.0;
}

template __global__ void stage<4096>(const double *, double *);
template __global__ void stage<6144>(const double *, double *);

// GEN-NOT: [poseidon-profgen] _Z5stageILi6144EEvPKdPd
// GEN: warning: {{.*}}Poseidon: void stage<6144>(double const*, double*) would declare 98304 bytes of static shared memory once profiled (a shadow next to each floating-point shared buffer), above the 65536 a kernel can declare. The kernel is not profiled and stays as written.
// GEN-NOT: [poseidon-profgen] _Z5stageILi6144EEvPKdPd
// GEN: [poseidon-profgen] _Z5stageILi4096EEvPKdPd: site 0, 2 shadow buffer(s)
// GEN-NOT: [poseidon-profgen] _Z5stageILi6144EEvPKdPd

// DESC: _Z5stageILi4096EEvPKdPd 0
// DESC-NOT: _Z5stageILi6144EEvPKdPd

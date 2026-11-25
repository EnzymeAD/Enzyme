// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/cache -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=IR --implicit-check-not=_ZZ5stageILi6144EEvPKdPdE3buf_shadow %s < %t/gen.ll
// RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC --implicit-check-not=_Z5stageILi6144EEvPKdPd %s

// The derivative a profiled kernel runs keeps a shadow in the local data share
// next to each of its floating-point shared buffers. A kernel whose buffers
// would then exceed the local data share a gfx942 workgroup addresses (64 KiB)
// is not profiled and keeps its body; the others are profiled as usual.

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
// GEN: shared_limit_amdgpu.hip:[[@LINE-12]]:35: warning: Poseidon: void stage<6144>(double const*, double*) would declare 98304 bytes of static shared memory once profiled (a shadow next to each floating-point shared buffer), above the 65536 a kernel can declare. The kernel is not profiled and stays as written.
// GEN-NOT: [poseidon-profgen] _Z5stageILi6144EEvPKdPd
// GEN: {{^}}[poseidon-profgen] _Z5stageILi4096EEvPKdPd: site 0, 2 shadow buffer(s), output(s) 1{{$}}
// GEN-NOT: [poseidon-profgen] _Z5stageILi6144EEvPKdPd

// IR: @[[SHADOW:_ZZ5stageILi4096EEvPKdPdE3buf_shadow]] = {{.*}}addrspace(3) global [4096 x double] undef,
// IR: @_ZZ5stageILi4096EEvPKdPdE3buf = {{.*}}addrspace(3) global [4096 x double] undef, {{.*}}!enzyme_shadow ![[SHMD:[0-9]+]]{{$}}
// IR: @_ZZ5stageILi6144EEvPKdPdE3buf = {{.*}}addrspace(3) global [6144 x double] undef, comdat, align 16{{$}}
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z5stageILi6144EEvPKdPd(
// IR-NOT: call {{.*}}@{{poseidon|diffe}}
// IR: fmul contract double {{%[0-9]+}}, 3.000000e+00
// IR-NOT: call {{.*}}@{{poseidon|diffe}}
// IR: {{^}}}
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z5stageILi4096EEvPKdPd(
// IR: call void @llvm.memset.p3.i64(ptr addrspace(3) {{.*}}@[[SHADOW]], i8 0, i64 32768, i1 false)
// IR: call void @poseidonLogValueCUDA(
// IR: ![[SHMD]] = !{ptr addrspace(3) @[[SHADOW]]}

// DESC: "profile generation": [
// DESC-NEXT: "_Z5stageILi4096EEvPKdPd 0 2 2 0 1 0 8"
// DESC-NEXT: ],
// DESC-NOT: _Z5stageILi6144EEvPKdPd

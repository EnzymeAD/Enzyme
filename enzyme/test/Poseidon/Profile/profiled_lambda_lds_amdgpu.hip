// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%poseidon_profile_inc %loadClangPoseidonEnzyme -Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-print %s -o %t/gen.co 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: llvm-readelf --notes %t/gen.co | FileCheck %s
// RUN: llvm-readelf --symbols %t/gen.co | FileCheck --check-prefix=SYM --implicit-check-not=clEi %s

// AMDGPU allocates in a kernel that makes an unknown call (a profiled one
// calls the probes) the shared memory of every address-taken function. The
// annotated lambdas, otherwise dead, are dropped: the profiled kernel holds its
// buffer and shadow (2 x 16 KiB), each other kernel its own buffer (20 KiB).

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

template <typename BODY> __global__ void forall(int n, BODY body) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    body(i);
}

template <int K> void stage(int n, const double *x, double *y) {
  auto body = [=] __device__(int i) POSEIDON_OPTIMIZE {
    __shared__ double buf[2048];
    buf[threadIdx.x] = x[i] * x[i] + K;
    __syncthreads();
    y[i] = buf[(threadIdx.x + 1) % blockDim.x] * 3.0;
  };
  (void)(void *)forall<decltype(body)>;
}
template void stage<1>(int, const double *, double *);

template <int K> void istage(int n, const int *a, int *b) {
  auto body = [=] __device__(int i) POSEIDON_OPTIMIZE {
    __shared__ int buf[5120];
    buf[threadIdx.x] = a[i] * a[i] + K;
    __syncthreads();
    b[i] = buf[(threadIdx.x + 1) % blockDim.x] * 3;
  };
  (void)(void *)forall<decltype(body)>;
}
template void istage<1>(int, const int *, int *);
template void istage<2>(int, const int *, int *);
template void istage<3>(int, const int *, int *);

// GEN-NOT: error
// GEN-NOT: [poseidon] {{.*}}istage
// GEN: profiled_lambda_lds_amdgpu.hip:[[@LINE-32]]:42: warning: Poseidon: void forall<void istage<1>(int, int const*, int*)::'lambda'(int)>(int, void istage<1>(int, int const*, int*)::'lambda'(int)) holds no floating-point operation the pass profiles; the kernel is not profiled and stays as written{{$}}
// GEN: profiled_lambda_lds_amdgpu.hip:[[@LINE-33]]:42: warning: Poseidon: void forall<void istage<2>(int, int const*, int*)::'lambda'(int)>(int, void istage<2>(int, int const*, int*)::'lambda'(int)) holds no floating-point operation the pass profiles; the kernel is not profiled and stays as written{{$}}
// GEN: profiled_lambda_lds_amdgpu.hip:[[@LINE-34]]:42: warning: Poseidon: void forall<void istage<3>(int, int const*, int*)::'lambda'(int)>(int, void istage<3>(int, int const*, int*)::'lambda'(int)) holds no floating-point operation the pass profiles; the kernel is not profiled and stays as written{{$}}
// GEN-NOT: error
// GEN: {{^}}[poseidon] block-geometry probe placed in 1 profiled function(s){{$}}
// GEN-NOT: error
// GEN-NOT: [poseidon] {{.*}}istage
// GEN: {{^}}[poseidon] _Z6forallIZ5stageILi1EEviPKdPdEUliE_EviT_: site 0, 2 shadow buffer(s), output(s) 1[+8]{{$}}
// GEN-NOT: error
// GEN-NOT: {{shadow (buffer|object)\(s\)}}
// CHECK: amdhsa.kernels:
// CHECK: .group_segment_fixed_size: 20480{{$}}
// CHECK: .name: _Z6forallIZ6istageILi1EEviPKiPiEUliE_EviT_{{$}}
// CHECK: .group_segment_fixed_size: 20480{{$}}
// CHECK: .name: _Z6forallIZ6istageILi2EEviPKiPiEUliE_EviT_{{$}}
// CHECK: .group_segment_fixed_size: 20480{{$}}
// CHECK: .name: _Z6forallIZ6istageILi3EEviPKiPiEUliE_EviT_{{$}}
// CHECK: .group_segment_fixed_size: 32768{{$}}
// CHECK: .name: _Z6forallIZ5stageILi1EEviPKdPdEUliE_EviT_{{$}}
// CHECK-NOT: .group_segment_fixed_size:

// The probes the profiled kernel calls are linked into the code object.
// SYM-DAG: FUNC LOCAL DEFAULT {{[0-9]+}} __poseidon_device_log_value{{$}}
// SYM-DAG: FUNC LOCAL DEFAULT {{[0-9]+}} __poseidon_device_log_grad{{$}}

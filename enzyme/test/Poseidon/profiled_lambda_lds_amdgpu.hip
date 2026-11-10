// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -I%FPProfileInc %clangLoadPoseidonEnzyme -Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -poseidon-cache=%t/c %s -o %t/gen.co 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: llvm-readelf --notes %t/gen.co | FileCheck %s
// REQUIRES: poseidon, enzyme, amdgpu-registered-target, rocm

// A profiled AMDGPU kernel calls the probes, and AMDGPU allocates the shared
// memory of every function whose address is taken in each kernel that may
// make an unknown call. An annotation takes the address of the lambda it
// names, which inlining has otherwise left dead: kept, every profiled kernel
// below would allocate all four lambdas' buffers and exceed the 64 KiB a
// workgroup addresses. Each allocates its own buffer and its shadow.

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
template void stage<2>(int, const double *, double *);
template void stage<3>(int, const double *, double *);
template void stage<4>(int, const double *, double *);

// GEN-COUNT-4: [poseidon-profgen] {{.*}}forall{{.*}}: site {{[0-3]}}, 2 shadow buffer(s)
// GEN-NOT: error

// CHECK-COUNT-4: .group_segment_fixed_size: 32768

// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/other_sites %s -o %t/k.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/k.ll
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll

// per_call_nvptx.cu for HIP: in AMDGPU device code a per-call site is a site
// as on the host, with the shadow of every local object a call passes made by
// the pass, and one shadow for two pointers into device memory at a constant
// distance.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

struct holder {
  const double *c;
};

__device__ double sq(double x) { return x * x + 1.0; }

__device__ void scale(const double *x, double *y) {
  y[0] = x[0] * x[0] + x[1];
  y[1] = x[1] * x[1];
}

__device__ void weigh(const holder &h, const double *x, double *y) {
  y[0] = h.c[0] * x[0] * x[0];
}

__global__ void k(double *x) {
  x[0] = __poseidon_optimize(sq, x[1]);
  double a[2] = {x[2], x[3]}, b[2];
  __poseidon_optimize(scale, a, b);
  x[4] = b[0] + b[1];
  __poseidon_optimize(scale, x + 5, x + 7);
  holder h{x + 9};
  double c, d = x[10];
  __poseidon_optimize(weigh, h, &d, &c);
  x[11] = c;
}

// CHECK-DAG: [poseidon] _Z2sqd.poseidon._Z1kPd.0: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// CHECK-DAG: [poseidon] _Z5scalePKdPd.poseidon._Z1kPd.1: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// CHECK-DAG: [poseidon] _Z5weighRK6holderPKdPd.poseidon._Z1kPd.3: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// CHECK-DAG: [poseidon] _Z5scalePKdPd.poseidon._Z1kPd.2: the profiled program did not contain this site (no profile at {{.*}}); left unchanged
// CHECK-NOT: is not a Poseidon site
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z1kPd(
// IR-NOT: call
// IR: ret void
// IR-NOT: __poseidon_fp_optimize
// IR-NOT: __poseidon_site

// GEN-DAG: [poseidon-profgen] scale(double const*, double*) at call 1 in k(double*): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] scale(double const*, double*) at call 2 in k(double*): 1 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] weigh(holder const&, double const*, double*) at call 3 in k(double*): 2 shadow object(s), output(s) 2
// GENIR-LABEL: define {{.*}}amdgpu_kernel void @_Z1kPd(
// GENIR: call void @llvm.memset
// GENIR: store double 1.000000e+00
// GENIR: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1(ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}})
// x + 5 and x + 7 share one shadow, the second 16 bytes into it, as far as the
// pointers are apart.
// GENIR: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[X:[0-9]+]], i32 16
// GENIR: %[[XG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[X]] to ptr
// GENIR: %[[YG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[Y]] to ptr
// GENIR: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2(ptr {{.*}}, ptr %[[XG]], ptr {{.*}}, ptr %[[YG]])
// The holder has no shadow; d and c do.
// GENIR: call {{.*}} @diffe{{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3(ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}}, ptr {{%[0-9]+}})
// GENIR: ret void

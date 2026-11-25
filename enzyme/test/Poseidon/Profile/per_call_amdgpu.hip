// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/k.ll 2>&1 | FileCheck %s
// RUN: FileCheck --check-prefix=IR %s < %t/k.ll
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll

// In AMDGPU device code a per-call site is a site as on the host. The pass
// makes the shadow of every local object a call passes in private memory,
// zero, with the seed on what the call writes; two pointers into device memory
// at a constant distance share one shadow, which covers what the call reaches
// through them; and an object holding pointers is passed as a constant.

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

// CHECK-DAG: per_call_amdgpu.hip:[[@LINE-12]]:17: warning: Poseidon: no profile at {{.*}}_Z2sqd.poseidon._Z1kPd.0.fpprofile:
// CHECK-DAG: per_call_amdgpu.hip:[[@LINE-13]]:17: warning: Poseidon: no profile at {{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1.fpprofile:
// CHECK-DAG: per_call_amdgpu.hip:[[@LINE-14]]:17: warning: Poseidon: no profile at {{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2.fpprofile:
// CHECK-DAG: per_call_amdgpu.hip:[[@LINE-15]]:17: warning: Poseidon: no profile at {{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3.fpprofile:
// CHECK-NOT: cannot be a Poseidon site
// CHECK: {{^}}4 warnings generated when compiling for {{.+}}.{{$}}
// IR-LABEL: define {{.*}}amdgpu_kernel void @_Z1kPd(
// IR-NOT: call
// IR: ret void
// IR-NOT: __poseidon_fp_optimize
// IR-NOT: __poseidon_site

// GEN-DAG: [poseidon-profgen] scale(double const*, double*) at call 1 in k(double*): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] scale(double const*, double*) at call 2 in k(double*): 1 shadow object(s), output(s) 1
// GEN-DAG: [poseidon-profgen] weigh(holder const&, double const*, double*) at call 3 in k(double*): 2 shadow object(s), output(s) 2
// GENIR-LABEL: define {{.*}}amdgpu_kernel void @_Z1kPd(
// GENIR-SAME: ptr addrspace(1) noundef %[[XP:[0-9]+]])
// GENIR-DAG: %[[X:[0-9]+]] = alloca [32 x i8], align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[A:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[DA:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[B:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[DB:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[H:[0-9]+]] = alloca %struct.holder, align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[C:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[DC:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[D:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[DD:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// GENIR-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// GENIR-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// GENIR-DAG: %[[HG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[H]] to ptr
// GENIR-DAG: %[[CG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[C]] to ptr
// GENIR-DAG: %[[DG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[D]] to ptr
// GENIR-DAG: %[[XG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XP]] to ptr
// A call that passes no pointer has no shadow.
// GENIR: %[[X1P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 8
// GENIR: %[[X1:[0-9]+]] = load double, ptr addrspace(1) %[[X1P]],
// GENIR: %[[SQ:[0-9]+]] = {{.*}}call {{.*}}double @diffe{{.*}}_Z2sqd.poseidon._Z1kPd.0(double %[[X1]])
// GENIR: store double %[[SQ]], ptr addrspace(1) %[[XP]],
// a and b are local: a holds x[2] and x[3], its shadow is zero; b is read
// after the call, its shadow has the seed in both elements the call writes.
// GENIR: %[[X2P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 16
// GENIR: %[[X2:[0-9]+]] = load double, ptr addrspace(1) %[[X2P]],
// GENIR: store double %[[X2]], ptr addrspace(5) %[[A]],
// GENIR: %[[A1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[A]], i32 8
// GENIR: %[[X3P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 24
// GENIR: %[[X3:[0-9]+]] = load double, ptr addrspace(1) %[[X3P]],
// GENIR: store double %[[X3]], ptr addrspace(5) %[[A1]],
// GENIR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[DA]], i8 0, i64 16, i1 false)
// GENIR: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// GENIR: %[[DB1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[DB]], i32 8
// GENIR: store double 1.000000e+00, ptr addrspace(5) %[[DB1]],
// GENIR: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// GENIR: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// GENIR: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1(ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// GENIR: load double, ptr addrspace(5) %[[B]],
// GENIR: %[[B1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[B]], i32 8
// GENIR: load double, ptr addrspace(5) %[[B1]],
// x + 5 and x + 7 share one shadow of the 32 bytes the call reaches, the second
// 16 bytes into it, as far as the pointers are apart: zero where the call reads,
// the seed where it writes.
// GENIR: %[[X5:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 40
// GENIR: %[[X7:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 56
// GENIR: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[X]], i8 0, i64 16, i1 false)
// GENIR: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[X]], i32 16
// GENIR: store double 1.000000e+00, ptr addrspace(5) %[[Y]],
// GENIR: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[X]], i32 24
// GENIR: store double 1.000000e+00, ptr addrspace(5) %[[Y1]],
// GENIR: %[[SXG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[X]] to ptr
// GENIR: %[[SYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[Y]] to ptr
// GENIR: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2(ptr {{(nonnull )?}}%[[X5]], ptr %[[SXG]], ptr {{(nonnull )?}}%[[X7]], ptr %[[SYG]])
// The holder, which holds x + 9, has no shadow; d holds x[10] and its shadow is
// zero; c is read after the call and its shadow seeded.
// GENIR: %[[X9:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 72
// GENIR: store ptr %[[X9]], ptr addrspace(5) %[[H]],
// GENIR: %[[X10P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 80
// GENIR: %[[X10:[0-9]+]] = load double, ptr addrspace(1) %[[X10P]],
// GENIR: store double %[[X10]], ptr addrspace(5) %[[D]],
// GENIR: store i64 0, ptr addrspace(5) %[[DD]],
// GENIR: store double 1.000000e+00, ptr addrspace(5) %[[DC]],
// GENIR: %[[DDG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DD]] to ptr
// GENIR: %[[DCG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DC]] to ptr
// GENIR: call {{.*}} @diffe{{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3(ptr %[[HG]], ptr %[[DG]], ptr %[[DDG]], ptr %[[CG]], ptr %[[DCG]])
// GENIR: %[[CV:[0-9]+]] = load double, ptr addrspace(5) %[[C]],
// GENIR: %[[X11P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 88
// GENIR: store double %[[CV]], ptr addrspace(1) %[[X11P]],
// GENIR: ret void

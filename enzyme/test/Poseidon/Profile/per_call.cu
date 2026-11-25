// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// DEFINE: %{nvptx} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O1 -S -emit-llvm -I%poseidon_profile_inc
// RUN: rm -rf %t && mkdir %t
// RUN: %if nvptx-device-profiler %{ %{nvptx} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/nvptx.ll 2>&1 | FileCheck %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefixes=IR,IR-NVPTX %s < %t/nvptx.ll %}
// RUN: %if nvptx-device-profiler %{ %{nvptx} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/nvptx-c -mllvm -poseidon-print %s -o %t/nvptx-gen.ll 2>&1 | FileCheck --check-prefix=GEN %s %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=GENIR-NVPTX %s < %t/nvptx-gen.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/amdgpu.ll 2>&1 | FileCheck %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=IR,IR-AMDGPU %s < %t/amdgpu.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/amdgpu-c -mllvm -poseidon-print %s -o %t/amdgpu-gen.ll 2>&1 | FileCheck --check-prefix=GEN %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=GENIR-AMDGPU %s < %t/amdgpu-gen.ll %}

// In device code a per-call site is a site as on the host. The pass makes the
// shadow of every local object a call passes (in private memory on AMDGPU)
// zero, with the seed on what the call writes; two pointers into device memory
// at a constant distance share one shadow, which covers what the call reaches
// through them; and an object holding pointers is passed as a constant.

#include "poseidon/poseidon.h"

struct holder {
  const double *c;
};

__attribute__((device)) double sq(double x) { return x * x + 1.0; }

__attribute__((device)) void scale(const double *x, double *y) {
  y[0] = x[0] * x[0] + x[1];
  y[1] = x[1] * x[1];
}

__attribute__((device)) void weigh(const holder &h, const double *x,
                                   double *y) {
  y[0] = h.c[0] * x[0] * x[0];
}

__attribute__((global)) void k(double *x) {
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

// CHECK-DAG: per_call.cu:[[@LINE-12]]:30: warning: Poseidon: no profile at {{.*}}_Z2sqd.poseidon._Z1kPd.0.fpprofile:
// CHECK-DAG: per_call.cu:[[@LINE-13]]:30: warning: Poseidon: no profile at {{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1.fpprofile:
// CHECK-DAG: per_call.cu:[[@LINE-14]]:30: warning: Poseidon: no profile at {{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2.fpprofile:
// CHECK-DAG: per_call.cu:[[@LINE-15]]:30: warning: Poseidon: no profile at {{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3.fpprofile:
// CHECK-NOT: cannot be a Poseidon site
// CHECK: {{^}}4 warnings generated when compiling for {{.+}}.{{$}}
// IR-NVPTX-LABEL: define {{.*}}ptx_kernel void @_Z1kPd(
// IR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z1kPd(
// IR-NOT: call
// IR: ret void
// IR-NOT: __poseidon_fp_optimize
// IR-NOT: __poseidon_site

// GEN-DAG: [poseidon] scale(double const*, double*) at call 1 in k(double*): 2 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] scale(double const*, double*) at call 2 in k(double*): 1 shadow object(s), output(s) 1
// GEN-DAG: [poseidon] weigh(holder const&, double const*, double*) at call 3 in k(double*): 2 shadow object(s), output(s) 2
// GENIR-NVPTX-LABEL: define {{.*}} @_Z1kPd(
// GENIR-NVPTX-SAME: ptr {{.*}}%[[XP:[0-9]+]])
// GENIR-NVPTX-DAG: %[[X:[0-9]+]] = alloca [32 x i8], align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[A:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[DA:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[B:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[DB:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[H:[0-9]+]] = alloca %struct.holder, align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[C:[0-9]+]] = alloca double, align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[DC:[0-9]+]] = alloca double, align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[D:[0-9]+]] = alloca double, align {{[0-9]+}}
// GENIR-NVPTX-DAG: %[[DD:[0-9]+]] = alloca double, align {{[0-9]+}}
// A call that passes no pointer has no shadow.
// GENIR-NVPTX: %[[X1P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 8
// GENIR-NVPTX: %[[X1:[0-9]+]] = load double, ptr %[[X1P]],
// GENIR-NVPTX: %[[SQ:[0-9]+]] = {{.*}}call {{.*}}double @diffe{{.*}}_Z2sqd.poseidon._Z1kPd.0(double %[[X1]])
// GENIR-NVPTX: store double %[[SQ]], ptr %[[XP]],
// a and b are local: a holds x[2] and x[3], its shadow is zero; b is read
// after the call, its shadow has the seed in both elements the call writes.
// GENIR-NVPTX: %[[X2P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 16
// GENIR-NVPTX: %[[X2:[0-9]+]] = load double, ptr %[[X2P]],
// GENIR-NVPTX: store double %[[X2]], ptr %[[A]],
// GENIR-NVPTX: %[[A1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[A]], i64 8
// GENIR-NVPTX: %[[X3P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 24
// GENIR-NVPTX: %[[X3:[0-9]+]] = load double, ptr %[[X3P]],
// GENIR-NVPTX: store double %[[X3]], ptr %[[A1]],
// GENIR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[DA]], i8 0, i64 16, i1 false)
// GENIR-NVPTX: store double 1.000000e+00, ptr %[[DB]],
// GENIR-NVPTX: %[[DB1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[DB]], i64 8
// GENIR-NVPTX: store double 1.000000e+00, ptr %[[DB1]],
// GENIR-NVPTX: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1(ptr %[[A]], ptr %[[DA]], ptr %[[B]], ptr %[[DB]])
// GENIR-NVPTX: load double, ptr %[[B]],
// GENIR-NVPTX: %[[B1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[B]], i64 8
// GENIR-NVPTX: load double, ptr %[[B1]],
// x + 5 and x + 7 share one shadow of the 32 bytes the call reaches, the second
// 16 bytes into it, as far as the pointers are apart: zero where the call reads,
// the seed where it writes.
// GENIR-NVPTX: %[[X5:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 40
// GENIR-NVPTX: %[[X7:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 56
// GENIR-NVPTX: call void @llvm.memset.p0.i64(ptr {{.*}}%[[X]], i8 0, i64 16, i1 false)
// GENIR-NVPTX: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X]], i64 16
// GENIR-NVPTX: store double 1.000000e+00, ptr %[[Y]],
// GENIR-NVPTX: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[X]], i64 24
// GENIR-NVPTX: store double 1.000000e+00, ptr %[[Y1]],
// GENIR-NVPTX: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2(ptr {{(nonnull )?}}%[[X5]], ptr %[[X]], ptr {{(nonnull )?}}%[[X7]], ptr %[[Y]])
// The holder, which holds x + 9, has no shadow; d holds x[10] and its shadow is
// zero; c is read after the call and its shadow seeded.
// GENIR-NVPTX: %[[X9:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 72
// GENIR-NVPTX: store ptr %[[X9]], ptr %[[H]],
// GENIR-NVPTX: %[[X10P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 80
// GENIR-NVPTX: %[[X10:[0-9]+]] = load double, ptr %[[X10P]],
// GENIR-NVPTX: store double %[[X10]], ptr %[[D]],
// GENIR-NVPTX: store i64 0, ptr %[[DD]],
// GENIR-NVPTX: store double 1.000000e+00, ptr %[[DC]],
// GENIR-NVPTX: call {{.*}} @diffe{{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3(ptr %[[H]], ptr %[[D]], ptr %[[DD]], ptr %[[C]], ptr %[[DC]])
// GENIR-NVPTX: %[[CV:[0-9]+]] = load double, ptr %[[C]],
// GENIR-NVPTX: %[[X11P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XP]], i64 88
// GENIR-NVPTX: store double %[[CV]], ptr %[[X11P]],
// GENIR-NVPTX: ret void

// The same on AMDGPU, where the locals and their shadows are in private memory
// and reach the derivative through casts to generic pointers.
// GENIR-AMDGPU-LABEL: define {{.*}}amdgpu_kernel void @_Z1kPd(
// GENIR-AMDGPU-SAME: ptr addrspace(1) noundef %[[XP:[0-9]+]])
// GENIR-AMDGPU-DAG: %[[X:[0-9]+]] = alloca [32 x i8], align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[A:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[DA:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[B:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[DB:[0-9]+]] = alloca [2 x double], align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[H:[0-9]+]] = alloca %struct.holder, align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[C:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[DC:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[D:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[DD:[0-9]+]] = alloca double, align {{[0-9]+}}, addrspace(5)
// GENIR-AMDGPU-DAG: %[[AG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[A]] to ptr
// GENIR-AMDGPU-DAG: %[[BG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[B]] to ptr
// GENIR-AMDGPU-DAG: %[[HG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[H]] to ptr
// GENIR-AMDGPU-DAG: %[[CG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[C]] to ptr
// GENIR-AMDGPU-DAG: %[[DG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[D]] to ptr
// GENIR-AMDGPU-DAG: %[[XG:[0-9]+]] = addrspacecast ptr addrspace(1) %[[XP]] to ptr
// GENIR-AMDGPU: %[[X1P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 8
// GENIR-AMDGPU: %[[X1:[0-9]+]] = load double, ptr addrspace(1) %[[X1P]],
// GENIR-AMDGPU: %[[SQ:[0-9]+]] = {{.*}}call {{.*}}double @diffe{{.*}}_Z2sqd.poseidon._Z1kPd.0(double %[[X1]])
// GENIR-AMDGPU: store double %[[SQ]], ptr addrspace(1) %[[XP]],
// GENIR-AMDGPU: %[[X2P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 16
// GENIR-AMDGPU: %[[X2:[0-9]+]] = load double, ptr addrspace(1) %[[X2P]],
// GENIR-AMDGPU: store double %[[X2]], ptr addrspace(5) %[[A]],
// GENIR-AMDGPU: %[[A1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[A]], i32 8
// GENIR-AMDGPU: %[[X3P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 24
// GENIR-AMDGPU: %[[X3:[0-9]+]] = load double, ptr addrspace(1) %[[X3P]],
// GENIR-AMDGPU: store double %[[X3]], ptr addrspace(5) %[[A1]],
// GENIR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[DA]], i8 0, i64 16, i1 false)
// GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB]],
// GENIR-AMDGPU: %[[DB1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[DB]], i32 8
// GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DB1]],
// GENIR-AMDGPU: %[[DAG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DA]] to ptr
// GENIR-AMDGPU: %[[DBG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DB]] to ptr
// GENIR-AMDGPU: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.1(ptr %[[AG]], ptr %[[DAG]], ptr %[[BG]], ptr %[[DBG]])
// GENIR-AMDGPU: load double, ptr addrspace(5) %[[B]],
// GENIR-AMDGPU: %[[B1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[B]], i32 8
// GENIR-AMDGPU: load double, ptr addrspace(5) %[[B1]],
// GENIR-AMDGPU: %[[X5:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 40
// GENIR-AMDGPU: %[[X7:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 56
// GENIR-AMDGPU: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}%[[X]], i8 0, i64 16, i1 false)
// GENIR-AMDGPU: %[[Y:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[X]], i32 16
// GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[Y]],
// GENIR-AMDGPU: %[[Y1:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(5) %[[X]], i32 24
// GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[Y1]],
// GENIR-AMDGPU: %[[SXG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[X]] to ptr
// GENIR-AMDGPU: %[[SYG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[Y]] to ptr
// GENIR-AMDGPU: call {{.*}} @diffe{{.*}}_Z5scalePKdPd.poseidon._Z1kPd.2(ptr {{(nonnull )?}}%[[X5]], ptr %[[SXG]], ptr {{(nonnull )?}}%[[X7]], ptr %[[SYG]])
// GENIR-AMDGPU: %[[X9:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr %[[XG]], i64 72
// GENIR-AMDGPU: store ptr %[[X9]], ptr addrspace(5) %[[H]],
// GENIR-AMDGPU: %[[X10P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 80
// GENIR-AMDGPU: %[[X10:[0-9]+]] = load double, ptr addrspace(1) %[[X10P]],
// GENIR-AMDGPU: store double %[[X10]], ptr addrspace(5) %[[D]],
// GENIR-AMDGPU: store i64 0, ptr addrspace(5) %[[DD]],
// GENIR-AMDGPU: store double 1.000000e+00, ptr addrspace(5) %[[DC]],
// GENIR-AMDGPU: %[[DDG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DD]] to ptr
// GENIR-AMDGPU: %[[DCG:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DC]] to ptr
// GENIR-AMDGPU: call {{.*}} @diffe{{.*}}_Z5weighRK6holderPKdPd.poseidon._Z1kPd.3(ptr %[[HG]], ptr %[[DG]], ptr %[[DDG]], ptr %[[CG]], ptr %[[DCG]])
// GENIR-AMDGPU: %[[CV:[0-9]+]] = load double, ptr addrspace(5) %[[C]],
// GENIR-AMDGPU: %[[X11P:[0-9]+]] = getelementptr inbounds{{.*}} i8, ptr addrspace(1) %[[XP]], i64 88
// GENIR-AMDGPU: store double %[[CV]], ptr addrspace(1) %[[X11P]],
// GENIR-AMDGPU: ret void

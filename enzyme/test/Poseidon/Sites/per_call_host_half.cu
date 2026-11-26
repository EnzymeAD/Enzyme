// REQUIRES: enzyme
// REQUIRES: cuda-toolkit || (amdgpu-registered-target && rocm)
// Without a CUDA installation clang emits the old cudaLaunch API in the host
// half of a CUDA compile.
// DEFINE: %{nvptx} = %clang -x cuda --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -emit-llvm -I%poseidon_profile_inc %s
// DEFINE: %{amdgpu} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %s
// DEFINE: %{nvptx_body} = sed -e '/__poseidon_profile_init_device/d' -e '/^; Function Attrs/d' \
// DEFINE:   -e '/^attributes #/d' -e '/^$/d' -e 's/ #[0-9][0-9]*//g'
// DEFINE: %{amdgpu_body} = sed -e '/^define internal void @__poseidon_profile_register_module_table/,/^}/d' \
// DEFINE:   -e '/__poseidon_profile/d' -e '/__hipRegisterVar/d' -e '/@llvm.global_ctors/d' \
// DEFINE:   -e '/^; Function Attrs/d' -e '/^attributes #/d' -e '/^$/d' -e 's/ #[0-9][0-9]*//g'
// RUN: rm -rf %t && mkdir %t
// RUN: %if cuda-toolkit %{ %{nvptx} --cuda-host-only -o %t/nvptx-plain.ll %}
// RUN: %if cuda-toolkit %{ %{nvptx} --cuda-host-only %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-cache=%t/nvptx-u -o %t/nvptx-use.ll 2>&1 | FileCheck --allow-empty --check-prefix=HOST %s %}
// RUN: %if cuda-toolkit %{ diff %t/nvptx-plain.ll %t/nvptx-use.ll %}
// RUN: %if cuda-toolkit %{ %{nvptx} --cuda-host-only %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/nvptx-g -mllvm -poseidon-print -o %t/nvptx-gen.ll 2>&1 | FileCheck --check-prefix=GEN %s %}
// RUN: %if cuda-toolkit %{ FileCheck --check-prefix=GENIR %s < %t/nvptx-gen.ll %}
// RUN: %if cuda-toolkit %{ FileCheck --check-prefix=BIND %s < %t/nvptx-gen.ll %}
// RUN: %if cuda-toolkit %{ %{nvptx_body} %t/nvptx-plain.ll > %t/nvptx-plain.body %}
// RUN: %if cuda-toolkit %{ %{nvptx_body} %t/nvptx-gen.ll > %t/nvptx-gen.body %}
// RUN: %if cuda-toolkit %{ diff %t/nvptx-plain.body %t/nvptx-gen.body %}
// RUN: %if cuda-toolkit %{ %{nvptx} --cuda-device-only %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/nvptx-d -mllvm -poseidon-print -o %t/nvptx-dev.ll 2>&1 | FileCheck --check-prefix=DEV %s %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --cuda-host-only -o %t/amdgpu-plain.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --cuda-host-only %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/unrelated_site_profile -mllvm -poseidon-cache=%t/amdgpu-u -o %t/amdgpu-use.ll 2>&1 | FileCheck --allow-empty --check-prefix=HOST %s %}
// RUN: %if amdgpu-registered-target && rocm %{ diff %t/amdgpu-plain.ll %t/amdgpu-use.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --cuda-host-only %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/amdgpu-g -mllvm -poseidon-print -o %t/amdgpu-gen.ll 2>&1 | FileCheck --check-prefix=GEN %s %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=GENIR %s < %t/amdgpu-gen.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefixes=BIND,BIND-AMDGPU %s < %t/amdgpu-gen.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu_body} %t/amdgpu-plain.ll > %t/amdgpu-plain.body %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu_body} %t/amdgpu-gen.ll > %t/amdgpu-gen.body %}
// RUN: %if amdgpu-registered-target && rocm %{ diff %t/amdgpu-plain.body %t/amdgpu-gen.body %}
// RUN: %if amdgpu-registered-target && rocm %{ %{amdgpu} --cuda-device-only %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/amdgpu-d -mllvm -poseidon-print -o %t/amdgpu-dev.ll 2>&1 | FileCheck --check-prefix=DEV %s %}

// In a CUDA or HIP compilation only the device half holds sites: the host half
// of a host-device function with either form, and a host function, compile as
// written, silently, in profile generation and use.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))
// Keeps qpoint, which the kernel inlines, on AMDGPU as on NVPTX.
#define KEEP __attribute__((used))

#ifdef __NVPTX__
#define THREAD_X __nvvm_read_ptx_sreg_tid_x()
#else
#define THREAD_X __builtin_amdgcn_workitem_id_x()
#endif

#include "Inputs/kernel_stub_runtime.h"

HD void vol(const double *a, double *v) {
  double J = a[0] * a[1] - a[2] * a[3];
  *v = J * (J - 1.0);
}

HD KEEP double qpoint(const double *g) {
  double a[4] = {g[0], g[1], g[2], g[3]}, v;
  __poseidon_optimize(vol, a, &v);
  return 2.0 * v;
}

POSEIDON_OPTIMIZE HD double area(double r) { return 3.14159 * r * r; }

double host_only(const double *g) {
  double a[4] = {g[3], g[2], g[1], g[0]}, v;
  __poseidon_optimize(vol, a, &v);
  return v;
}

__attribute__((global)) void k(const double *g, double *r, int n) {
  int i = THREAD_X;
  if (i < n)
    r[i] = qpoint(g + 4 * i);
}

int main() {
  const int n = 64;
  static double g[4 * n], r[n];
  for (int i = 0; i < 4 * n; ++i)
    g[i] = 1.0 + 1e-3 * (i % 7);
  for (int i = 0; i < n; ++i)
    r[i] = qpoint(g + 4 * i) + area(g[i]) + host_only(g + 4 * i);
  return r[7] > 0.0;
}

// HOST-NOT: {{.}}

// GEN-NOT: warning
// GEN: [poseidon] device profiler registration injected at main
// GEN-NOT: {{.}}

// GENIR-NOT: call {{.*}}@__poseidon_log_value(
// GENIR-NOT: call {{.*}}@__poseidon_log_grad(
// GENIR-NOT: __poseidon_host_shadows
// GENIR-NOT: __poseidon_register_static
// GENIR-NOT: preprocess_

// Generating, the host half only binds the profiler (main, and on AMDGPU the
// module constructor); the rest is the plain compile's IR.
// BIND-AMDGPU: @__poseidon_profile_table = internal global {{\[}}[[SIZE:[0-9]+]] x i8] zeroinitializer
// BIND-AMDGPU: [[NAME:@[0-9]+]] = private unnamed_addr constant [25 x i8] c"__poseidon_profile_table\00"
// BIND-AMDGPU: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }]
// BIND: define {{.*}}i32 @main()
// BIND-NEXT: call void @__poseidon_profile_init_device()
// BIND-AMDGPU: define internal void @__hip_module_ctor()
// BIND-AMDGPU: call void @__hipRegisterVar(ptr %{{[0-9]+}}, ptr {{.*}}@__poseidon_profile_table, ptr {{.*}}[[NAME]], ptr {{.*}}[[NAME]], i32 0, i64 [[SIZE]], i32 1, i32 0)
// BIND-AMDGPU: define internal void @__poseidon_profile_register_module_table()
// BIND-AMDGPU-NEXT: call void @__poseidon_profile_register_module(ptr {{.*}}@__poseidon_profile_table, i32 0, i32 0)
// BIND: declare void @__poseidon_profile_init_device()

// The call in qpoint is lowered twice: in qpoint and where k inlines it.
// DEV: per_call_host_half.cu:[[@LINE-50]]:{{[0-9]+}}: warning: Poseidon: area(double) is a device function no kernel of this translation unit calls, so it is left as written; on the device, put POSEIDON_OPTIMIZE on a kernel or on a function a kernel calls
// DEV: [poseidon] vol(double const*, double*) at call 0 in qpoint(double const*): 2 shadow object(s), output(s) 1
// DEV-NEXT: [poseidon] vol(double const*, double*) at call 0 in qpoint(double const*): 2 shadow object(s), output(s) 1
// DEV-NEXT: [poseidon] block-geometry probe placed in 1 profiled function(s)
// DEV-NOT: {{shadow (buffer|object)\(s\)}}

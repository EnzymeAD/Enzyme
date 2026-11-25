// REQUIRES: enzyme, amdgpu-registered-target, rocm
// DEFINE: %{body} = sed -e '/^define internal void @__poseidon_profile_register_module_table/,/^}/d' \
// DEFINE:   -e '/__poseidon_profile/d' -e '/__hipRegisterVar/d' -e '/@llvm.global_ctors/d' \
// DEFINE:   -e '/^; Function Attrs/d' -e '/^attributes #/d' -e '/^$/d' -e 's/ #[0-9][0-9]*//g'
// RUN: rm -rf %t && mkdir %t
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %s -o %t/plain.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/u %s -o %t/use.ll 2>&1 | FileCheck --allow-empty --check-prefix=HOST %s
// RUN: diff %t/plain.ll %t/use.ll
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path --cuda-host-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/g -mllvm -poseidon-print %s -o %t/gen.ll 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// RUN: FileCheck --check-prefix=BIND %s < %t/gen.ll
// RUN: %{body} %t/plain.ll > %t/plain.body
// RUN: %{body} %t/gen.ll > %t/gen.body
// RUN: diff %t/plain.body %t/gen.body
// RUN: %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/d -mllvm -poseidon-print %s -o %t/dev.ll 2>&1 | FileCheck --check-prefix=DEV %s

// per_call_host_half_nvptx.cu for HIP: the host half of a HIP compilation
// holds no site of the call or the annotation form.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#define HD __host__ __device__

HD void vol(const double *a, double *v) {
  double J = a[0] * a[1] - a[2] * a[3];
  *v = J * (J - 1.0);
}

HD double qpoint(const double *g) {
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

__global__ void k(const double *g, double *r, int n) {
  int i = threadIdx.x;
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
// GEN: [poseidon] FP profiler CUDA registration injected at main
// GEN-NOT: {{.}}

// GENIR-NOT: call {{.*}}@poseidonLogValue(
// GENIR-NOT: call {{.*}}@poseidonLogGrad(
// GENIR-NOT: __poseidon_host_shadows
// GENIR-NOT: poseidonRegisterProfileStatic
// GENIR-NOT: preprocess_

// In profile generation the host half only binds the profiler: main
// initializes it, and the module constructor registers the device module's
// table with HIP and with the profiler. The rest of the host half is the plain
// compile's IR up to attribute groups.
// BIND: @__poseidon_profile_table = internal global {{\[}}[[SIZE:[0-9]+]] x i8] zeroinitializer
// BIND: [[NAME:@[0-9]+]] = private unnamed_addr constant [25 x i8] c"__poseidon_profile_table\00"
// BIND: @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__poseidon_profile_register_module_table, ptr null }]
// BIND: define {{.*}}i32 @main()
// BIND-NEXT: call void @__poseidon_profile_init_cuda()
// BIND: define internal void @__hip_module_ctor()
// BIND: call void @__hipRegisterVar(ptr %{{[0-9]+}}, ptr {{.*}}@__poseidon_profile_table, ptr {{.*}}[[NAME]], ptr {{.*}}[[NAME]], i32 0, i64 [[SIZE]], i32 1, i32 0)
// BIND: define internal void @__poseidon_profile_register_module_table()
// BIND-NEXT: call void @__poseidon_profile_register_module(ptr {{.*}}@__poseidon_profile_table, i32 0, i32 0)

// DEV: per_call_host_half_amdgpu.hip:[[@LINE-50]]:{{[0-9]+}}: warning: Poseidon: area(double) is a device function no kernel of this translation unit calls, so it is left as written; on the device, put POSEIDON_OPTIMIZE on a kernel or on a function a kernel calls
// DEV: [poseidon] vol(double const*, double*) at call 0 in qpoint(double const*): 2 shadow object(s), output(s) 1
// DEV-NEXT: [poseidon] block-geometry probe placed in 1 profiled function(s)
// DEV-NOT: {{shadow (buffer|object)\(s\)}}

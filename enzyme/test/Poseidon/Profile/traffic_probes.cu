// REQUIRES: enzyme
// REQUIRES: nvptx-device-profiler || (amdgpu-registered-target && rocm)
// DEFINE: %{gen} = %loadClangPoseidonEnzyme -O1 -S -emit-llvm -I%poseidon_profile_inc -mllvm -poseidon-profile-generate
// RUN: rm -rf %t && mkdir %t
// RUN: %if nvptx-device-profiler %{ %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib %{gen} -mllvm -poseidon-cache=%t/n %s -o %t/nvptx.ll %}
// RUN: %if nvptx-device-profiler %{ FileCheck --check-prefix=GENIR -D'PTR=ptr' -DSCOPE=device %s < %t/nvptx.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 %{gen} -mllvm -poseidon-cache=%t/a %s -o %t/amdgpu.ll %}
// RUN: %if amdgpu-registered-target && rocm %{ FileCheck --check-prefix=GENIR -D'PTR=ptr addrspace(1)' -DSCOPE=agent %s < %t/amdgpu.ll %}

// Traffic probes of a profiled kernel: one before every load, store and atomic
// that reaches global memory, with its record, width and kind (0 load, 1
// store, 2 both); records in the order of first access, one per array (idx,
// x, the by-value view's two pointers, y, sum); the static header names the
// launch value of each; shared memory gets none.

#ifdef __HIP__
#include <hip/hip_runtime.h>
#define TID threadIdx.x
#else
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define TID __nvvm_read_ptx_sreg_tid_x()
#define atomicAdd __nvvm_atom_add_gen_d
#endif

#include "poseidon/poseidon.h"

struct View {
  const double *a;
  double *b;
  int n;
};

POSEIDON_OPTIMIZE __global__ void traffic(const int *idx, const double *x,
                                          double *y, View v, double *sum) {
  __shared__ double s[32];
  int i = TID;
  s[i] = x[idx[i]];
  double t = s[(i + 1) % 32] * v.a[i];
  y[i] = t + 1.0;
  v.b[i] = t * 2.0;
  atomicAdd(sum, t);
}

// GENIR: @poseidon.static.[[SITE:preprocess__Z7trafficPKiPKdPd4ViewS3__poseidon_body]] = {{.*}} c"SiteId = 0\0ACanonicalHash = {{[0-9a-f]+}}\0ATrafficArray = 0 0 -1\0ATrafficArray = 1 1 -1\0ATrafficArray = 2 3 0\0ATrafficArray = 3 2 -1\0ATrafficArray = 4 3 8\0ATrafficArray = 5 4 -1\0A\00"
// GENIR: [[A0:%[0-9]+]] = ptrtoint [[PTR]] [[P0:%[0-9]+]] to i64
// GENIR-NEXT: call void @__poseidon_device_log_access(ptr addrspacecast (ptr addrspace(1) @poseidon.site.[[SITE]] to ptr), i32 0, i64 [[A0]], i32 4, i32 0)
// GENIR-NEXT: {{%[0-9]+ = }}load i32, [[PTR]] [[P0]]
// GENIR-NOT: @__poseidon_device_log_access(ptr
// GENIR: [[A1:%[0-9]+]] = ptrtoint [[PTR]] [[P1:%[0-9]+]] to i64
// GENIR-NEXT: call void @__poseidon_device_log_access(ptr addrspacecast (ptr addrspace(1) @poseidon.site.[[SITE]] to ptr), i32 1, i64 [[A1]], i32 8, i32 0)
// GENIR-NEXT: {{%[0-9]+ = }}load double, [[PTR]] [[P1]]
// GENIR-NOT: @__poseidon_device_log_access(ptr
// GENIR: [[A2:%[0-9]+]] = ptrtoint [[PTR]] [[P2:%[0-9]+]] to i64
// GENIR-NEXT: call void @__poseidon_device_log_access(ptr addrspacecast (ptr addrspace(1) @poseidon.site.[[SITE]] to ptr), i32 2, i64 [[A2]], i32 8, i32 0)
// GENIR-NEXT: {{%[0-9]+ = }}load double, [[PTR]] [[P2]]
// GENIR-NOT: @__poseidon_device_log_access(ptr
// GENIR: [[A3:%[0-9]+]] = ptrtoint [[PTR]] [[P3:%[0-9]+]] to i64
// GENIR-NEXT: call void @__poseidon_device_log_access(ptr addrspacecast (ptr addrspace(1) @poseidon.site.[[SITE]] to ptr), i32 3, i64 [[A3]], i32 8, i32 1)
// GENIR-NEXT: store double {{%[0-9]+}}, [[PTR]] [[P3]]
// GENIR-NOT: @__poseidon_device_log_access(ptr
// GENIR: [[A4:%[0-9]+]] = ptrtoint [[PTR]] [[P4:%[0-9]+]] to i64
// GENIR-NEXT: call void @__poseidon_device_log_access(ptr addrspacecast (ptr addrspace(1) @poseidon.site.[[SITE]] to ptr), i32 4, i64 [[A4]], i32 8, i32 1)
// GENIR-NEXT: store double {{%[0-9]+}}, [[PTR]] [[P4]]
// GENIR-NOT: @__poseidon_device_log_access(ptr
// GENIR: [[A5:%[0-9]+]] = ptrtoint [[PTR]] [[P5:%[0-9]+]] to i64
// GENIR-NEXT: call void @__poseidon_device_log_access(ptr addrspacecast (ptr addrspace(1) @poseidon.site.[[SITE]] to ptr), i32 5, i64 [[A5]], i32 8, i32 2)
// GENIR-NEXT: {{%[0-9]+ = }}atomicrmw fadd [[PTR]] [[P5]], double {{%[0-9]+}} syncscope("[[SCOPE]]")
// GENIR-NOT: call void @__poseidon_device_log_access(

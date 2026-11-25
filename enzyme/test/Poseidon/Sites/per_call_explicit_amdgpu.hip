// REQUIRES: enzyme, amdgpu-registered-target, rocm
// DEFINE: %{clang} = %clang -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O1 -S -emit-llvm -I%poseidon_profile_inc %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -o %t/gen.ll
// RUN: FileCheck --check-prefix=GEN --implicit-check-not=__poseidon_site_explicit %s < %t/gen.ll
// RUN: FileCheck --check-prefix=NOMARKER --implicit-check-not=__poseidon_site_explicit %s < %t/gen.ll
// RUN: %{clang} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print -o %t/use.ll 2>&1 | FileCheck --check-prefix=USE %s
// RUN: FileCheck --check-prefix=IR --implicit-check-not=__poseidon_site_explicit %s < %t/use.ll
// RUN: %{clang} -o - | tee %t/plain.ll | FileCheck --check-prefix=PLAIN %s
// RUN: FileCheck --check-prefix=DECL --implicit-check-not='define {{.*}}@__poseidon_site_explicit' %s < %t/plain.ll

// per_call_explicit_nvptx.cu for HIP: in AMDGPU device code an explicit list
// through __poseidon_optimize makes the call a site over the function itself,
// with the shadows the caller gives, and __poseidon_optimize_tau with the
// function's own arguments makes it a per-call site. Without the plugin the
// call goes through an identity nothing defines.

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#define HD __host__ __device__

__device__ int enzyme_dup, enzyme_const, poseidon_tau;

__device__ double sum(const double *x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += x[i] * x[i];
  return s;
}

__device__ void scale(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = x[i] * x[i] + 1.0;
}

HD __attribute__((noinline)) double norm2(const double *x) {
  return x[0] * x[0] + x[1] * x[1];
}

HD double twice(const double *x, double *dx) {
  return 2.0 * __poseidon_optimize(norm2, enzyme_dup, x, dx);
}

HD __attribute__((noinline)) double spread(const double *x) {
  return x[0] * x[1] - x[1] * x[1];
}

HD double third(const double *x) {
  return __poseidon_optimize_tau(1e-7, spread, x);
}

__global__ void k(const double *x, double *dx, double *y, double *dy, int n) {
  y[0] = __poseidon_optimize(sum, enzyme_dup, x, dx, n);
  y[3] = twice(x, dx);
  __poseidon_optimize_tau(1e-6, scale, enzyme_const, x, enzyme_dup, y + 1, dy,
                          n);
  y[2] = __poseidon_optimize_tau(1e-9, sum, enzyme_const, x, n);
  y[4] = third(x);
}

// Profile generation leaves no call of the identity anywhere in the module.
// NOMARKER: target triple = "{{(amdgcn|amdgpu)[^"]*}}-amd-amdhsa"

// GEN-DAG: @poseidon_site_preprocess__Z3sumPKdi = private unnamed_addr addrspace(1) constant [22 x i8] c"preprocess__Z3sumPKdi\00"
// GEN-DAG: @poseidon_site_preprocess__Z5scalePKdPdi = private unnamed_addr addrspace(1) constant [26 x i8] c"preprocess__Z5scalePKdPdi\00"
// GEN-DAG: @poseidon_site_preprocess__Z5norm2PKd = private unnamed_addr addrspace(1) constant [23 x i8] c"preprocess__Z5norm2PKd\00"
// GEN-DAG: @poseidon_site_preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0 = private unnamed_addr addrspace(1) constant [47 x i8] c"preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0\00"

// Each site's adjoint reaches the shadow the call gives it: dx for sum and
// norm2, dy for scale, and a zeroed local the pass makes for spread's x. The
// kernel holds norm2's call on both of its paths around the loop of sum.
// GEN: define protected amdgpu_kernel void @_Z1kPKdPdS1_S1_i(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}[[DX:%[0-9]+]], ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// GEN-DAG: call void @poseidonProfileStaticCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z3sumPKdi to ptr), ptr addrspacecast (ptr addrspace(1) @fpprofile_static_preprocess__Z3sumPKdi to ptr))
// GEN-DAG: [[DXI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DX]], i64 {{%[0-9]+}}
// GEN-DAG: atomicrmw fadd ptr addrspace(1) [[DXI]], double {{%[0-9]+}} monotonic
// GEN-DAG: call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr {{.*}}[[XN:%[0-9]+]], ptr [[DXN:%[0-9]+]])
// GEN-DAG: [[XN]] = addrspacecast ptr addrspace(1) [[X]] to ptr
// GEN-DAG: [[DXN]] = addrspacecast ptr addrspace(1) [[DX]] to ptr
// GEN-DAG: call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr {{.*}}[[XN2:%[0-9]+]], ptr [[DXN2:%[0-9]+]])
// GEN-DAG: [[XN2]] = addrspacecast ptr addrspace(1) [[X]] to ptr
// GEN-DAG: [[DXN2]] = addrspacecast ptr addrspace(1) [[DX]] to ptr
// GEN-DAG: call void @poseidonProfileStaticCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z5scalePKdPdi to ptr), ptr addrspacecast (ptr addrspace(1) @fpprofile_static_preprocess__Z5scalePKdPdi to ptr))
// GEN-DAG: [[DYI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[DY]], i64 {{%[0-9]+}}
// GEN-DAG: [[DYV:%[0-9]+]] = load double, ptr addrspace(1) [[DYI]]
// GEN-DAG: store double 0.000000e+00, ptr addrspace(1) [[DYI]]
// GEN-DAG: call void @poseidonLogGradCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z5scalePKdPdi to ptr), i64 0, double {{%[0-9]+}}, double [[DYV]])
// GEN-DAG: call {{.*}}double @diffepreprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0(ptr [[XS:%[0-9]+]], ptr [[SH:%[0-9]+]])
// GEN-DAG: [[XS]] = addrspacecast ptr addrspace(1) [[X]] to ptr
// GEN-DAG: [[SH]] = addrspacecast ptr addrspace(5) [[SHA:%[0-9]+]] to ptr
// GEN-DAG: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}[[SHA]], i8 0, i64 16, i1 false)
// GEN: {{^}}}

// USE: [poseidon] site spread(double const*) at its call in third(double const*): accuracy target 1e-07
// USE-DAG: per_call_explicit_amdgpu.hip:[[@LINE-42]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z3sumPKdi.fpprofile:
// USE-DAG: per_call_explicit_amdgpu.hip:[[@LINE-43]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z3sumPKdi.fpprofile:
// USE-DAG: per_call_explicit_amdgpu.hip:[[@LINE-44]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z5norm2PKd.fpprofile:
// USE-DAG: per_call_explicit_amdgpu.hip:[[@LINE-45]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z5scalePKdPdi.fpprofile:
// USE-DAG: per_call_explicit_amdgpu.hip:[[@LINE-46]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0.fpprofile:
// USE: [poseidon] every site of {{.*}}per_call_explicit_amdgpu.hip was left alone; emitting the module as written
// USE-NEXT: 5 warnings generated when compiling for {{.*}}

// Each call left alone calls the function itself with the primal arguments.
// IR: define protected amdgpu_kernel void @_Z1kPKdPdS1_S1_i(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}%{{[0-9]+}}, ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}%{{[0-9]+}}, i32 {{.*}}[[N:%[0-9]+]])
// IR-NEXT: [[XI:%[0-9]+]] = ptrtoint ptr addrspace(1) [[X]] to i64
// IR-NEXT: [[XP:%[0-9]+]] = inttoptr i64 [[XI]] to ptr
// IR-NEXT: [[YG:%[0-9]+]] = addrspacecast ptr addrspace(1) [[Y]] to ptr
// IR-NEXT: [[S0:%[0-9]+]] = {{.*}}call double @_Z3sumPKdi(ptr [[XP]], i32 [[N]])
// IR-NEXT: store double [[S0]], ptr addrspace(1) [[Y]]
// IR-NEXT: [[NR:%[0-9]+]] = {{.*}}call double @_Z5norm2PKd(ptr [[XP]])
// IR-NEXT: [[TW:%[0-9]+]] = fmul {{.*}}double [[NR]], 2.000000e+00
// IR-NEXT: [[Y3:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 24
// IR-NEXT: store double [[TW]], ptr addrspace(1) [[Y3]]
// IR-NEXT: [[Y1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[YG]], i64 8
// IR-NEXT: {{.*}}call void @_Z5scalePKdPdi(ptr [[XP]], ptr {{.*}}[[Y1]], i32 [[N]])
// IR-NEXT: [[S1:%[0-9]+]] = {{.*}}call double @_Z3sumPKdi(ptr [[XP]], i32 [[N]])
// IR-NEXT: [[Y2:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 16
// IR-NEXT: store double [[S1]], ptr addrspace(1) [[Y2]]
// IR-NEXT: [[SP:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[XP]])
// IR-NEXT: [[Y4:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 32
// IR-NEXT: store double [[SP]], ptr addrspace(1) [[Y4]]
// IR-NEXT: ret void

// Without the plugin each call goes through the identity with the call's
// arguments as written.
// PLAIN: define protected amdgpu_kernel void @_Z1kPKdPdS1_S1_i(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}[[DX:%[0-9]+]], ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// PLAIN-NEXT: [[XI:%[0-9]+]] = ptrtoint ptr addrspace(1) [[X]] to i64
// PLAIN-NEXT: [[XP:%[0-9]+]] = inttoptr i64 [[XI]] to ptr
// PLAIN-NEXT: [[DXI:%[0-9]+]] = ptrtoint ptr addrspace(1) [[DX]] to i64
// PLAIN-NEXT: [[DXP:%[0-9]+]] = inttoptr i64 [[DXI]] to ptr
// PLAIN-NEXT: [[YG:%[0-9]+]] = addrspacecast ptr addrspace(1) [[Y]] to ptr
// PLAIN-NEXT: [[DYI:%[0-9]+]] = ptrtoint ptr addrspace(1) [[DY]] to i64
// PLAIN-NEXT: [[DYP:%[0-9]+]] = inttoptr i64 [[DYI]] to ptr
// PLAIN-NEXT: [[F0:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z3sumPKdi)
// PLAIN-NEXT: [[DUP0:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_dup
// PLAIN-NEXT: [[R0:%[0-9]+]] = {{.*}}call {{.*}}double [[F0]](i32 {{.*}}[[DUP0]], ptr {{.*}}[[XP]], ptr {{.*}}[[DXP]], i32 {{.*}}[[N]])
// PLAIN-NEXT: store double [[R0]], ptr addrspace(1) [[Y]]
// PLAIN-NEXT: [[F1:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z5norm2PKd)
// PLAIN-NEXT: [[DUP1:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_dup
// PLAIN-NEXT: [[R1:%[0-9]+]] = {{.*}}call {{.*}}double [[F1]](i32 {{.*}}[[DUP1]], ptr {{.*}}[[XP]], ptr {{.*}}[[DXP]])
// PLAIN-NEXT: [[T1:%[0-9]+]] = fmul {{.*}}double [[R1]], 2.000000e+00
// PLAIN-NEXT: [[Y3:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 24
// PLAIN-NEXT: store double [[T1]], ptr addrspace(1) [[Y3]]
// PLAIN-NEXT: [[F2:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z5scalePKdPdi, double {{.*}}{{f?}}0x3EB0C6F7A0B5ED8D)
// PLAIN-NEXT: [[CONST2:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_const
// PLAIN-NEXT: [[DUP2:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_dup
// PLAIN-NEXT: [[Y1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[YG]], i64 8
// PLAIN-NEXT: {{.*}}call void [[F2]](i32 {{.*}}[[CONST2]], ptr {{.*}}[[XP]], i32 {{.*}}[[DUP2]], ptr {{.*}}[[Y1]], ptr {{.*}}[[DYP]], i32 {{.*}}[[N]])
// PLAIN-NEXT: [[F3:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z3sumPKdi, double {{.*}}1.000000e-09)
// PLAIN-NEXT: [[CONST3:%[0-9]+]] = load i32, ptr addrspace(1) @enzyme_const
// PLAIN-NEXT: [[R3:%[0-9]+]] = {{.*}}call {{.*}}double [[F3]](i32 {{.*}}[[CONST3]], ptr {{.*}}[[XP]], i32 {{.*}}[[N]])
// PLAIN-NEXT: [[Y2:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 16
// PLAIN-NEXT: store double [[R3]], ptr addrspace(1) [[Y2]]
// PLAIN-NEXT: [[R4:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[XP]])
// PLAIN-NEXT: [[Y4:%[0-9]+]] = getelementptr inbounds {{.*}}ptr addrspace(1) [[Y]], i64 32
// PLAIN-NEXT: store double [[R4]], ptr addrspace(1) [[Y4]]
// PLAIN-NEXT: ret void

// DECL: declare {{.*}}ptr @__poseidon_site_explicit(ptr{{[^,)]*}})
// DECL: declare {{.*}}ptr @__poseidon_site_explicit_tau(ptr{{[^,)]*}}, double{{[^,)]*}})

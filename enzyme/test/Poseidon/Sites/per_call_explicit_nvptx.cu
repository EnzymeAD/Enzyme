// REQUIRES: enzyme, nvptx-device-profiler
// DEFINE: %{clang} = %clang -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -o %t/gen.ll
// RUN: FileCheck --check-prefix=GEN --implicit-check-not=__poseidon_site_explicit %s < %t/gen.ll
// RUN: FileCheck --check-prefix=NOMARKER --implicit-check-not=__poseidon_site_explicit %s < %t/gen.ll
// RUN: %{clang} %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-print -o %t/use.ll 2>&1 | FileCheck --check-prefix=USE %s
// RUN: FileCheck --check-prefix=IR --implicit-check-not=__poseidon_site_explicit %s < %t/use.ll
// RUN: %{clang} -o - | tee %t/plain.ll | FileCheck --check-prefix=PLAIN %s
// RUN: FileCheck --check-prefix=DECL --implicit-check-not='define {{.*}}@__poseidon_site_explicit' %s < %t/plain.ll
// RUN: %clang -x cuda --cuda-host-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O1 -S -emit-llvm -I%poseidon_profile_inc %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites %s -o %t/host.ll 2>&1 | FileCheck --allow-empty --check-prefix=QUIET %s
// RUN: FileCheck --check-prefix=HOST --implicit-check-not=__poseidon_site_explicit --implicit-check-not=__poseidon_fp_optimize %s < %t/host.ll

// In device code an explicit list through __poseidon_optimize makes the call
// a site over the function itself, with the shadows the caller gives: here
// sum reaches no constant range of memory through x, which the pass could not
// make the shadow of. poseidon_tau followed by the function's own arguments
// is no explicit list: that call is a per-call site, as the call
// __poseidon_optimize_tau writes. Without the plugin the call goes through an
// identity nothing defines. The host half of a host-device function holds no
// site: there the call runs the function with the primal values, silently.

#include "poseidon/poseidon.h"

#define HD __attribute__((host, device))

struct dim3 {
  unsigned x, y, z;
};
typedef struct CUstream_st *cudaStream_t;
extern "C" int cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                void **args, unsigned long sharedMem,
                                cudaStream_t stream);

#ifdef __CUDA_ARCH__
__attribute__((device)) int enzyme_dup, enzyme_const, poseidon_tau;
#else
int enzyme_dup, enzyme_const, poseidon_tau;
#endif

__attribute__((device)) double sum(const double *x, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += x[i] * x[i];
  return s;
}

__attribute__((device)) void scale(const double *x, double *y, int n) {
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
  return __poseidon_optimize(spread, poseidon_tau, 1e-7, x);
}

__attribute__((global)) void k(const double *x, double *dx, double *y,
                               double *dy, int n) {
  y[0] = __poseidon_optimize(sum, enzyme_dup, x, dx, n);
  y[3] = twice(x, dx);
  __poseidon_optimize_tau(1e-6, scale, enzyme_const, x, enzyme_dup, y + 1, dy,
                          n);
  y[2] = __poseidon_optimize(sum, poseidon_tau, 1e-9, enzyme_const, x, n);
  y[4] = third(x);
}

// Profile generation leaves no call of the identity anywhere in the module.
// NOMARKER: target triple = "nvptx64-nvidia-cuda"

// GEN-DAG: @poseidon_site_preprocess__Z3sumPKdi = private unnamed_addr addrspace(1) constant [22 x i8] c"preprocess__Z3sumPKdi\00"
// GEN-DAG: @poseidon_site_preprocess__Z5scalePKdPdi = private unnamed_addr addrspace(1) constant [26 x i8] c"preprocess__Z5scalePKdPdi\00"
// GEN-DAG: @poseidon_site_preprocess__Z5norm2PKd = private unnamed_addr addrspace(1) constant [23 x i8] c"preprocess__Z5norm2PKd\00"
// GEN-DAG: @poseidon_site_preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0 = private unnamed_addr addrspace(1) constant [47 x i8] c"preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0\00"

// Each site's adjoint reaches the shadow the call gives it: dx for sum and
// norm2, dy for scale, and a zeroed local the pass makes for spread's x. The
// kernel holds norm2's call on both of its paths around the loop of sum.
// GEN: define {{.*}} @_Z5twicePKdPd(ptr {{.*}}[[TX:%[0-9]+]], ptr {{.*}}[[TDX:%[0-9]+]])
// GEN-NEXT: [[TR:%[0-9]+]] = {{.*}}call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr [[TX]], ptr [[TDX]])
// GEN-NEXT: [[TT:%[0-9]+]] = fmul {{.*}}double [[TR]], 2.000000e+00
// GEN-NEXT: ret double [[TT]]
// GEN: define {{.*}} @_Z5thirdPKd(ptr {{.*}}[[HX:%[0-9]+]])
// GEN-NEXT: [[HSH:%[0-9]+]] = alloca [16 x i8]
// GEN-NEXT: call void @llvm.memset.p0.i64(ptr {{.*}}[[HSH]], i8 0, i64 16, i1 false)
// GEN-NEXT: [[HR:%[0-9]+]] = {{.*}}call {{.*}}double @diffepreprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0(ptr [[HX]], ptr [[HSH]])
// GEN-NEXT: ret double [[HR]]
// GEN: define {{.*}}ptx_kernel void @_Z1kPKdPdS1_S1_i(ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}[[DX:%[0-9]+]], ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// GEN-DAG: call void @poseidonProfileStaticCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z3sumPKdi to ptr), ptr addrspacecast (ptr addrspace(1) @fpprofile_static_preprocess__Z3sumPKdi to ptr))
// GEN-DAG: [[DXI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[DX]], i64 {{%[0-9]+}}
// GEN-DAG: atomicrmw fadd ptr [[DXI]], double {{%[0-9]+}} monotonic
// GEN-DAG: call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr [[X]], ptr [[DX]])
// GEN-DAG: call {{.*}}double @diffepreprocess__Z5norm2PKd(ptr [[X]], ptr [[DX]])
// GEN-DAG: call void @poseidonProfileStaticCUDA(ptr addrspacecast (ptr addrspace(1) @poseidon_site_preprocess__Z5scalePKdPdi to ptr), ptr addrspacecast (ptr addrspace(1) @fpprofile_static_preprocess__Z5scalePKdPdi to ptr))
// GEN-DAG: [[DYI:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[DY]], i64 {{%[0-9]+}}
// GEN-DAG: [[DYV:%[0-9]+]] = load double, ptr [[DYI]]
// GEN-DAG: store double 0.000000e+00, ptr [[DYI]]
// GEN-DAG: call {{.*}}void @fixgradient___poseidon_probe_preprocess__Z5scalePKdPdi_0(double {{%[0-9]+}}, double [[DYV]])
// GEN-DAG: call void @llvm.memset.p0.i64(ptr {{.*}}[[SHA:%[0-9]+]], i8 0, i64 16, i1 false)
// GEN-DAG: [[SHA]] = alloca [16 x i8]
// GEN-DAG: call {{.*}}double @diffepreprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0(ptr [[X]], ptr [[SHA]])
// GEN: {{^}}}

// USE: [poseidon] site spread(double const*) at its call in third(double const*): accuracy target 1e-07
// USE-DAG: per_call_explicit_nvptx.cu:[[@LINE-59]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z5norm2PKd.fpprofile:
// USE-DAG: per_call_explicit_nvptx.cu:[[@LINE-52]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0.fpprofile:
// USE-DAG: per_call_explicit_nvptx.cu:[[@LINE-49]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z3sumPKdi.fpprofile:
// USE-DAG: per_call_explicit_nvptx.cu:[[@LINE-50]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z3sumPKdi.fpprofile:
// USE-DAG: per_call_explicit_nvptx.cu:[[@LINE-51]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z5norm2PKd.fpprofile:
// USE-DAG: per_call_explicit_nvptx.cu:[[@LINE-52]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z5scalePKdPdi.fpprofile:
// USE-DAG: per_call_explicit_nvptx.cu:[[@LINE-53]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/Inputs/other_sites/preprocess__Z6spreadPKd.poseidon._Z5thirdPKd.0.fpprofile:
// USE: [poseidon] every site of {{.*}}per_call_explicit_nvptx.cu was left alone; emitting the module as written
// USE-NEXT: 7 warnings generated when compiling for {{.*}}

// Each call left alone calls the function itself with the primal arguments.
// IR: define {{.*}} @_Z5twicePKdPd(ptr {{.*}}[[TX:%[0-9]+]], ptr {{.*}}%{{[0-9]+}})
// IR-NEXT: [[TR:%[0-9]+]] = {{.*}}call double @_Z5norm2PKd(ptr [[TX]])
// IR-NEXT: [[TT:%[0-9]+]] = fmul {{.*}}double [[TR]], 2.000000e+00
// IR-NEXT: ret double [[TT]]
// IR: define {{.*}} @_Z5thirdPKd(ptr {{.*}}[[HX:%[0-9]+]])
// IR-NEXT: [[HR:%[0-9]+]] = {{.*}}call double @_Z6spreadPKd(ptr [[HX]])
// IR-NEXT: ret double [[HR]]
// IR: define {{.*}}ptx_kernel void @_Z1kPKdPdS1_S1_i(ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}%{{[0-9]+}}, ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}%{{[0-9]+}}, i32 {{.*}}[[N:%[0-9]+]])
// IR-NEXT: [[S0:%[0-9]+]] = {{.*}}call double @_Z3sumPKdi(ptr [[X]], i32 [[N]])
// IR-NEXT: store double [[S0]], ptr [[Y]]
// IR-NEXT: [[NR:%[0-9]+]] = {{.*}}call double @_Z5norm2PKd(ptr [[X]])
// IR-NEXT: [[TW:%[0-9]+]] = fmul {{.*}}double [[NR]], 2.000000e+00
// IR-NEXT: [[Y3:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 24
// IR-NEXT: store double [[TW]], ptr [[Y3]]
// IR-NEXT: [[Y1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 8
// IR-NEXT: {{.*}}call void @_Z5scalePKdPdi(ptr [[X]], ptr {{.*}}[[Y1]], i32 [[N]])
// IR-NEXT: [[S1:%[0-9]+]] = {{.*}}call double @_Z3sumPKdi(ptr [[X]], i32 [[N]])
// IR-NEXT: [[Y2:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 16
// IR-NEXT: store double [[S1]], ptr [[Y2]]
// IR-NEXT: [[SP:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr {{.*}}[[X]])
// IR-NEXT: [[Y4:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 32
// IR-NEXT: store double [[SP]], ptr [[Y4]]
// IR-NEXT: ret void

// Without the plugin each call goes through the identity with the call's
// arguments as written.
// PLAIN: define {{.*}} @_Z5twicePKdPd(ptr {{.*}}[[TX:%[0-9]+]], ptr {{.*}}[[TDX:%[0-9]+]])
// PLAIN-NEXT: [[TF:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z5norm2PKd)
// PLAIN-NEXT: [[TDUP:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)
// PLAIN-NEXT: [[TR:%[0-9]+]] = {{.*}}call {{.*}}double [[TF]](i32 {{.*}}[[TDUP]], ptr {{.*}}[[TX]], ptr {{.*}}[[TDX]])
// PLAIN-NEXT: [[TT:%[0-9]+]] = fmul {{.*}}double [[TR]], 2.000000e+00
// PLAIN-NEXT: ret double [[TT]]
// PLAIN: define {{.*}} @_Z5thirdPKd(ptr {{.*}}[[HX:%[0-9]+]])
// PLAIN-NEXT: [[HF:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z6spreadPKd)
// PLAIN-NEXT: [[HTAU:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @poseidon_tau to ptr)
// PLAIN-NEXT: [[HR:%[0-9]+]] = {{.*}}call {{.*}}double [[HF]](i32 {{.*}}[[HTAU]], double {{.*}}{{f?}}0x3E7AD7F29ABCAF48, ptr {{.*}}[[HX]])
// PLAIN-NEXT: ret double [[HR]]
// PLAIN: define {{.*}}ptx_kernel void @_Z1kPKdPdS1_S1_i(ptr {{.*}}[[X:%[0-9]+]], ptr {{.*}}[[DX:%[0-9]+]], ptr {{.*}}[[Y:%[0-9]+]], ptr {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// PLAIN-NEXT: [[F0:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z3sumPKdi)
// PLAIN-NEXT: [[DUP0:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)
// PLAIN-NEXT: [[R0:%[0-9]+]] = {{.*}}call {{.*}}double [[F0]](i32 {{.*}}[[DUP0]], ptr {{.*}}[[X]], ptr {{.*}}[[DX]], i32 {{.*}}[[N]])
// PLAIN-NEXT: store double [[R0]], ptr [[Y]]
// PLAIN-NEXT: [[F1:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z5norm2PKd)
// PLAIN-NEXT: [[DUP1:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)
// PLAIN-NEXT: [[R1:%[0-9]+]] = {{.*}}call {{.*}}double [[F1]](i32 {{.*}}[[DUP1]], ptr {{.*}}[[X]], ptr {{.*}}[[DX]])
// PLAIN-NEXT: [[T1:%[0-9]+]] = fmul {{.*}}double [[R1]], 2.000000e+00
// PLAIN-NEXT: [[Y3:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 24
// PLAIN-NEXT: store double [[T1]], ptr [[Y3]]
// PLAIN-NEXT: [[F2:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit_tau(ptr {{.*}}@_Z5scalePKdPdi, double {{.*}}{{f?}}0x3EB0C6F7A0B5ED8D)
// PLAIN-NEXT: [[CONST2:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr)
// PLAIN-NEXT: [[DUP2:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)
// PLAIN-NEXT: [[Y1:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 8
// PLAIN-NEXT: {{.*}}call void [[F2]](i32 {{.*}}[[CONST2]], ptr {{.*}}[[X]], i32 {{.*}}[[DUP2]], ptr {{.*}}[[Y1]], ptr {{.*}}[[DY]], i32 {{.*}}[[N]])
// PLAIN-NEXT: [[F3:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z3sumPKdi)
// PLAIN-NEXT: [[TAU3:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @poseidon_tau to ptr)
// PLAIN-NEXT: [[CONST3:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr)
// PLAIN-NEXT: [[R3:%[0-9]+]] = {{.*}}call {{.*}}double [[F3]](i32 {{.*}}[[TAU3]], double {{.*}}1.000000e-09, i32 {{.*}}[[CONST3]], ptr {{.*}}[[X]], i32 {{.*}}[[N]])
// PLAIN-NEXT: [[Y2:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 16
// PLAIN-NEXT: store double [[R3]], ptr [[Y2]]
// PLAIN-NEXT: [[F4:%[0-9]+]] = {{.*}}call ptr @__poseidon_site_explicit(ptr {{.*}}@_Z6spreadPKd)
// PLAIN-NEXT: [[TAU4:%[0-9]+]] = load i32, ptr addrspacecast (ptr addrspace(1) @poseidon_tau to ptr)
// PLAIN-NEXT: [[R4:%[0-9]+]] = {{.*}}call {{.*}}double [[F4]](i32 {{.*}}[[TAU4]], double {{.*}}{{f?}}0x3E7AD7F29ABCAF48, ptr {{.*}}[[X]])
// PLAIN-NEXT: [[Y4:%[0-9]+]] = getelementptr inbounds {{.*}}ptr [[Y]], i64 32
// PLAIN-NEXT: store double [[R4]], ptr [[Y4]]
// PLAIN-NEXT: ret void

// DECL: declare {{.*}}ptr @__poseidon_site_explicit(ptr{{[^,)]*}})
// DECL: declare {{.*}}ptr @__poseidon_site_explicit_tau(ptr{{[^,)]*}}, double{{[^,)]*}})

// QUIET-NOT: {{.}}
// HOST-LABEL: define {{.*}} @_Z5twicePKdPd(
// HOST-SAME: ptr {{.*}}[[TX:%[0-9]+]], ptr {{.*}}%{{[0-9]+}})
// HOST-NEXT: [[TR:%[0-9]+]] = {{.*}}call {{.*}}double @_Z5norm2PKd(ptr [[TX]])
// HOST-NEXT: [[TT:%[0-9]+]] = fmul {{.*}}double [[TR]], 2.000000e+00
// HOST-NEXT: ret double [[TT]]
// HOST-LABEL: define {{.*}} @_Z5thirdPKd(
// HOST-SAME: ptr {{.*}}[[HX:%[0-9]+]])
// HOST-NEXT: [[HR:%[0-9]+]] = {{.*}}call {{.*}}double @_Z6spreadPKd(ptr [[HX]])
// HOST-NEXT: ret double [[HR]]

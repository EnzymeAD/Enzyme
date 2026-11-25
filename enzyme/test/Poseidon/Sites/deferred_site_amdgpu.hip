// REQUIRES: enzyme, amdgpu-registered-target, rocm
// DEFINE: %{clang} = %clang -x hip --rocm-path=%rocm_path --cuda-device-only -cuid=0 -O2 \
// DEFINE:   -I%poseidon_profile_inc %loadClangPoseidonEnzyme %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{clang} --offload-arch=gfx942 -S -emit-llvm -mllvm -poseidon-print -o %t/plain.ll 2>&1 | FileCheck --check-prefix=PLAIN %s
// RUN: FileCheck --check-prefix=LEFT --implicit-check-not=jvp_body --implicit-check-not=__enzyme %s < %t/plain.ll
// RUN: %{clang} --offload-arch=%amdgpu_arch -S -emit-llvm -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-compilation-id=ir -mllvm -poseidon-print -o %t/gen.ll 2>&1 \
// RUN:   | FileCheck --check-prefix=GEN %s
// RUN: FileCheck --check-prefix=GENIR %s < %t/gen.ll
// RUN: %{clang} --offload-arch=%amdgpu_arch --no-gpu-bundle-output -c -Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-compilation-id=object -o %t/gen.co 2>&1 | count 0
// RUN: llvm-readelf --dyn-syms %t/gen.co | FileCheck --check-prefix=CO %s

// A site whose marked body is a forward-mode derivative request, in AMDGPU
// device code: the early run defers it, the late run profiles the tangent code
// Enzyme generated, and a site left alone calls the module's own copy of the
// body as the pipeline left it. Its profiled code object links: the derivative
// mallocs its caches, the probes are the module's, and the table is exported.
// (-mllvm would also reach the device linker, which knows no Poseidon flag.)

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

template <typename R, typename... T> __device__ R __enzyme_fwddiff(void *, T...);
template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_dup, poseidon_tau;

extern "C" __device__ __attribute__((noinline)) void qf(const double *x,
                                                        double *y, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += x[i] * x[i];
  y[0] = s + 1.0;
}

extern "C" __device__ void jvp_body(const double *x, const double *dx,
                                    double *y, double *dy, int n) {
  __enzyme_fwddiff<void>((void *)qf, enzyme_dup, x, dx, enzyme_dup, y, dy, n);
}

__global__ void jvp(const double *x, const double *dx, double *y, double *dy,
                    int n) {
  double xl[4] = {x[0], x[1], x[2], x[3]}, dxl[4] = {dx[0], dx[1], dx[2], dx[3]};
  double yl, dyl;
  double x_sh[4] = {}, dx_sh[4] = {}, y_sh = 1.0, dy_sh = 1.0;
  __poseidon_fp_optimize<void>((void *)jvp_body, poseidon_tau, 1e-6,
                               enzyme_dup, xl, x_sh, enzyme_dup, dxl, dx_sh,
                               enzyme_dup, &yl, &y_sh, enzyme_dup, &dyl, &dy_sh,
                               n);
  y[0] = yl;
  dy[0] = dyl;
}

// PLAIN: [poseidon] jvp_body: marked body still holds an __enzyme_* request; deferred to the late run
// PLAIN-NEXT: [poseidon] jvp_body: folded 1 Enzyme-generated call(s) into the deferred body
// PLAIN-NEXT: deferred_site_amdgpu.hip:[[@LINE-15]]:{{[0-9]+}}: warning: Poseidon: a Poseidon site was compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>, so it is left as written; compile with one of the two to profile or optimize it

// The kernel passes its locals to the module's copy of the tangent and
// stores what the copy wrote; the copy sums x[i] * x[i] and 2 * dx[i] * x[i]
// and writes y = sum + 1 and dy, or 1 and 0 when n <= 0.
// LEFT: define protected amdgpu_kernel void @_Z3jvpPKdS0_PdS1_i(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}[[DX:%[0-9]+]], ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// LEFT-DAG: call {{.*}}void @fwddiffeqf.left(ptr {{.*}}[[XP:%[0-9]+]], ptr {{.*}}[[DXP:%[0-9]+]], ptr {{.*}}[[YP:%[0-9]+]], ptr {{.*}}[[DYP:%[0-9]+]], i32 [[N]])
// LEFT-DAG: [[XP]] = addrspacecast ptr addrspace(5) [[XA:%[0-9]+]] to ptr
// LEFT-DAG: [[DXP]] = addrspacecast ptr addrspace(5) [[DXA:%[0-9]+]] to ptr
// LEFT-DAG: [[YP]] = addrspacecast ptr addrspace(5) [[YA:%[0-9]+]] to ptr
// LEFT-DAG: [[DYP]] = addrspacecast ptr addrspace(5) [[DYA:%[0-9]+]] to ptr
// LEFT-DAG: [[XV:%[0-9]+]] = load <2 x double>, ptr addrspace(1) [[X]],
// LEFT-DAG: [[X0:%[0-9]+]] = extractelement <2 x double> [[XV]], i64 0
// LEFT-DAG: store double [[X0]], ptr addrspace(5) [[XA]],
// LEFT-DAG: [[DXV:%[0-9]+]] = load <2 x double>, ptr addrspace(1) [[DX]],
// LEFT-DAG: [[DX0:%[0-9]+]] = extractelement <2 x double> [[DXV]], i64 0
// LEFT-DAG: store double [[DX0]], ptr addrspace(5) [[DXA]],
// LEFT-DAG: [[YV:%[0-9]+]] = load double, ptr addrspace(5) [[YA]]
// LEFT-DAG: store double [[YV]], ptr addrspace(1) [[Y]]
// LEFT-DAG: [[DYV:%[0-9]+]] = load double, ptr addrspace(5) [[DYA]]
// LEFT-DAG: store double [[DYV]], ptr addrspace(1) [[DY]]
// LEFT: {{^}}}
// LEFT: define internal {{.*}}void @fwddiffeqf.left(ptr {{.*}}[[QX:%[0-9]+]], ptr {{.*}}[[QDX:%[0-9]+]], ptr {{.*}}[[QY:%[0-9]+]], ptr {{.*}}[[QDY:%[0-9]+]], i32 {{.*}}%{{[0-9]+}})
// LEFT-DAG: [[DXP:%[0-9]+]] = getelementptr {{.*}}ptr [[QDX]], i64 [[I:%[0-9]+]]{{$}}
// LEFT-DAG: [[XP:%[0-9]+]] = getelementptr {{.*}}ptr [[QX]], i64 [[I]]{{$}}
// LEFT-DAG: [[DXI:%[0-9]+]] = load double, ptr [[DXP]],
// LEFT-DAG: [[XI:%[0-9]+]] = load double, ptr [[XP]],
// LEFT-DAG: [[SQ:%[0-9]+]] = fmul {{.*}}double [[XI]], [[XI]]{{$}}
// LEFT-DAG: [[S:%[0-9]+]] = fadd {{.*}}double [[SACC:%[0-9]+]], [[SQ]]{{$}}
// LEFT-DAG: [[T:%[0-9]+]] = fmul {{.*}}double [[DXI]], 2.000000e+00{{$}}
// LEFT-DAG: [[D:%[0-9]+]] = fmul {{.*}}double [[T]], [[XI]]{{$}}
// LEFT-DAG: [[DS:%[0-9]+]] = fadd {{.*}}double [[D]], [[DACC:%[0-9]+]]{{$}}
// LEFT-DAG: [[SACC]] = phi double [ [[S]], %{{[0-9]+}} ], [ %{{[0-9]+}}, %{{[0-9]+}} ]{{$}}
// LEFT-DAG: [[DACC]] = phi fast double [ [[DS]], %{{[0-9]+}} ], [ %{{[0-9]+}}, %{{[0-9]+}} ]{{$}}
// LEFT-DAG: [[SE:%[0-9]+]] = phi double [ %{{[0-9]+}}, %{{[0-9]+}} ], [ [[S]], %{{[0-9]+}} ]{{$}}
// LEFT-DAG: [[DE:%[0-9]+]] = phi double [ %{{[0-9]+}}, %{{[0-9]+}} ], [ [[DS]], %{{[0-9]+}} ]{{$}}
// LEFT-DAG: [[YE:%[0-9]+]] = fadd {{.*}}double [[SE]], 1.000000e+00{{$}}
// LEFT-DAG: [[DYV:%[0-9]+]] = phi fast double [ 0.000000e+00, %{{[0-9]+}} ], [ [[DE]], %{{[0-9]+}} ]{{$}}
// LEFT-DAG: [[YV:%[0-9]+]] = phi double [ 1.000000e+00, %{{[0-9]+}} ], [ [[YE]], %{{[0-9]+}} ]{{$}}
// LEFT-DAG: store double [[YV]], ptr [[QY]],
// LEFT-DAG: store double [[DYV]], ptr [[QDY]],
// LEFT: {{^}}}

// GEN: [poseidon] jvp_body: marked body still holds an __enzyme_* request; deferred to the late run
// GEN-NEXT: [poseidon] jvp_body: folded 1 Enzyme-generated call(s) into the deferred body
// GEN-NEXT: [poseidon] block-geometry probe placed in 1 profiled function(s)
// GEN-NOT: {{.}}

// The kernel passes each local with its shadow (x's and dx's zeroed, y's and
// dy's seeded 1.0) to the profiled tangent, which logs into the site's table.
// GENIR: @poseidon.site.preprocess_jvp_body = private unnamed_addr addrspace(1) constant [9 x i8] c"jvp_body\00"
// GENIR: define protected amdgpu_kernel void @_Z3jvpPKdS0_PdS1_i(ptr addrspace(1) {{.*}}[[X:%[0-9]+]], ptr addrspace(1) {{.*}}[[DX:%[0-9]+]], ptr addrspace(1) {{.*}}[[Y:%[0-9]+]], ptr addrspace(1) {{.*}}[[DY:%[0-9]+]], i32 {{.*}}[[N:%[0-9]+]])
// GENIR-DAG: call void @diffepreprocess_jvp_body(ptr [[XP:%[0-9]+]], ptr [[XSP:%[0-9]+]], ptr [[DXP:%[0-9]+]], ptr [[DXSP:%[0-9]+]], ptr [[YP:%[0-9]+]], ptr [[YSP:%[0-9]+]], ptr [[DYP:%[0-9]+]], ptr [[DYSP:%[0-9]+]], i32 [[N]])
// GENIR-DAG: [[XP]] = addrspacecast ptr addrspace(5) [[XA:%[0-9]+]] to ptr
// GENIR-DAG: [[XSP]] = addrspacecast ptr addrspace(5) [[XSA:%[0-9]+]] to ptr
// GENIR-DAG: [[DXSP]] = addrspacecast ptr addrspace(5) [[DXSA:%[0-9]+]] to ptr
// GENIR-DAG: [[YP]] = addrspacecast ptr addrspace(5) [[YA:%[0-9]+]] to ptr
// GENIR-DAG: [[YSP]] = addrspacecast ptr addrspace(5) [[YSA:%[0-9]+]] to ptr
// GENIR-DAG: [[DYP]] = addrspacecast ptr addrspace(5) [[DYA:%[0-9]+]] to ptr
// GENIR-DAG: [[DYSP]] = addrspacecast ptr addrspace(5) [[DYSA:%[0-9]+]] to ptr
// GENIR-DAG: load {{.*}}, ptr addrspace(1) [[X]]
// GENIR-DAG: store double {{%[0-9]+}}, ptr addrspace(5) [[XA]]
// GENIR-DAG: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}[[XSA]], i8 0, i64 32, i1 false)
// GENIR-DAG: call void @llvm.memset.p5.i64(ptr addrspace(5) {{.*}}[[DXSA]], i8 0, i64 32, i1 false)
// GENIR-DAG: store double 1.000000e+00, ptr addrspace(5) [[YSA]]
// GENIR-DAG: store double 1.000000e+00, ptr addrspace(5) [[DYSA]]
// GENIR-DAG: [[YV:%[0-9]+]] = load double, ptr addrspace(5) [[YA]]
// GENIR-DAG: store double [[YV]], ptr addrspace(1) [[Y]]
// GENIR-DAG: [[DYV:%[0-9]+]] = load double, ptr addrspace(5) [[DYA]]
// GENIR-DAG: store double [[DYV]], ptr addrspace(1) [[DY]]
// GENIR: {{^}}}
// GENIR: define internal void @diffepreprocess_jvp_body(
// GENIR: call void @__poseidon_device_log_block_dims(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess_jvp_body to ptr))
// GENIR-NEXT: call void @__poseidon_device_publish_static(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess_jvp_body to ptr), ptr addrspacecast (ptr addrspace(1) @poseidon.static.preprocess_jvp_body to ptr))
// GENIR: call {{.*}}ptr @malloc(i64 {{%[0-9]+}}), !enzyme_cache_alloc
// GENIR: [[V:%[0-9]+]] = call contract double @llvm.fmuladd.f64(double [[XI:%[0-9]+]], double [[XI]], double {{%[0-9]+}}){{.*}}, !poseidon.slot
// GENIR-NEXT: [[P:%[0-9]+]] = call {{.*}}double @__poseidon_probe_preprocess_jvp_body_0_aug(double [[V]])
// GENIR: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess_jvp_body to ptr), i64 0, double [[P]], i64 3, ptr {{%[0-9]+}})
// GENIR: {{^}}}
// GENIR-DAG: define internal void @__poseidon_device_log_value(
// GENIR-DAG: define internal void @__poseidon_device_log_grad(

// CO-NOT: UND {{[_a-zA-Z]}}
// CO: OBJECT WEAK PROTECTED {{[0-9]+}} __poseidon_profile_table{{$}}
// CO-NOT: UND {{[_a-zA-Z]}}
// CO: FUNC GLOBAL PROTECTED {{[0-9]+}} _Z3jvpPKdS0_PdS1_i{{$}}
// CO-NOT: UND {{[_a-zA-Z]}}

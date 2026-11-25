// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t/drv %t/drv-rdc
// RUN: %poseidon_clangxx -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -O2 -c -poseidon-profile-generate -poseidon-cache=%t/drv/c %s -o %t/drv/k.o
// RUN: cd %t/drv && llvm-objdump --offloading k.o > /dev/null && llvm-readelf --dyn-syms k.o.*%amdgpu_arch | FileCheck --check-prefix=CO --implicit-check-not=__enzyme %s
// RUN: %poseidon_clangxx -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -fgpu-rdc -O2 -c -poseidon-profile-generate -poseidon-cache=%t/drv-rdc/c %s -o %t/drv-rdc/k.o
// RUN: llvm-offload-binary %t/drv-rdc/k.o --image=file=%t/drv-rdc/dev.bc,arch=%amdgpu_arch
// RUN: llvm-dis %t/drv-rdc/dev.bc -o - | FileCheck --check-prefixes=DEV,DRVDEV --implicit-check-not=__enzyme %s
//
// RUN: %cmake %cmake_generator -S %S/Inputs/package_hip -B %t/pkg-gen -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DSOURCE=%s -DAMDGPU_ARCH=%amdgpu_arch -DROCM_PATH=%rocm_path -DACTION=ProfileGenerate -DPOSEIDON_FLAGS=-poseidon-cache=%t/pkg-gen/c > /dev/null
// RUN: %cmake --build %t/pkg-gen -v 2>&1 | FileCheck --check-prefix=PKGGEN -DARCH=%amdgpu_arch %s
// RUN: cd %t/pkg-gen && cp CMakeFiles/k.dir/k.cpp.o k.o && llvm-objdump --offloading k.o > /dev/null && llvm-readelf --dyn-syms k.o.*%amdgpu_arch | FileCheck --check-prefix=CO --implicit-check-not=__enzyme %s
// RUN: llvm-offload-binary %t/pkg-gen/CMakeFiles/k_rdc.dir/k.cpp.o --image=file=%t/pkg-gen/dev.bc,arch=%amdgpu_arch
// RUN: llvm-dis %t/pkg-gen/dev.bc -o - | FileCheck --check-prefixes=DEV,PKGDEV --implicit-check-not=__enzyme %s
//
// RUN: %cmake %cmake_generator -S %S/Inputs/package_hip -B %t/pkg-plain -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DSOURCE=%s -DAMDGPU_ARCH=%amdgpu_arch -DROCM_PATH=%rocm_path -DACTION=Plain -DLOAD_ENZYME=ON -DENZYME_OPTIONS=ON -DDEFINES=DEFERRED > /dev/null
// RUN: %cmake --build %t/pkg-plain -v 2>&1 | FileCheck --check-prefix=PKGENZYME -DARCH=%amdgpu_arch %s
// RUN: %cmake %cmake_generator -S %S/Inputs/package_hip -B %t/pkg-use -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DSOURCE=%s -DAMDGPU_ARCH=%amdgpu_arch -DROCM_PATH=%rocm_path -DACTION=ProfileUse -DLOAD_ENZYME=ON -DENZYME_OPTIONS=ON -DDEFINES=DEFERRED "-DPOSEIDON_FLAGS=-poseidon-profile-use=%S/../../Inputs/unrelated_site_profile -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -poseidon-enable-rewrite=0 -poseidon-cache=%t/pkg-use/c" > /dev/null
// RUN: %cmake --build %t/pkg-use -v 2>&1 | FileCheck --check-prefix=PKGENZYME -DARCH=%amdgpu_arch %s
// RUN: %cmake %cmake_generator -S %S/Inputs/package_hip -B %t/pkg-noopt -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DSOURCE=%s -DAMDGPU_ARCH=%amdgpu_arch -DROCM_PATH=%rocm_path -DACTION=Plain -DLOAD_ENZYME=ON -DDEFINES=DEFERRED > /dev/null
// RUN: env TMPDIR=%t not %cmake --build %t/pkg-noopt 2>&1 | FileCheck --check-prefix=DEFERRED-PKG --implicit-check-not=REPRODUCER --implicit-check-not=ld.lld %s

// Full HIP compiles (-c, neither -S nor --offload-device-only), whose device
// half is an LTO pre-link compile. The device link loads no plugin, so Enzyme
// must lower derivative requests in that compile (-enzyme-lto-prelink).
//
// Profile generation, through poseidon-clang++ and through the package's
// Poseidon::ProfileGenerate: the compile succeeds, the code object resolves
// every symbol, and the device bitcode holds no derivative request and calls
// the profiling hooks.
//
// A site whose body is the user's own derivative request (DEFERRED), in a
// project that loads Enzyme itself under the package's Plain or ProfileUse:
// it compiles once the targets that load Enzyme link Poseidon::EnzymeOptions,
// and without it the build stops in the compile with the reason, not at the
// device link (deferred_site_lto_amdgpu.hip has the same without ROCm, for
// clang and poseidon-clang++).

#include <hip/hip_runtime.h>

#include "poseidon/poseidon.h"

#ifdef DEFERRED
template <typename R, typename... T> __device__ R __enzyme_fwddiff(void *, T...);
template <typename R, typename... T>
__device__ R __poseidon_fp_optimize(void *, T...);
__device__ int enzyme_dup, poseidon_tau;

extern "C" __device__ __attribute__((noinline)) void qf(const double *x,
                                                        double *y) {
  y[0] = x[0] * x[0] + 1.0;
}

extern "C" __device__ void jvp_body(const double *x, const double *dx,
                                    double *y, double *dy) {
  __enzyme_fwddiff<void>((void *)qf, enzyme_dup, x, dx, enzyme_dup, y, dy);
}

__global__ void jvp(const double *x, const double *dx, double *y, double *dy) {
  double xl = x[0], dxl = dx[0], yl, dyl;
  double x_sh = 0.0, dx_sh = 0.0, y_sh = 1.0, dy_sh = 1.0;
  __poseidon_fp_optimize<void>((void *)jvp_body, poseidon_tau, 1e-6,
                               enzyme_dup, &xl, &x_sh, enzyme_dup, &dxl,
                               &dx_sh, enzyme_dup, &yl, &y_sh, enzyme_dup, &dyl,
                               &dy_sh);
  y[0] = yl;
  dy[0] = dyl;
}
#else
POSEIDON_OPTIMIZE __global__ void site(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double a = x[i];
    y[i] = (3.0 * a + 0.5) * a + 1.0;
  }
}

void launch_site(const double *x, double *y, int n) {
  site<<<(n + 127) / 128, 128>>>(x, y, n);
}
#endif

// CO-NOT: UND {{[_a-zA-Z]}}
// CO: FUNC GLOBAL PROTECTED {{[0-9]+}} _Z4sitePKdPdi{{$}}
// CO-NOT: UND {{[_a-zA-Z]}}
// CO: OBJECT WEAK PROTECTED {{[0-9]+}} __poseidon_profile_table{{$}}
// CO-NOT: UND {{[_a-zA-Z]}}

// The kernel probes both operations of the site, each with the operands it
// takes and its gradient read from the output's shadow, and the profiler is
// linked into the module. With -O2 the derivative is inlined into the kernel;
// the package's compile, at -O0, calls it with the kernel's shadows.
// DEV: @[[SITE:poseidon.site.preprocess__Z4sitePKdPdi_poseidon_body]] = private unnamed_addr addrspace(1) constant [14 x i8] c"_Z4sitePKdPdi\00"
// DEV: @__poseidon_profile_table = weak_odr protected addrspace(4) externally_initialized constant %struct.DeviceProfileTable zeroinitializer,
// DEV: define amdgpu_kernel void @_Z4sitePKdPdi(ptr addrspace(1) {{[^%]*}}%[[X:[0-9]+]], ptr addrspace(1) {{[^%]*}}%[[Y:[0-9]+]], i32 %[[N:[0-9]+]], ptr {{[^%]*}}%[[DX:[0-9]+]], ptr {{[^%]*}}%[[DY:[0-9]+]])
// DRVDEV: %[[DYG:[0-9]+]] = addrspacecast ptr %[[DY]] to ptr addrspace(1)
// DRVDEV: %[[DXG:[0-9]+]] = addrspacecast ptr %[[DX]] to ptr addrspace(1)
// PKGDEV: %[[DXG:[0-9]+]] = addrspacecast ptr %[[DX]] to ptr addrspace(1)
// PKGDEV: %[[DYG:[0-9]+]] = addrspacecast ptr %[[DY]] to ptr addrspace(1)
// PKGDEV: call fastcc void @[[DIFFE:diffepreprocess__Z4sitePKdPdi_poseidon_body]](ptr addrspace(1) %[[X]], ptr addrspace(1) %[[DXG]], ptr addrspace(1) %[[Y]], ptr addrspace(1) %[[DYG]], i32 %[[N]])
// PKGDEV: define internal fastcc void @[[DIFFE]](ptr addrspace(1) {{[^%]*}}%{{[0-9]+}}, ptr addrspace(1) {{[^%]*}}%{{[0-9]+}}, ptr addrspace(1) {{[^%]*}}%{{[0-9]+}}, ptr addrspace(1) {{[^%]*}}%[[PDY:[0-9]+]], i32
// PKGDEV: %[[PDYI:[0-9]+]] = ptrtoint ptr addrspace(1) %[[PDY]] to i64
// PKGDEV: %[[PDYP:[0-9]+]] = inttoptr i64 %[[PDYI]] to ptr
// DEV: %[[A:[0-9]+]] = load double, ptr {{(addrspace\(1\) )?}}%{{[0-9]+}}, align 8
// DEV: %[[T:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[A]], double 3.000000e+00, double 5.000000e-01)
// DEV: %[[TP:[0-9]+]] = {{(tail )?}}call double @__poseidon_device_perturb(i32 0, double %[[T]])
// DEV: store double %[[A]], ptr addrspace(5) %[[OPS0:[0-9]+]], align 8
// DEV: %[[OPS0G:[0-9]+]] = addrspacecast ptr addrspace(5) %[[OPS0]] to ptr
// DEV: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @[[SITE]] to ptr), i64 0, double %[[TP]], i64 3, ptr %[[OPS0G]])
// DEV: %[[R:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[TP]], double %[[A]], double 1.000000e+00)
// DEV: %[[RP:[0-9]+]] = {{(tail )?}}call double @__poseidon_device_perturb(i32 0, double %[[R]])
// DEV: store double %[[TP]], ptr addrspace(5) %[[OPS1:[0-9]+]], align 8
// DEV: %[[OPS1G:[0-9]+]] = addrspacecast ptr addrspace(5) %[[OPS1]] to ptr
// DEV: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @[[SITE]] to ptr), i64 1, double %[[RP]], i64 3, ptr %[[OPS1G]])
// DEV: store double %[[RP]], ptr {{(addrspace\(1\) )?}}%{{[0-9]+}}, align 8
// DRVDEV: %[[DYI:[0-9]+]] = getelementptr inbounds [8 x i8], ptr addrspace(1) %[[DYG]], i64
// PKGDEV: %[[DYI:[0-9]+]] = getelementptr inbounds [8 x i8], ptr %[[PDYP]], i64
// DEV: %[[G:[0-9]+]] = load double, ptr {{(addrspace\(1\) )?}}%[[DYI]], align 8
// DEV: store double 0.000000e+00, ptr {{(addrspace\(1\) )?}}%[[DYI]], align 8
// DEV: %[[R2:[0-9]+]] = {{(tail )?}}call double @llvm.fmuladd.f64(double %[[T]], double %[[A]], double 1.000000e+00)
// DEV: call void @__poseidon_device_log_grad(ptr addrspacecast (ptr addrspace(1) @[[SITE]] to ptr), i64 1, double %[[R2]], double %[[G]])
// DEV: %[[GT:[0-9]+]] = fmul fast double %[[G]], %[[A]]
// DEV: call void @__poseidon_device_log_grad(ptr addrspacecast (ptr addrspace(1) @[[SITE]] to ptr), i64 0, double %[[T]], double %[[GT]])
// DEV: define internal void @__poseidon_device_log_value(
// DEV: define internal void @__poseidon_device_log_grad(

// Each object compiles with Poseidon's plugin, then Enzyme's, generating a
// profile with Enzyme lowering derivative requests in the pre-link compile.
// PKGGEN-DAG: {{ }}-x hip --offload-arch=[[ARCH]] --rocm-path={{[^ ]+}} -fpass-plugin={{[^ ]+}}/Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-fpass-plugin={{[^ ]+}}/ClangEnzyme-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -enzyme-lto-prelink=1 -ffp-contract=on -MD -MT CMakeFiles/k.dir/k.cpp.o {{.*}} -c {{[^ ]+}}/k.cpp{{$}}
// PKGGEN-DAG: {{ }}-fgpu-rdc -x hip --offload-arch=[[ARCH]] --rocm-path={{[^ ]+}} -fpass-plugin={{[^ ]+}}/Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-fpass-plugin={{[^ ]+}}/ClangEnzyme-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-Xclang -mllvm -Xclang -poseidon-profile-generate -Xclang -mllvm -Xclang -enzyme-lto-prelink=1 -ffp-contract=on -MD -MT CMakeFiles/k_rdc.dir/k.cpp.o {{.*}} -c {{[^ ]+}}/k.cpp{{$}}

// The project's own Enzyme plugin comes after Poseidon's, and
// Poseidon::EnzymeOptions adds the pre-link lowering to the targets that load
// it.
// PKGENZYME-DAG: {{ }}-x hip --offload-arch=[[ARCH]] --rocm-path={{[^ ]+}} -fpass-plugin={{[^ ]+}}/Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-ffp-contract=on -fpass-plugin={{[^ ]+}}/ClangEnzyme-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-Xclang -mllvm -Xclang -enzyme-lto-prelink=1 -MD -MT CMakeFiles/k.dir/k.cpp.o {{.*}} -c {{[^ ]+}}/k.cpp{{$}}
// PKGENZYME-DAG: {{ }}-fgpu-rdc -x hip --offload-arch=[[ARCH]] --rocm-path={{[^ ]+}} -fpass-plugin={{[^ ]+}}/Poseidon-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-ffp-contract=on -fpass-plugin={{[^ ]+}}/ClangEnzyme-{{[0-9]+}}.so {{(-Xclang -load -Xclang [^ ]+ )?}}-Xclang -mllvm -Xclang -enzyme-lto-prelink=1 -MD -MT CMakeFiles/k_rdc.dir/k.cpp.o {{.*}} -c {{[^ ]+}}/k.cpp{{$}}

// DEFERRED-PKG: FAILED: {{(\[code=[0-9]+\] )?}}CMakeFiles/{{k|k_rdc}}.dir/k.cpp.o
// DEFERRED-PKG: k.cpp:[[@LINE-85]]:28: error: Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered. This is an LTO pre-link compile, where Enzyme leaves derivative requests to the post-link run unless -enzyme-lto-prelink is given; pass -mllvm -enzyme-lto-prelink=1 (for HIP as -Xclang -mllvm -Xclang -enzyme-lto-prelink=1).

// REQUIRES: enzyme, amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t/drv %t/drv-rdc
// RUN: %poseidon_clangxx -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -O2 -c -poseidon-profile-generate -poseidon-cache=%t/drv/c %s -o %t/drv/k.o
// RUN: cd %t/drv && llvm-objdump --offloading k.o > /dev/null && llvm-readelf --dyn-syms k.o.*%amdgpu_arch | FileCheck --check-prefix=CO --implicit-check-not=__enzyme %s
// RUN: %poseidon_clangxx -x hip --offload-arch=%amdgpu_arch --rocm-path=%rocm_path -fgpu-rdc -O2 -c -poseidon-profile-generate -poseidon-cache=%t/drv-rdc/c %s -o %t/drv-rdc/k.o
// RUN: llvm-offload-binary %t/drv-rdc/k.o --image=file=%t/drv-rdc/dev.bc,arch=%amdgpu_arch
// RUN: llvm-dis %t/drv-rdc/dev.bc -o - | FileCheck --check-prefix=DEV --implicit-check-not=__enzyme %s
//
// RUN: %cmake %cmake_generator -S %S/../Inputs/package_hip -B %t/pkg-gen -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DSOURCE=%s -DAMDGPU_ARCH=%amdgpu_arch -DROCM_PATH=%rocm_path -DACTION=ProfileGenerate -DPOSEIDON_FLAGS=-poseidon-cache=%t/pkg-gen/c > /dev/null
// RUN: %cmake --build %t/pkg-gen -v > %t/pkg-gen.log 2>&1
// RUN: FileCheck --check-prefix=PKGGEN %s < %t/pkg-gen.log
// RUN: cd %t/pkg-gen && cp CMakeFiles/k.dir/k.cpp.o k.o && llvm-objdump --offloading k.o > /dev/null && llvm-readelf --dyn-syms k.o.*%amdgpu_arch | FileCheck --check-prefix=CO --implicit-check-not=__enzyme %s
// RUN: llvm-offload-binary %t/pkg-gen/CMakeFiles/k_rdc.dir/k.cpp.o --image=file=%t/pkg-gen/dev.bc,arch=%amdgpu_arch
// RUN: llvm-dis %t/pkg-gen/dev.bc -o - | FileCheck --check-prefix=DEV --implicit-check-not=__enzyme %s
//
// RUN: %cmake %cmake_generator -S %S/../Inputs/package_hip -B %t/pkg-plain -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DSOURCE=%s -DAMDGPU_ARCH=%amdgpu_arch -DROCM_PATH=%rocm_path -DACTION=Plain -DLOAD_ENZYME=ON -DENZYME_OPTIONS=ON -DDEFINES=DEFERRED > /dev/null
// RUN: %cmake --build %t/pkg-plain -v > %t/pkg-plain.log 2>&1
// RUN: FileCheck --check-prefix=PKGENZYME %s < %t/pkg-plain.log
// RUN: %cmake %cmake_generator -S %S/../Inputs/package_hip -B %t/pkg-use -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DSOURCE=%s -DAMDGPU_ARCH=%amdgpu_arch -DROCM_PATH=%rocm_path -DACTION=ProfileUse -DLOAD_ENZYME=ON -DENZYME_OPTIONS=ON -DDEFINES=DEFERRED "-DPOSEIDON_FLAGS=-poseidon-profile-use=%S/../Inputs/other_sites -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/pkg-use/c" > /dev/null
// RUN: %cmake --build %t/pkg-use -v > %t/pkg-use.log 2>&1
// RUN: FileCheck --check-prefix=PKGENZYME %s < %t/pkg-use.log
// RUN: %cmake %cmake_generator -S %S/../Inputs/package_hip -B %t/pkg-noopt -DCMAKE_CXX_COMPILER=%clang++ -DPoseidon_DIR=%poseidon_package_dir -DSOURCE=%s -DAMDGPU_ARCH=%amdgpu_arch -DROCM_PATH=%rocm_path -DACTION=Plain -DLOAD_ENZYME=ON -DDEFINES=DEFERRED > /dev/null
// RUN: env TMPDIR=%t not %cmake --build %t/pkg-noopt > %t/pkg-noopt.log 2>&1
// RUN: FileCheck --check-prefix=DEFERRED-PKG --implicit-check-not=REPRODUCER --implicit-check-not=ld.lld %s < %t/pkg-noopt.log

// Full HIP compiles, -c without -S or --offload-device-only: the device half is
// an LTO pre-link compile whose bitcode the same -c links into a code object,
// or with relocatable device code embeds as it is. Enzyme leaves derivative
// requests to a post-link run there unless -enzyme-lto-prelink is given, and
// the device link loads no plugin.
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
// CO: __poseidon_profile_table
// CO-NOT: UND {{[_a-zA-Z]}}

// DEV-DAG: define internal void @poseidonLogValueCUDA(
// DEV-DAG: define internal void @poseidonLogGradCUDA(
// DEV-DAG: call {{.*}}@poseidonLogValueCUDA(
// DEV-DAG: call {{.*}}@poseidonLogGradCUDA(

// PKGGEN: -Xclang -mllvm -Xclang -poseidon-profile-generate
// PKGGEN-SAME: -Xclang -mllvm -Xclang -enzyme-lto-prelink=1

// PKGENZYME: -fpass-plugin={{[^ ]*}}ClangEnzyme
// PKGENZYME-SAME: -Xclang -mllvm -Xclang -enzyme-lto-prelink=1

// DEFERRED-PKG: Poseidon: the marked body jvp_body still holds an __enzyme_* request that nothing lowered. This is an LTO pre-link compile

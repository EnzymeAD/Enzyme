// REQUIRES: amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t/fpc %t/use %t/refused %t/fpc.thin %t/use.thin %t/quiet
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -c -DNO_SITE %s -o %t/fpc/k.o
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/rewritten_lto_hip -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-print %s -o %t/use/k.o 2>&1 | FileCheck --check-prefix=SOLVE %s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -c -DONE_SITE -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/rewritten_lto_hip -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 -Xclang -mllvm -Xclang -poseidon-cache=%t/c.quiet %s -o %t/quiet/k.o 2>&1 | count 0
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -c -DONE_SITE -DSITE_TAU=1e-18 -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/rewritten_lto_hip -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 -Xclang -mllvm -Xclang -poseidon-cache=%t/c.quiet.refused %s -o %t/quiet/refused.o 2>&1 | count 0
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -foffload-lto=thin -c -DNO_SITE %s -o %t/fpc.thin/k.o
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -foffload-lto=thin -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/rewritten_lto_hip -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 -Xclang -mllvm -Xclang -poseidon-cache=%t/c.thin -Xclang -mllvm -Xclang -poseidon-print %s -o %t/use.thin/k.o 2>&1 | FileCheck --check-prefix=SOLVE %s
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -c -DSITE_TAU=1e-18 -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/rewritten_lto_hip -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-rewrite=0 -Xclang -mllvm -Xclang -poseidon-cache=%t/c.refused -Xclang -mllvm -Xclang -poseidon-print %s -o %t/refused/k.o 2>&1 | FileCheck --check-prefix=REFUSED %s
// RUN: cd %t/fpc && llvm-objdump --offloading k.o && llvm-objdump -d --no-leading-addr k.o.*gfx942 | sed -E 's#// [0-9A-F]+: #// #' > %t/fpc.isa
// RUN: cd %t/use && llvm-objdump --offloading k.o && llvm-objdump -d --no-leading-addr k.o.*gfx942 | sed -E 's#// [0-9A-F]+: #// #' > %t/use.isa
// RUN: cd %t/refused && llvm-objdump --offloading k.o && llvm-objdump -d --no-leading-addr k.o.*gfx942 | sed -E 's#// [0-9A-F]+: #// #' > %t/refused.isa
// RUN: diff %t/fpc.isa %t/refused.isa
// RUN: cd %t/fpc.thin && llvm-objdump --offloading k.o && llvm-objdump -d --no-leading-addr k.o.*gfx942 | sed -E 's#// [0-9A-F]+: #// #' > %t/fpc.thin.isa
// RUN: cd %t/use.thin && llvm-objdump --offloading k.o && llvm-objdump -d --no-leading-addr k.o.*gfx942 | sed -E 's#// [0-9A-F]+: #// #' > %t/use.thin.isa
// RUN: awk '/^<.*>:$/{print "NAME " $0; p = $0 ~ /^<_Z(5other|10unprofiled)PKdPdi>:$/} p' %t/fpc.isa > %t/fpc.fns
// RUN: awk '/^<.*>:$/{print "NAME " $0; p = $0 ~ /^<_Z(5other|10unprofiled)PKdPdi>:$/} p' %t/use.isa > %t/use.fns
// RUN: diff %t/fpc.fns %t/use.fns
// RUN: FileCheck --check-prefix=FNS %s < %t/use.fns
// RUN: awk '/^<.*>:$/{print "NAME " $0; p = $0 ~ /^<_Z(5other|10unprofiled)PKdPdi>:$/} p' %t/fpc.thin.isa > %t/fpc.thin.fns
// RUN: awk '/^<.*>:$/{print "NAME " $0; p = $0 ~ /^<_Z(5other|10unprofiled)PKdPdi>:$/} p' %t/use.thin.isa > %t/use.thin.fns
// RUN: diff %t/fpc.thin.fns %t/use.thin.fns
// RUN: awk '/^<.*>:$/{p = $0 ~ /^<_Z4sitePKdPdi>:$/} p' %t/fpc.isa | FileCheck --check-prefix=FPC-SITE %s
// RUN: awk '/^<.*>:$/{p = $0 ~ /^<_Z4sitePKdPdi>:$/} p' %t/use.isa | FileCheck --check-prefix=USE-SITE --implicit-check-not=v_fma_f64 %s
// RUN: awk '/^<.*>:$/{p = $0 ~ /^<_Z4sitePKdPdi>:$/} p' %t/use.thin.isa | FileCheck --check-prefix=USE-SITE --implicit-check-not=v_fma_f64 %s

// Under full and thin offload LTO, every function no rewrite reaches (a kernel
// that is no site, a site the profile does not cover) is in the code object as
// the unannotated source compiled without the plugin has it, in the same order;
// only the rewritten site differs. Under a tolerance no rewrite meets, the whole
// code object is the one as written. The profile of `site` is the one its CUDA
// twin Integration/rewritten_module_as_written.cu records; no gfx942 cost model
// is measured, so the compile prices from a synthetic one.
// Without the unprofiled site the compile has nothing to warn about, and
// both halves print nothing, whether the site is rewritten or no rewrite
// meets its tolerance.

#include <hip/hip_runtime.h>

#ifdef NO_SITE
#define POSEIDON_OPTIMIZE
#define POSEIDON_OPTIMIZE_TAU(tau)
#else
#include "poseidon/poseidon.h"
#endif
#ifndef SITE_TAU
#define SITE_TAU 1e-6
#endif

POSEIDON_OPTIMIZE_TAU(SITE_TAU)
__global__ void site(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double a = x[i];
    y[i] = (3.0 * a + 0.5) * a + 1.0;
  }
}

__global__ void other(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sqrt(x[i]) + y[i];
}

#ifndef ONE_SITE
POSEIDON_OPTIMIZE __global__ void unprofiled(const double *x, double *y,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sqrt(x[i]) * y[i];
}
#endif

// SOLVE: [poseidon] site site(double const*, double*, int): accuracy target 1e-06
// SOLVE: [poseidon] pin: _Z4sitePKdPdi:P0=All FP64(0%) + FP32(100%)
// SOLVE: rewritten_module_lto_amdgpu.hip:[[@LINE-10]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z10unprofiledPKdPdi.fpprofile:
// SOLVE: [poseidon] {{.*}}rewritten_module_lto_amdgpu.hip: 1 function(s) a rewrite reached; every other function as written

// REFUSED: [poseidon] site site(double const*, double*, int): accuracy target 1e-18
// REFUSED: [poseidon] pin: {{$}}
// REFUSED: rewritten_module_lto_amdgpu.hip:[[@LINE-15]]:{{[0-9]+}}: warning: Poseidon: no profile at {{.*}}/_Z10unprofiledPKdPdi.fpprofile:
// REFUSED: [poseidon] every site of {{.*}}rewritten_module_lto_amdgpu.hip was left alone; emitting the module as written

// FNS: NAME <_Z4sitePKdPdi>:
// FNS-NEXT: NAME <_Z5otherPKdPdi>:
// FNS-NOT: s_swappc
// FNS: v_rsq_f64
// FNS-NOT: s_swappc
// FNS: NAME <_Z10unprofiledPKdPdi>:
// FNS-NOT: s_swappc
// FNS: v_rsq_f64
// FNS-NOT: s_swappc

// As written, y = (3 x + 0.5) x + 1 in two FP64 fmas; rewritten, the same two
// fmas in FP32 between a narrowing of x and a widening of the result.
// FPC-SITE: global_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, off
// FPC-SITE: v_fma_f64 [[T:v\[[0-9]+:[0-9]+\]]], [[X]], s{{\[[0-9]+:[0-9]+\]}}, 0.5
// FPC-SITE-NEXT: v_fma_f64 [[R:v\[[0-9]+:[0-9]+\]]], [[T]], [[X]], 1.0
// FPC-SITE-NEXT: global_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[R]], off

// USE-SITE: global_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, off
// USE-SITE-DAG: v_cvt_f32_f64_e32 [[XF:v[0-9]+]], [[X]]
// USE-SITE-DAG: v_mov_b32_e32 [[H:v[0-9]+]], 0.5
// USE-SITE: v_fmamk_f32 [[T:v[0-9]+]], [[XF]], 0x40400000, [[H]]
// USE-SITE-NEXT: v_fma_f32 [[R:v[0-9]+]], [[T]], [[XF]], 1.0
// USE-SITE-NEXT: v_cvt_f64_f32_e32 [[RD:v\[[0-9]+:[0-9]+\]]], [[R]]
// USE-SITE-NEXT: global_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RD]], off

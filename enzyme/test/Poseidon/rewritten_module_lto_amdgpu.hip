// REQUIRES: amdgpu-registered-target, rocm
// RUN: rm -rf %t && mkdir -p %t/fpc %t/use %t/refused %t/fpc.thin %t/use.thin
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -c -DNO_SITE %s -o %t/fpc/k.o
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/rewritten_lto_hip -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-herbie=0 -Xclang -mllvm -Xclang -poseidon-cache=%t/c -Xclang -mllvm -Xclang -poseidon-print %s -o %t/use/k.o 2> %t/use.err
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -foffload-lto=thin -c -DNO_SITE %s -o %t/fpc.thin/k.o
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -foffload-lto=thin -c -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/rewritten_lto_hip -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-herbie=0 -Xclang -mllvm -Xclang -poseidon-cache=%t/c.thin -Xclang -mllvm -Xclang -poseidon-print %s -o %t/use.thin/k.o 2> %t/use.thin.err
// RUN: %clang -x hip --offload-arch=gfx942 --rocm-path=%rocm_path -cuid=0 -O2 -ffp-contract=on -c -DSITE_TAU=1e-18 -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%S/Inputs/rewritten_lto_hip -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx942_synthetic.csv -Xclang -mllvm -Xclang -poseidon-enable-herbie=0 -Xclang -mllvm -Xclang -poseidon-cache=%t/c.refused -Xclang -mllvm -Xclang -poseidon-print %s -o %t/refused/k.o 2> %t/refused.err
// RUN: FileCheck --check-prefix=SOLVE %s < %t/use.err
// RUN: FileCheck --check-prefix=REFUSED %s < %t/refused.err
// RUN: FileCheck --check-prefix=SOLVE %s < %t/use.thin.err
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

// HIP compiles device code through LTO: each -c writes device bitcode, from
// which the device link generates the code object. There too every function
// no rewrite reached is the one as written: in the code object of a module
// whose site is rewritten, a kernel that is no site and a site the profile
// does not cover are those of the unannotated source compiled without the
// plugin, in the same order, and only the rewritten site differs, under full
// and thin LTO; under a target no rewrite meets, the whole code object is the
// one as written. The profile of `site` is the one its CUDA twin
// Integration/rewritten_module_as_written.cu records on an RTX 5090; no gfx942
// cost model is measured, so the compile prices from a synthetic one.

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

POSEIDON_OPTIMIZE __global__ void unprofiled(const double *x, double *y,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = sqrt(x[i]) * y[i];
}

// SOLVE: [poseidon] {{.*}}: 1 function(s) a rewrite reached; every other function as written

// REFUSED: [poseidon] every site of {{.*}} was left alone; emitting the module as written

// FNS: NAME <_Z4sitePKdPdi>:
// FNS-NEXT: NAME <_Z5otherPKdPdi>:
// FNS: v_rsq_f64
// FNS: NAME <_Z10unprofiledPKdPdi>:
// FNS: v_rsq_f64

// FPC-SITE: v_fma_f64
// FPC-SITE: v_fma_f64

// USE-SITE: v_cvt_f32_f64
// USE-SITE: v_fma_f32
// USE-SITE: v_cvt_f64_f32

// REQUIRES: nvptx-registered-target
// DEFINE: %{driver} = %poseidon_clangxx -x cuda --cuda-device-only \
// DEFINE:   --cuda-gpu-arch=sm_120 -nocudainc -nocudalib -O2 -S \
// DEFINE:   -poseidon-profile-use=%S/../Inputs/ds_expansion -poseidon-enable-herbie=false \
// DEFINE:   -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
// DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
// DEFINE:   -poseidon-cost-budget=-1e-9 %s
// RUN: rm -rf %t && mkdir %t
// RUN: %{driver} -poseidon-cache=%t/on -poseidon-print -o %t/on.ptx 2>&1 | FileCheck %s
// RUN: %{driver} -poseidon-cache=%t/fast -ffp-contract=fast -poseidon-print -o %t/fast.ptx 2>&1 | FileCheck %s
// RUN: %{driver} -poseidon-cache=%t/fastmath -ffast-math -poseidon-print -o %t/fastmath.ptx 2>&1 | FileCheck %s
// RUN: sed -n '/\.entry .*fms_kernel/,/^}/p' %t/on.ptx | FileCheck --check-prefix=FMA --implicit-check-not=fma.rn.f32 %s
// RUN: sed -n '/\.entry .*fms_kernel/,/^}/p' %t/fast.ptx | FileCheck --check-prefix=FMA --implicit-check-not=fma.rn.f32 %s
// RUN: sed -n '/\.entry .*fms_kernel/,/^}/p' %t/fastmath.ptx | FileCheck --check-prefix=FMA --implicit-check-not=fma.rn.f32 %s
// UNSUPPORTED: global-fp-contract

// The driver's -ffp-contract=on, and a user's -ffp-contract=fast or -ffast-math
// after it, which marks every floating-point operation 'contract'. From LLVM 24
// that mark reaches only the operations that carry it and the expansion code
// carries none, so the site of Inputs/ds_expansion gets its expansion in all
// three modes, and the kernel's FP32 fused multiply-adds are the expansion's
// own 15 (three per fmuladd): nothing of the expansion is fused further.

#include "poseidon/poseidon.h"

#define DEVICE __attribute__((device))
template <typename R, typename... T> DEVICE R __poseidon_fp_optimize(void *, T...);
DEVICE int enzyme_const;
DEVICE int enzyme_dup;

extern "C" DEVICE __attribute__((noinline)) void
fms_body(const double *x, const double *y, double *out) {
  unsigned i = __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() +
               __nvvm_read_ptx_sreg_tid_x();
  double a = x[i], b = y[i];
  double t = a * b + a;
  double u = t * t - b;
  double v = u * b + t;
  double w = v * v - a;
  out[i] = w * u + v;
}

__attribute__((global)) void fms_kernel(const double *x, const double *y,
                                        double *out, double *dout) {
  __poseidon_fp_optimize<void>((void *)fms_body, enzyme_const, x, enzyme_const,
                               y, enzyme_dup, out, dout);
}

// CHECK-NOT: are not proposed
// CHECK: Applying solution for precision change All FP64(0%) + Expansion2(100%)

// FMA-COUNT-15: fma.rn.f32

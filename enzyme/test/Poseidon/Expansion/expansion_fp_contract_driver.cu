// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -poseidon-profile-use=%S/../Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1e-9 -poseidon-cache=%t/on %s -o %t/on.ptx 2>&1 | FileCheck %s
// RUN: %poseidon_clangxx -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -poseidon-profile-use=%S/../Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1e-9 -poseidon-cache=%t/fast -ffp-contract=fast %s -o %t/fast.ptx 2>&1 | FileCheck %s
// RUN: %poseidon_clangxx -x cuda --cuda-device-only --cuda-gpu-arch=%gpu_arch -nocudainc -nocudalib -O2 -S -poseidon-profile-use=%S/../Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1e-9 -poseidon-cache=%t/fastmath -ffast-math %s -o %t/fastmath.ptx 2>&1 | FileCheck %s
// RUN: sed -n '/\.entry .*fms_kernel/,/^}/p' %t/on.ptx | grep fma.rn.f32 | count 15
// RUN: sed -n '/\.entry .*fms_kernel/,/^}/p' %t/fast.ptx | grep fma.rn.f32 | count 15
// RUN: sed -n '/\.entry .*fms_kernel/,/^}/p' %t/fastmath.ptx | grep fma.rn.f32 | count 15
// UNSUPPORTED: global-fp-contract

// What clang emits under each contraction mode, through the driver. Under the
// driver's own -ffp-contract=on a contractible expression is an llvm.fmuladd
// call and no operation carries 'contract'; a user's -ffp-contract=fast or
// -ffast-math comes after the driver's flag and wins, and clang marks every
// floating-point operation 'contract'. From LLVM 24 that mark reaches only
// the operations that carry it, and the expansion code carries none, so the
// site of Inputs/ds_expansion gets its expansion in all three modes, and the
// kernel's FP32 fused multiply-adds are the expansion's own 15 (three per
// fmuladd) in each: nothing of the expansion is fused further.

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

// CHECK-NOT: NOT PROPOSED
// CHECK: Applying solution for CS: All FP64(0%) + Expansion2(100%)

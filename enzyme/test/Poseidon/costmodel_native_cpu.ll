; RUN: rm -rf %t && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t/native 2>&1 | FileCheck --check-prefix=NATIVE %s
; RUN: (%opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cost-model=%S/Inputs/cm_cpu_missing.csv -poseidon-cache=%t/missing 2>&1 || true) | FileCheck --check-prefix=MISSING %s
; RUN: (%opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cost-model=%S/Inputs/cm_cpu_foreign.csv -poseidon-cache=%t/foreign 2>&1 || true) | FileCheck --check-prefix=FOREIGN %s
; RUN: (%opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cost-model=%S/Inputs/cm_cpu_noarch.csv -poseidon-cache=%t/noarch 2>&1 || true) | FileCheck --check-prefix=NOARCH %s
; RUN: (%opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cache=%t/none 2>&1 || true) | FileCheck --check-prefix=NOMODEL %s
; REQUIRES: poseidon
; The measured CSV is the only price source. A model whose native_arch matches the function's target-cpu is consumed as is (2 FP64 ops at 4 = 8); a model missing a required row, a model measured on another device, a model with no native_arch header and no model at all each abort instead of substituting a price.

define double @tester_a(double %x, double %y) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %x
  ret double %mul
}

define double @site_a(double %x, double %y) #0 {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_a, double %x, double %y, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; NATIVE: Initial ComputationCost: 8.000000e+00
; NATIVE: {{^[-+.0-9e]+}} -192 All FP64(0%) + FP32(100%)
; NATIVE: [poseidon] Finished Optimization

; MISSING: Custom cost model: entry not found for fmul @ double
; MISSING-NOT: Initial ComputationCost
; MISSING-NOT: [poseidon] Finished Optimization

; FOREIGN: was measured on native_arch 'sm_120' but preprocess_tester_a compiles for target-cpu 'x86-64'
; FOREIGN-NOT: Initial ComputationCost
; FOREIGN-NOT: [poseidon] Finished Optimization

; NOARCH: carries no '# native_arch=' header
; NOARCH-NOT: Initial ComputationCost
; NOARCH-NOT: [poseidon] Finished Optimization

; NOMODEL: no cost model for target-cpu 'x86-64' in {{.*}}; measure this device with poseidon-calibrate, or pass -poseidon-cost-model=<csv>
; NOMODEL-NOT: Initial ComputationCost
; NOMODEL-NOT: [poseidon] Finished Optimization

; RUN: rm -rf %t && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/native 2>&1 | FileCheck --check-prefix=NATIVE %s
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cost-model=%cost_models/cm_cpu_missing.csv -poseidon-cache=%t/missing 2>&1 | FileCheck --check-prefix=MISSING %s
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cost-model=%cost_models/cm_cpu_foreign.csv -poseidon-cache=%t/foreign 2>&1 | FileCheck --check-prefix=FOREIGN %s
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cost-model=%cost_models/cm_cpu_noarch.csv -poseidon-cache=%t/noarch 2>&1 | FileCheck --check-prefix=NOARCH %s
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print -S -poseidon-cache=%t/none 2>&1 | FileCheck --check-prefix=NOMODEL %s
; RUN: (cat %cost_models/cm_cpu_native.csv; echo 'fadd,float,1') > %t/twice.csv
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -S -poseidon-cost-model=%t/twice.csv -poseidon-cache=%t/twice 2>&1 | FileCheck --check-prefix=TWICE %s
; RUN: (cat %cost_models/cm_cpu_native.csv; echo 'fadd,float') > %t/short.csv
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/joint_profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -S -poseidon-cost-model=%t/short.csv -poseidon-cache=%t/short 2>&1 | FileCheck --check-prefix=SHORT %s
; The measured CSV is the only price source. A model whose native_arch matches the function's target-cpu is consumed as is (2 FP64 ops at 4 = 8); a model missing a required row, a model measured on another device, a model with no native_arch header, a model with a row twice or a line that is not a row, and no model at all each stop the compile with an error instead of substituting a price.

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
; NATIVE: {{^[-+.0-9e]+}} -1.920000e+02 All FP64(0%) + FP32(100%)
; NATIVE: [poseidon] Finished Optimization

; MISSING: error: {{.*}}Poseidon: cost model {{.*}}cm_cpu_missing.csv has no row 'fmul,double'; measure it with poseidon-calibrate
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

; TWICE: error: {{.*}}Poseidon: cost model {{.*}}twice.csv has the row 'fadd,float' twice
; TWICE-NOT: [poseidon] Finished Optimization

; SHORT: error: {{.*}}Poseidon: cost model {{.*}}short.csv line {{[0-9]+}} is not '<opcode>,<type>,<cost>': fadd,float
; SHORT-NOT: [poseidon] Finished Optimization

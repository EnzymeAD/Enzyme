; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-200 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=SOLVE,PICK %s
; RUN: head -c $(( $(wc -c < %t/c/table.json) / 2 )) %t/c/table.json > %t/half.json && mv %t/half.json %t/c/table.json
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-200 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=TRUNC,SOLVE,PICK %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-comp-cost-budget=-200 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=HIT,PICK %s
; REQUIRES: poseidon

; A table.json cut off in the middle (an interrupted write) is reported,
; solved again and overwritten; the compile after that reads it back.

define void @tester(double %T, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %l = call double @llvm.log.f64(double %T)
  %nl = fneg double %l
  %e = call double @llvm.fma.f64(double %nl, double 1.790000e-02, double -1.484800e+00)
  %p = call double @llvm.pow.f64(double %T, double %e)
  %k = fmul double %p, 1.000000e-20
  %r1 = fmul double %k, %y
  %h = fmul double %nl, %y
  %r2 = fadd double %h, %T
  store double %r1, ptr %o1, align 8
  store double %r2, ptr %o2, align 8
  ret void
}

define void @site(double %T, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %T, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2)
  ret void
}

declare double @llvm.log.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.pow.f64(double, double)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; TRUNC: [poseidon] DP cache entry for preprocess_tester is unreadable ({{.*}}table.json: {{.*}}); re-solving.
; SOLVE: Solving the DP for preprocess_tester.
; SOLVE: DP table for preprocess_tester cached to file.
; HIT-NOT: DP cache entry
; HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; HIT-NOT: cached to file
; PICK: Applying solution for CS: Funcs FP64(70%) + FP32(30%) (#10)

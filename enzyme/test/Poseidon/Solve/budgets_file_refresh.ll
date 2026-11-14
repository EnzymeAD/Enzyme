; RUN: rm -rf %t && mkdir -p %t/p2
; RUN: sed '0,/Exec = 64/s//Exec = 128/' %S/../Inputs/pow_term/out0/preprocess_tester.fpprofile > %t/p2/preprocess_tester.fpprofile
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=FIRST %s
; RUN: cp %t/c/budgets.txt %t/first_budgets.txt
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/p2 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=SECOND %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/p2 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/fresh -S -o /dev/null 2>&1 | FileCheck --check-prefix=FIRST %s
; RUN: not diff %t/first_budgets.txt %t/fresh/budgets.txt
; RUN: diff %t/fresh/budgets.txt %t/c/budgets.txt

; budgets.txt lists the breakpoints of the last DP table solved into the cache.
; The second solve, under a profile in which the pow runs twice as often, has
; other breakpoints (twice the executions of every operation would leave every
; share of the site's cost, and so every breakpoint, as it was), and
; budgets.txt has to be the one a solve into an empty cache writes.

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

; FIRST: Solving the DP for preprocess_tester.
; SECOND: DP cache entry for preprocess_tester was solved for a different profile; re-solving.
; SECOND: Solving the DP for preprocess_tester.

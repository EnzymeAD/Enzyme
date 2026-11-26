; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: %python %S/Inputs/expression_depth_limit/chain.py %t/tester.ll %t/prof/tester.fpprofile
; RUN: cat %s %t/tester.ll | not %opt %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/prof -poseidon-enable-algebraic=true -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck %s

; An output deeper than the deepest expression the algebraic search forms is a
; user error naming the operation where the limit is crossed.

; CHECK: error: {{.*}}Poseidon: the expression of + is deeper than 100 operations; compile the site without loop unrolling

define double @test_opt(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=SOLVE,PICK %s
; RUN: sed -e '/^ *"compiler[^"]*":/d' -e 's/^\( *"-536870912": \)[0-9][0-9.e+-]*,$/\11,/' %t/c/table.json > %t/table.json && mv %t/table.json %t/c/table.json
; RUN: grep -q '"-536870912": 1,' %t/c/table.json
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=FOREIGN,PICK %s

; A table.json entry whose key names no compiler is a miss. The cached table
; gets other numbers (the point the solve picks made the worst) and loses any
; key part naming the compiler; the next compile solves again.

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

; SOLVE: Solving the DP for preprocess_tester.
; SOLVE: DP table for preprocess_tester cached to file.

; FOREIGN-NOT: Loaded DP tables from cache
; FOREIGN: [poseidon] DP cache entry for preprocess_tester {{.*}}; re-solving.
; FOREIGN: Solving the DP for preprocess_tester.

; PICK: Applying solution for CS: Funcs FP64(70%) + FP32(30%) (#10)

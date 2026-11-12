; RUN: rm -rf %t && mkdir -p %t/p
; RUN: sed '0,/Exec = 64/s//Exec = 65/' %S/../Inputs/pow_term/profiles/preprocess_tester.fpprofile > %t/p/preprocess_tester.fpprofile
; RUN: sed 's/^fadd,float,1$/fadd,float,2/' %cost_models/cm_cpu_native.csv > %t/cm.csv
; RUN: sed 's/fmul double %%p, 1.000000e-20/fmul double %%p, 2.000000e-20/' %s > %t/const.ll
; RUN: sed 's/call double @scale_a(/call double @scale_b(/' %s > %t/callee.ll
; RUN: not cmp -s %S/../Inputs/pow_term/profiles/preprocess_tester.fpprofile %t/p/preprocess_tester.fpprofile
; RUN: not cmp -s %cost_models/cm_cpu_native.csv %t/cm.csv
; RUN: not cmp -s %s %t/const.ll
; RUN: not cmp -s %s %t/callee.ll
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/base -S -o /dev/null 2>&1 | FileCheck --check-prefixes=FRESH,SOLVED %s
; RUN: cp -r %t/base %t/same
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/same -S -o /dev/null 2>&1 | FileCheck --check-prefix=HIT %s
; RUN: cp -r %t/base %t/profile
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/p -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/profile -S -o /dev/null 2>&1 | FileCheck --check-prefixes=PROFILE,SOLVED %s
; RUN: cp -r %t/base %t/costmodel
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%t/cm.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/costmodel -S -o /dev/null 2>&1 | FileCheck --check-prefixes=COSTMODEL,SOLVED %s
; RUN: cp -r %t/base %t/flag
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/flag -poseidon-random-seed=7 -S -o /dev/null 2>&1 | FileCheck --check-prefixes=FLAG,SOLVED %s
; RUN: cp -r %t/base %t/const
; RUN: %opt %t/const.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/const -S -o /dev/null 2>&1 | FileCheck --check-prefixes=BODY,SOLVED %s
; RUN: cp -r %t/base %t/callee
; RUN: %opt %t/callee.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/callee -S -o /dev/null 2>&1 | FileCheck --check-prefixes=BODY,SOLVED %s

; Each compile starts from the first one's cache and changes one input of the
; DP table (a profile byte, a cost-model row, a solve flag, a constant of the
; body, the function it calls): it names that input and solves again. An
; unchanged compile reads the table back.

define void @tester(double %T, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %l = call double @llvm.log.f64(double %T)
  %nl = fneg double %l
  %e = call double @llvm.fma.f64(double %nl, double 1.790000e-02, double -1.484800e+00)
  %p = call double @llvm.pow.f64(double %T, double %e)
  %k = fmul double %p, 1.000000e-20
  %ys = call double @scale_a(double %y)
  %r1 = fmul double %k, %ys
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

declare double @scale_a(double)
declare double @scale_b(double)
declare double @llvm.log.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.pow.f64(double, double)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; FRESH-NOT: DP cache entry
; HIT-NOT: DP cache entry
; PROFILE: [poseidon] DP cache entry for preprocess_tester was solved for a different profile; re-solving.
; COSTMODEL: [poseidon] DP cache entry for preprocess_tester was solved for a different cost model; re-solving.
; FLAG: [poseidon] DP cache entry for preprocess_tester was solved for a different -poseidon-random-seed; re-solving.
; BODY: [poseidon] DP cache entry for preprocess_tester was solved for a different function body; re-solving.
; SOLVED: Solving the DP for preprocess_tester.
; SOLVED: DP table for preprocess_tester cached to file.
; HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; HIT-NOT: Solving the DP

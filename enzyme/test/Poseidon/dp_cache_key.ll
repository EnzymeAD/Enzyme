; RUN: rm -rf %t && mkdir -p %t/old %t/bad
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/profiles -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=A %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=B,PICK-B %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/fresh -S -o /dev/null 2>&1 | FileCheck --check-prefix=PICK-B %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=HIT,PICK-B %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/c -poseidon-num-samples=512 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SAMPLES %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.4 -poseidon-cache=%t/c -poseidon-num-samples=512 -poseidon-tau-cheapest=false -S -o /dev/null 2>&1 | FileCheck --check-prefix=HIT %s
; RUN: echo '{"costModelFingerprint": "0", "candidateKeys": true, "functions": {"preprocess_tester": {"costToAccuracyMap": {"0": 0}, "costToSolutionMap": {"0": []}}}}' > %t/old/table.json
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/old -S -o /dev/null 2>&1 | FileCheck --check-prefixes=OLD,PICK-B %s
; RUN: echo '{"functions": {"preprocess_tester": nan}}' > %t/bad/table.json
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/bad -S -o /dev/null 2>&1 | FileCheck --check-prefixes=BAD,PICK-B %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/pow_term/out0 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/bad -S -o /dev/null 2>&1 | FileCheck --check-prefixes=HIT,PICK-B %s

; A DP table is cached per site together with the key of everything it was
; solved from. The site below is solved under profile A and then, against the
; same cache, under profile B of the same canonical form (Inputs/pow_term/out0,
; recorded with only the first output seeded). B's solve picks another
; precision split than A's; replaying A's table would have applied A's pick and
; reported A's accuracy. The second compile re-solves, says why, and picks what
; a solve under B in an empty cache picks; a third one reads the table back.
;
; A flag the table depends on (-poseidon-num-samples) re-solves too; one that
; only selects from the table (the budget, -poseidon-tau-cheapest) does not. A
; table in the format of an earlier build, or one that does not parse, is
; reported, solved again and overwritten.

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

; Each run's lines in the order they are printed.
; A-NOT: DP cache entry
; HIT-NOT: DP cache entry
; B: [poseidon] DP cache entry for preprocess_tester was solved for a different profile; re-solving.
; SAMPLES: [poseidon] DP cache entry for preprocess_tester was solved for a different -poseidon-num-samples; re-solving.
; OLD: [poseidon] DP cache entry for preprocess_tester was written in another cache format; re-solving.
; BAD: [poseidon] DP cache entry for preprocess_tester is unreadable ({{.*}}table.json: {{.*}}); re-solving.
; A: Solving the DP for preprocess_tester.
; B: Solving the DP for preprocess_tester.
; SAMPLES: Solving the DP for preprocess_tester.
; OLD: Solving the DP for preprocess_tester.
; BAD: Solving the DP for preprocess_tester.
; HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; A: DP table for preprocess_tester cached to file.
; B: DP table for preprocess_tester cached to file.
; OLD: DP table for preprocess_tester cached to file.
; BAD: DP table for preprocess_tester cached to file.
; HIT-NOT: cached to file
; A: Applying solution for CS: All FP64(70%) + FP32(30%)
; PICK-B: Applying solution for CS: Funcs FP64(70%) + FP32(30%)

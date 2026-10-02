; RUN: rm -rf %t && mkdir -p %t && cp %S/Inputs/fma_opt_cache/cachedHerbieOutput_* %t/
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t -S 2>&1 | FileCheck --check-prefixes=CHECK,CHEAPEST %s
; RUN: rm -rf %t2 && mkdir -p %t2 && cp %S/Inputs/fma_opt_cache/cachedHerbieOutput_* %t2/
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/fma_opt_profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-pt=false -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-cache=%t2 -poseidon-tau-cheapest=false -S 2>&1 | FileCheck --check-prefixes=CHECK,ACCURATE %s
; REQUIRES: poseidon

; Cached Herbie output replayed from Inputs/fma_opt_cache without invoking Herbie: (x + y) * z  -->  fma(x, z, y * z)
;
; The site's tolerance of 0.5 is an enormous per-operation relative rounding
; level, so the two arms take opposite ends of the same cached frontier:
; CHEAPEST (the default) takes the binary32 core, ACCURATE the binary64 one.

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @test_opt(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; CHECK: Using cached Herbie output from {{.*}}cachedHerbieOutput_default_f9f2bd660c90d989_0_0.txt

; The selection line reports the tolerance in the unit it is compared in: the
; original FP64 body sits at A0/S = 1.08e-15, the chosen point at `rel`.
; CHEAPEST: [poseidon] preprocess_tester: tau=5.000000e-01 S={{.*}} A0={{.*}} selected cost=-200 accCost={{.*}} rel={{.*}}
; ACCURATE: [poseidon] preprocess_tester: tau=5.000000e-01 S={{.*}} A0={{.*}} selected cost=800 accCost={{.*}} rel={{.*}}

; CHECK: define double @test_opt(double %x, double %y, double %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[call:.+]] = call double @preprocess_tester(double %x, double %y, double %z)
; CHECK-NEXT:   ret double %[[call]]

; CHECK: define double @preprocess_tester(double %x, double %y, double %z)
; CHECK-NEXT: entry:

; CHEAPEST-NEXT:   %[[cy:.+]] = fptrunc fast double %y to float
; CHEAPEST-NEXT:   %[[cz:.+]] = fptrunc fast double %z to float
; CHEAPEST-NEXT:   %[[cx:.+]] = fptrunc fast double %x to float
; CHEAPEST-NEXT:   %[[mul32:.+]] = fmul fast float %[[cz]], %[[cx]]
; CHEAPEST-NEXT:   %[[fma32:.+]] = tail call fast float @llvm.fma.f32(float %[[cy]], float %[[cz]], float %[[mul32]])
; CHEAPEST-NEXT:   %[[back:.+]] = fpext fast float %[[fma32]] to double
; CHEAPEST-NEXT:   ret double %[[back]]

; ACCURATE-NEXT:   %[[fmax:.+]] = tail call fast double @llvm.maxnum.f64(double %x, double %y)
; ACCURATE-NEXT:   %[[fmin:.+]] = tail call fast double @llvm.minnum.f64(double %x, double %y)
; ACCURATE-NEXT:   %[[mul:.+]] = fmul fast double %z, %[[fmin]]
; ACCURATE-NEXT:   %[[fma:.+]] = tail call fast double @llvm.fma.f64(double %[[fmax]], double %z, double %[[mul]])
; ACCURATE-NEXT:   ret double %[[fma]]

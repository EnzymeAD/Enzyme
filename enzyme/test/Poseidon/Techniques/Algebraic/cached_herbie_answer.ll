; RUN: rm -rf %t && mkdir %t
; RUN: mkdir -p %t/c && cp %S/../../Inputs/fma_opt_cache/herbie_*.json %t/c/
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../../Inputs/fma_site_profile -poseidon-early-prune -poseidon-enable-algebraic=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c -poseidon-print -S 2>&1 | FileCheck --check-prefixes=CHECK,CHEAPEST %s
; RUN: mkdir -p %t/c2 && cp %S/../../Inputs/fma_opt_cache/herbie_*.json %t/c2/
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../../Inputs/fma_site_profile -poseidon-early-prune -poseidon-enable-algebraic=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/c2 -poseidon-tau-cheapest=false -poseidon-print -S 2>&1 | FileCheck --check-prefixes=CHECK,ACCURATE %s
; RUN: %opt < %s %loadPoseidon -passes=poseidon,poseidon-finalize -poseidon-profile-use=%S/../../Inputs/fma_site_profile -poseidon-early-prune -poseidon-enable-algebraic=true -poseidon-herbie-binary=/nonexistent/herbie -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/long -poseidon-herbie-max-expr-length=1 -pass-remarks-missed=poseidon -o /dev/null 2>&1 | FileCheck --check-prefix=LONG %s

; Herbie answers replayed from Inputs/fma_opt_cache for (x + y) * z: at the
; site's tau of 0.5 the default takes the cheapest point of the cached frontier,
; binary32 (CHEAPEST), -poseidon-tau-cheapest=false the binary64 one (ACCURATE);
; an output over -poseidon-herbie-max-expr-length is not sent, with a remark.

; LONG: remark: <unknown>:0:0: Poseidon does not send Herbie an output expression of {{[0-9]+}} characters, over -poseidon-herbie-max-expr-length=1{{$}}

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

; CHECK: [poseidon] Using cached Herbie output from {{.*}}herbie_default_958ABA9C1BC0AE00.json

; CHEAPEST: [poseidon] preprocess_tester: tau=5.000000e-01 S=1.500000e+02 A0=1.618931e-13 selected cost=-2.500000e-01 accCost=7.671461e-05 rel=5.114307e-07{{$}}
; CHEAPEST: Applying solution for (* (+ v0 v1) v2) --(0)-> (fma.f32 v1 v2 (*.f32 v2 v0)){{$}}
; CHEAPEST: [poseidon] pin: tester:A1=(fma.f32 v1 v2 (*.f32 v2 v0)){{$}}
; ACCURATE: [poseidon] preprocess_tester: tau=5.000000e-01 S=1.500000e+02 A0=1.618931e-13 selected cost=1.000000e+00 accCost=-5.436190e-14 rel=7.168745e-16{{$}}
; ACCURATE: Applying solution for (* (+ v0 v1) v2) --(0)-> (fma.f64 (fmax.f64 v0 v1) v2 (*.f64 v2 (fmin.f64 v0 v1))){{$}}
; ACCURATE: [poseidon] pin: tester:A0=(fma.f64 (fmax.f64 v0 v1) v2 (*.f64 v2 (fmin.f64 v0 v1))){{$}}

; CHECK: define double @test_opt(double %x, double %y, double %z)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[call:.+]] = call double @preprocess_tester(double %x, double %y, double %z)
; CHECK-NEXT:   ret double %[[call]]{{$}}

; CHECK: define double @preprocess_tester(double %x, double %y, double %z)
; CHECK-NEXT: entry:

; CHEAPEST-NEXT:   %[[cy:.+]] = fptrunc fast double %y to float
; CHEAPEST-NEXT:   %[[cz:.+]] = fptrunc fast double %z to float
; CHEAPEST-NEXT:   %[[cx:.+]] = fptrunc fast double %x to float
; CHEAPEST-NEXT:   %[[mul32:.+]] = fmul fast float %[[cz]], %[[cx]]{{$}}
; CHEAPEST-NEXT:   %[[fma32:.+]] = tail call fast float @llvm.fma.f32(float %[[cy]], float %[[cz]], float %[[mul32]])
; CHEAPEST-NEXT:   %[[back:.+]] = fpext fast float %[[fma32]] to double
; CHEAPEST-NEXT:   ret double %[[back]]{{$}}

; ACCURATE-NEXT:   %[[fmax:.+]] = tail call fast double @llvm.maxnum.f64(double %x, double %y)
; ACCURATE-NEXT:   %[[fmin:.+]] = tail call fast double @llvm.minnum.f64(double %x, double %y)
; ACCURATE-NEXT:   %[[mul:.+]] = fmul fast double %z, %[[fmin]]{{$}}
; ACCURATE-NEXT:   %[[fma:.+]] = tail call fast double @llvm.fma.f64(double %[[fmax]], double %z, double %[[mul]])
; ACCURATE-NEXT:   ret double %[[fma]]{{$}}

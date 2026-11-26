; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/one_rewrite_per_output %opt < %S/../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-early-prune -poseidon-enable-algebraic=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t -S -poseidon-print 2>&1 | FileCheck %s

; The binary64 and binary32 searches of one output return the same rewrite:
; the solution applies and names it once.

; CHECK: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (/.f64 (*.f64 v1 v1) (+.f64 (hypot.f64 v0 v1) v0)){{$}}
; CHECK-NOT: Applying solution for
; CHECK: [poseidon] pin: tester:A0=(/.f64 (*.f64 v1 v1) (+.f64 (hypot.f64 v0 v1) v0)){{$}}
; CHECK: define double @test_opt(
; CHECK-NEXT: entry:
; CHECK-NEXT: [[R:%.+]] = call double @preprocess_tester(double %x, double %y)
; CHECK-NEXT: ret double [[R]]{{$}}
; CHECK: define double @preprocess_tester(double %x, double %y)
; CHECK-NEXT: entry:
; CHECK-NEXT: [[YY:%.+]] = fmul double %y, %y
; CHECK-NEXT: [[H:%.+]] = tail call double @hypot(double %x, double %y)
; CHECK-NEXT: [[S:%.+]] = fadd double %x, [[H]]{{$}}
; CHECK-NEXT: [[D:%.+]] = fdiv double [[YY]], [[S]]{{$}}
; CHECK-NEXT: ret double [[D]]{{$}}
; CHECK-NEXT: }
; CHECK-NOT: call {{.*}}@hypot

; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/one_rewrite_per_output %opt < %S/../Inputs/regime_split/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t -S 2>&1 | FileCheck %s

; The binary64 and binary32 searches of one output return the same exact
; rewrite. Each search is its own item, so a solve that may take one rewrite
; from each counts the rewrite's accuracy gain twice; only one is ever
; materialized, and the solution may name only one.

; CHECK: !!! Applying solution (1 step(s)) ... !!!
; CHECK-NEXT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(0)-> (/.f64 (*.f64 v1 v1) (+.f64 (hypot.f64 v0 v1) v0)){{$}}
; CHECK-NOT: Applying solution for
; CHECK: !!! Solution applied !!!
; CHECK: [poseidon] pin: tester:R0=(/.f64 (*.f64 v1 v1) (+.f64 (hypot.f64 v0 v1) v0)){{$}}
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

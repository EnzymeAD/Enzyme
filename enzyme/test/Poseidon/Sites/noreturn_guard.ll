; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-apply-rewrites=R0_1 -poseidon-cache=%t -S 2>&1 | FileCheck %s

; The rewritten clone is the code that ships, so its guard in front of a call
; that never returns has to survive canonicalization: the clone branches to
; exit() exactly when y <= 0 and otherwise computes the R0_1 rewrite.

define double @tester(double %x, double %y) #0 {
entry:
  %bad = fcmp ole double %y, 0.000000e+00
  br i1 %bad, label %abort, label %ok

abort:
  call void @exit(i32 1)
  unreachable

ok:
  %xx = fmul double %x, %x
  %yy = fmul double %y, %y
  %s = fadd double %xx, %yy
  %r = call double @llvm.sqrt.f64(double %s)
  %d = fsub double %r, %x
  ret double %d
}

define double @test_opt(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

declare void @exit(i32) noreturn nounwind
declare double @llvm.sqrt.f64(double)
declare double @__poseidon_fp_optimize(double (double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; CHECK: [poseidon] Selecting R0_1: (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) -> (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; CHECK-LABEL: define double @preprocess_tester(
; CHECK-SAME: double [[X:%[a-z0-9._]+]], double [[Y:%[a-z0-9._]+]])
; CHECK-NOT: @llvm.assume
; CHECK-NEXT: entry:
; CHECK-NEXT: [[C:%[a-z0-9._]+]] = fcmp {{.*}}ugt double [[Y]], 0.000000e+00
; CHECK-NEXT: br i1 [[C]], label %[[OK:[a-z0-9._]+]], label %[[ABORT:[a-z0-9._]+]]
; CHECK-EMPTY:
; CHECK-NEXT: [[ABORT]]:
; CHECK-NEXT: call void @exit(i32 1)
; CHECK-NEXT: unreachable
; CHECK-EMPTY:
; CHECK-NEXT: [[OK]]:
; CHECK-NEXT: [[YY:%[a-z0-9._]+]] = fmul double [[Y]], [[Y]]{{$}}
; CHECK-NEXT: [[XX:%[a-z0-9._]+]] = fmul double [[X]], [[X]]{{$}}
; CHECK-NEXT: [[S:%[a-z0-9._]+]] = fadd double [[XX]], [[YY]]{{$}}
; CHECK-NEXT: [[R:%[a-z0-9._]+]] = {{(tail )?}}call double @llvm.sqrt.f64(double [[S]])
; CHECK-NEXT: [[D:%[a-z0-9._]+]] = fadd double [[X]], [[R]]{{$}}
; CHECK-NEXT: [[Q:%[a-z0-9._]+]] = fdiv double [[YY]], [[D]]{{$}}
; CHECK-NEXT: ret double [[Q]]{{$}}
; CHECK-NEXT: }

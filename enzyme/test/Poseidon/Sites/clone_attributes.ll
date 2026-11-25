; RUN: rm -rf %t && split-file %s %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %t/spin.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-pin=R0_1 -poseidon-cache=%t/spin -S 2>&1 | FileCheck --check-prefix=SPIN --implicit-check-not=@llvm.assume %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %t/exit.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-pin=R0_1 -poseidon-cache=%t/exit -S -poseidon-print 2>&1 | FileCheck --check-prefix=EXIT --implicit-check-not=@llvm.assume %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %t/progress.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-pin=R0_1 -poseidon-cache=%t/progress -S 2>&1 | FileCheck --check-prefix=PROGRESS %s

; The rewritten clone is the code that ships, so it carries the function
; attributes of the original and no more. A body that spins forever on bad
; input, which C allows for a loop with a constant condition, keeps its loop
; (SPIN); one that calls exit() returns only on some paths, so it is not
; willreturn either, and branches to exit() exactly when y <= 0 and otherwise
; computes the R0_1 rewrite (EXIT). Only an original that is mustprogress, as
; every C++ function is, lets the clone drop the spin for an assume (PROGRESS).

; SPIN-LABEL: define double @preprocess_tester(
; SPIN-SAME: double [[X:%[a-z0-9._]+]], double [[Y:%[a-z0-9._]+]]) #[[ATTR:[0-9]+]] {
; SPIN-NEXT: entry:
; SPIN-NEXT: [[C:%[a-z0-9._]+]] = fcmp ugt double [[Y]], 0.000000e+00
; SPIN-NEXT: br i1 [[C]], label %[[OK:[a-z0-9._]+]], label %[[SPIN:[a-z0-9._]+]]
; SPIN-EMPTY:
; SPIN-NEXT: [[SPIN]]:
; SPIN-NEXT: br label %[[SPIN]]{{$}}
; SPIN-EMPTY:
; SPIN-NEXT: [[OK]]:
; SPIN: attributes #[[ATTR]] = { alwaysinline "target-cpu"="x86-64" }

; EXIT: [poseidon] Selecting R0_1: (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) -> (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; EXIT-LABEL: define double @preprocess_tester(
; EXIT-SAME: double [[X:%[a-z0-9._]+]], double [[Y:%[a-z0-9._]+]]) #[[ATTR:[0-9]+]] {
; EXIT-NEXT: entry:
; EXIT-NEXT: [[C:%[a-z0-9._]+]] = fcmp {{.*}}ugt double [[Y]], 0.000000e+00
; EXIT-NEXT: br i1 [[C]], label %[[OK:[a-z0-9._]+]], label %[[ABORT:[a-z0-9._]+]]
; EXIT-EMPTY:
; EXIT-NEXT: [[ABORT]]:
; EXIT-NEXT: call void @exit(i32 1)
; EXIT-NEXT: unreachable
; EXIT-EMPTY:
; EXIT-NEXT: [[OK]]:
; EXIT-NEXT: [[YY:%[a-z0-9._]+]] = fmul double [[Y]], [[Y]]{{$}}
; EXIT-NEXT: [[XX:%[a-z0-9._]+]] = fmul double [[X]], [[X]]{{$}}
; EXIT-NEXT: [[S:%[a-z0-9._]+]] = fadd double [[XX]], [[YY]]{{$}}
; EXIT-NEXT: [[R:%[a-z0-9._]+]] = {{(tail )?}}call double @llvm.sqrt.f64(double [[S]])
; EXIT-NEXT: [[D:%[a-z0-9._]+]] = fadd double [[X]], [[R]]{{$}}
; EXIT-NEXT: [[Q:%[a-z0-9._]+]] = fdiv double [[YY]], [[D]]{{$}}
; EXIT-NEXT: ret double [[Q]]{{$}}
; EXIT-NEXT: }
; EXIT: attributes #[[ATTR]] = { alwaysinline "target-cpu"="x86-64" }

; PROGRESS-LABEL: define double @preprocess_tester(
; PROGRESS-SAME: double {{%[a-z0-9._]+}}, double [[Y:%[a-z0-9._]+]]) #[[ATTR:[0-9]+]] {
; PROGRESS-NEXT: entry:
; PROGRESS-NEXT: [[C:%[a-z0-9._]+]] = fcmp ugt double [[Y]], 0.000000e+00
; PROGRESS-NEXT: call void @llvm.assume(i1 [[C]])
; PROGRESS: attributes #[[ATTR]] = { alwaysinline mustprogress "target-cpu"="x86-64" }

;--- spin.ll
define double @tester(double %x, double %y) #0 {
entry:
  %bad = fcmp ole double %y, 0.000000e+00
  br i1 %bad, label %spin, label %ok

spin:
  br label %spin

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
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @llvm.sqrt.f64(double)
declare double @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

;--- exit.ll
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
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y)
  ret double %0
}

declare void @exit(i32) noreturn nounwind
declare double @llvm.sqrt.f64(double)
declare double @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

;--- progress.ll
define double @tester(double %x, double %y) #0 {
entry:
  %bad = fcmp ole double %y, 0.000000e+00
  br i1 %bad, label %spin, label %ok

spin:
  br label %spin

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
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @llvm.sqrt.f64(double)
declare double @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { mustprogress "target-cpu"="x86-64" }

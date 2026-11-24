; RUN: rm -rf %t && split-file %s %t
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %t/spin.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-apply-rewrites=R0_1 -poseidon-cache=%t/spin -S 2>&1 | FileCheck --check-prefix=SPIN --implicit-check-not=@llvm.assume %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %t/exit.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-apply-rewrites=R0_1 -poseidon-cache=%t/exit -S 2>&1 | FileCheck --check-prefix=EXIT %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %t/progress.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-apply-rewrites=R0_1 -poseidon-cache=%t/progress -S 2>&1 | FileCheck --check-prefix=PROGRESS %s

; The rewritten clone is the code that ships, so it carries the function
; attributes of the original and no more. A body that spins forever on bad
; input, which C allows for a loop with a constant condition, keeps its loop
; (SPIN); one that calls exit() returns only on some paths, so it is not
; willreturn either (EXIT, whose body Sites/noreturn_guard.ll checks). Only an
; original that is mustprogress, as every C++ function is, lets the clone drop
; the spin for an assume (PROGRESS).

; SPIN: [poseidon] Selecting R0_1: (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) -> (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
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
; SPIN-NEXT: [[YY:%[a-z0-9._]+]] = fmul double [[Y]], [[Y]]{{$}}
; SPIN-NEXT: [[XX:%[a-z0-9._]+]] = fmul double [[X]], [[X]]{{$}}
; SPIN-NEXT: [[S:%[a-z0-9._]+]] = fadd double [[XX]], [[YY]]{{$}}
; SPIN-NEXT: [[R:%[a-z0-9._]+]] = {{(tail )?}}call double @llvm.sqrt.f64(double [[S]])
; SPIN-NEXT: [[D:%[a-z0-9._]+]] = fadd double [[X]], [[R]]{{$}}
; SPIN-NEXT: [[Q:%[a-z0-9._]+]] = fdiv double [[YY]], [[D]]{{$}}
; SPIN-NEXT: ret double [[Q]]{{$}}
; SPIN-NEXT: }
; SPIN: attributes #[[ATTR]] = { alwaysinline "target-cpu"="x86-64" }

; EXIT-LABEL: define double @preprocess_tester(
; EXIT-SAME: double {{%[a-z0-9._]+}}, double {{%[a-z0-9._]+}}) #[[ATTR:[0-9]+]] {
; EXIT: attributes #[[ATTR]] = { alwaysinline "target-cpu"="x86-64" }

; PROGRESS-LABEL: define double @preprocess_tester(
; PROGRESS-SAME: double [[X:%[a-z0-9._]+]], double [[Y:%[a-z0-9._]+]]) #[[ATTR:[0-9]+]] {
; PROGRESS-NEXT: entry:
; PROGRESS-NEXT: [[C:%[a-z0-9._]+]] = fcmp ugt double [[Y]], 0.000000e+00
; PROGRESS-NEXT: call void @llvm.assume(i1 [[C]])
; PROGRESS-NEXT: [[YY:%[a-z0-9._]+]] = fmul double [[Y]], [[Y]]{{$}}
; PROGRESS-NEXT: [[XX:%[a-z0-9._]+]] = fmul double [[X]], [[X]]{{$}}
; PROGRESS-NEXT: [[S:%[a-z0-9._]+]] = fadd double [[XX]], [[YY]]{{$}}
; PROGRESS-NEXT: [[R:%[a-z0-9._]+]] = {{(tail )?}}call double @llvm.sqrt.f64(double [[S]])
; PROGRESS-NEXT: [[D:%[a-z0-9._]+]] = fadd double [[X]], [[R]]{{$}}
; PROGRESS-NEXT: [[Q:%[a-z0-9._]+]] = fdiv double [[YY]], [[D]]{{$}}
; PROGRESS-NEXT: ret double [[Q]]{{$}}
; PROGRESS-NEXT: }
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

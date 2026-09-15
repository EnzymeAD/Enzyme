; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; `sub` is called twice from `f`. `%x` is written after the first call and nothing is
; written after the second, so the two call sites differ only in whether argument 0
; may be overwritten before the reverse pass runs. The augmentation generated for
; the first call site caches a superset of what the second needs, so the second
; reuses it -- one augmented primal and one reverse pass for `sub`, rather than a
; copy per call site.

define void @sub(double* %x, double* %out) {
entry:
  %v = load double, double* %x
  %m = fmul double %v, %v
  store double %m, double* %out
  ret void
}

define void @f(double %in, double* %o1, double* %o2) {
entry:
  %x = alloca double
  store double %in, double* %x
  call void @sub(double* %x, double* %o1)
  store double 3.000000e+00, double* %x
  call void @sub(double* %x, double* %o2)
  ret void
}

declare i8* @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)

define void @test(double %in, double* %o1, double* %do1, double* %o2, double* %do2) {
entry:
  %tape = call i8* (...) @__enzyme_augmentfwd(void (double, double*, double*)* @f, double %in, double* %o1, double* %do1, double* %o2, double* %do2)
  call void (...) @__enzyme_reverse(void (double, double*, double*)* @f, double %in, double* %o1, double* %do1, double* %o2, double* %do2, i8* %tape)
  ret void
}

; CHECK: define internal double @augmented_sub(
; CHECK-NOT: define internal double @augmented_sub.

; CHECK: define internal {{.*}} @augmented_f(
; CHECK: call fast double @augmented_sub({{.*}} %x, {{.*}} undef, {{.*}} %o1, {{.*}} %"o1'")
; CHECK: call fast double @augmented_sub({{.*}} %x, {{.*}} undef, {{.*}} %o2, {{.*}} %"o2'")

; CHECK: define internal { double } @diffef(
; CHECK: call void @diffesub({{.*}} %x, {{.*}} %"x'mi", {{.*}} undef, {{.*}} %"o2'", double %tapeArg1)
; CHECK: call void @diffesub({{.*}} %x, {{.*}} %"x'mi", {{.*}} undef, {{.*}} %"o1'", double %tapeArg2)

; CHECK: define internal void @diffesub(
; CHECK-NOT: define internal void @diffesub.

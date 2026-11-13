; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-postopt=1 -S | FileCheck %s --check-prefix=POSTOPT; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-postopt=1 -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

; Regression test for issue #3842:
; Split reverse mode (__enzyme_augmentfwd + __enzyme_reverse) on functions
; referencing an internal global must consistently recognize clones, generated
; augmented/reverse functions, and inlined instructions from clones as belonging
; to the same logical function during ActivityAnalysis.

; CHECK-NOT: @g_orig_shadow
; CHECK-NOT: @g_two_shadow
; CHECK-NOT: @g_inl_shadow
; CHECK-NOT: @g_wo_shadow
; CHECK-NOT: @g_gep_shadow
; CHECK: @g_shared_shadow = internal{{.*}} global double 0.000000e+00

; POSTOPT-NOT: @g_orig_shadow
; POSTOPT-NOT: @g_two_shadow
; POSTOPT-NOT: @g_inl_shadow
; POSTOPT-NOT: @g_wo_shadow
; POSTOPT-NOT: @g_gep_shadow
; POSTOPT: @g_shared_shadow = internal{{.*}} global double 0.000000e+00

; EVAL: wo=6.0 shared=3.0 gep=5.0

@g_orig = internal global [4 x double] zeroinitializer
@g_two = internal global [4 x double] zeroinitializer
@g_inl = internal global [4 x double] zeroinitializer
@g_wo = internal global [4 x double] zeroinitializer
@g_shared = internal global double 0.0
@g_gep = internal global [4 x double] zeroinitializer

@.fmt = private unnamed_addr constant [29 x i8] c"wo=%.1f shared=%.1f gep=%.1f\00"

declare i32 @printf(ptr, ...)
declare ptr @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)
declare double @__enzyme_reverse_f64(...)
declare void @__enzyme_autodiff(...)

; Case 1: Original issue #3842 reproducer (single caller, split reverse mode)
define void @f_orig() {
  call void @sub_orig(ptr @g_orig)
  ret void
}

define void @sub_orig(ptr %p) {
  store i32 0, ptr null, align 4
  ret void
}

define void @test_orig() {
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @f_orig)
  call void (...) @__enzyme_reverse(ptr @f_orig, ptr %tape)
  ret void
}

; Case 2: Split reverse across two separate functions (exercises PostOpt inlining
; of augmented_f_two into test_two_fwd before test_two_rev is lowered)
define void @f_two() {
  call void @sub_two(ptr @g_two)
  ret void
}

define void @sub_two(ptr %p) {
  store i32 0, ptr null, align 4
  ret void
}

define ptr @test_two_fwd() {
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @f_two)
  ret ptr %tape
}

define void @test_two_rev(ptr %tape) {
  call void (...) @__enzyme_reverse(ptr @f_two, ptr %tape)
  ret void
}

; Case 3: Alwaysinline function inlined into another differentiated caller between
; __enzyme_augmentfwd and __enzyme_reverse
define void @f_inl() alwaysinline {
  call void @sub_inl(ptr @g_inl)
  ret void
}

define void @sub_inl(ptr %p) {
  store i32 0, ptr null, align 4
  ret void
}

define void @caller_inl() {
  call void @f_inl()
  ret void
}

define void @test_inl() {
  %tape = call ptr (...) @__enzyme_augmentfwd(ptr @f_inl)
  call void (...) @__enzyme_autodiff(ptr @caller_inl)
  call void (...) @__enzyme_reverse(ptr @f_inl, ptr %tape)
  ret void
}

; Case 4: Executable write-only internal global in split reverse across two functions
define void @sub_wo(ptr %p, double %x) noinline {
  store double %x, ptr %p
  ret void
}

define double @f_wo(double %x) {
  call void @sub_wo(ptr @g_wo, double %x)
  %sq = fmul double %x, %x
  ret double %sq
}

define { ptr, double } @test_wo_fwd(double %x) noinline {
  %res = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @f_wo, double %x)
  ret { ptr, double } %res
}

define double @test_wo_rev(double %x, ptr %tape) noinline {
  %d = call double (...) @__enzyme_reverse_f64(ptr @f_wo, double %x, double 1.0, ptr %tape)
  ret double %d
}

; Case 5: Soundness — test_shared_fwd directly uses @g_shared AND calls
; __enzyme_augmentfwd(@f_shared). Even when PostOpt inlines augmented_f_shared
; into test_shared_fwd, @g_shared must remain active in both augmented & reverse.
define void @sub_shared(ptr %p, double %x) noinline {
  %t = fmul double %x, 3.0
  store double %t, ptr %p
  ret void
}

define double @f_shared(double %x) {
  call void @sub_shared(ptr @g_shared, double %x)
  %v = load double, ptr @g_shared
  ret double %v
}

define { ptr, double } @test_shared_fwd(double %x) noinline {
  %res = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @f_shared, double %x)
  %v = load double, ptr @g_shared
  store double %v, ptr @g_shared
  ret { ptr, double } %res
}

define double @test_shared_rev(double %x, ptr %tape) noinline {
  %d = call double (...) @__enzyme_reverse_f64(ptr @f_shared, double %x, double 1.0, ptr %tape)
  ret double %d
}

; Case 6: GEP of internal global (exercises isValueActivelyStoredOrReturned
; skipping clones of the analyzed function)
define void @sub_gep(ptr %p) noinline {
  store i32 0, ptr %p
  ret void
}

define double @f_gep(double %x, i64 %i) {
  %p = getelementptr inbounds [4 x double], ptr @g_gep, i64 0, i64 %i
  call void @sub_gep(ptr %p)
  %r = fmul double %x, 5.0
  ret double %r
}

define { ptr, double } @test_gep_fwd(double %x) noinline {
  %res = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @f_gep, double %x, i64 0)
  ret { ptr, double } %res
}

define double @test_gep_rev(double %x, ptr %tape) noinline {
  %d = call double (...) @__enzyme_reverse_f64(ptr @f_gep, double %x, i64 0, double 1.0, ptr %tape)
  ret double %d
}

define i32 @main() {
  %r1 = call { ptr, double } @test_wo_fwd(double 3.0)
  %t1 = extractvalue { ptr, double } %r1, 0
  %d1 = call double @test_wo_rev(double 3.0, ptr %t1)

  %r2 = call { ptr, double } @test_shared_fwd(double 4.0)
  %t2 = extractvalue { ptr, double } %r2, 0
  %d2 = call double @test_shared_rev(double 4.0, ptr %t2)

  %r3 = call { ptr, double } @test_gep_fwd(double 7.0)
  %t3 = extractvalue { ptr, double } %r3, 0
  %d3 = call double @test_gep_rev(double 7.0, ptr %t3)

  call i32 (ptr, ...) @printf(ptr @.fmt, double %d1, double %d2, double %d3)
  ret i32 0
}

; CHECK: define internal void @augmented_sub_orig(ptr {{.*}}%p)
; CHECK: define internal void @diffesub_orig(ptr {{.*}}%p)
; CHECK: define internal void @augmented_sub_two(ptr {{.*}}%p)
; CHECK: define internal void @diffesub_two(ptr {{.*}}%p)
; CHECK: define internal void @augmented_sub_inl(ptr {{.*}}%p)
; CHECK: define internal void @diffesub_inl(ptr {{.*}}%p)
; CHECK: define internal void @augmented_sub_wo(ptr {{.*}}%p, double %x)
; CHECK: define internal { double } @diffesub_wo(ptr {{.*}}%p, double %x)
; CHECK: define internal void @augmented_sub_shared(ptr {{.*}}%p, ptr {{.*}}%"p'", double %x)
; CHECK: define internal { double } @diffesub_shared(ptr {{.*}}%p, ptr {{.*}}%"p'", double %x)
; CHECK: define internal void @nofree_sub_gep(ptr {{.*}}%p)
; CHECK: define internal { ptr, double } @augmented_f_gep(double %x, i64 %i)
; CHECK: define internal { double } @diffef_gep(double %x, i64 %i, double %differeturn, ptr %tapeArg)

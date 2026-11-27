; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S 2>&1 | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S 2>&1 | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 8.000000 3.000000

; In forward mode, freeing an inactive pointer has no shadow allocation to
; release, so CallDerivatives must preserve the primal free and return rather
; than falling through into the reverse-mode deallocation cases, which dropped
; both reverse-pass cache frees (!enzyme_cache_free) and plain inactive frees.

; CHECK-NOT: freeing without malloc

@.fmt = private unnamed_addr constant [7 x i8] c"%f %f\0A\00", align 1

declare noalias ptr @malloc(i64)
declare void @free(ptr)
declare double @__enzyme_autodiff(...)
declare double @__enzyme_fwddiff(...)
declare i32 @printf(ptr, ...)

define double @stack_loop(double %x, i64 %n) {
entry:
  %flags = alloca [4 x i64], align 16
  store i64 0, ptr %flags, align 16
  %slot1 = getelementptr inbounds [4 x i64], ptr %flags, i64 0, i64 1
  store i64 1, ptr %slot1, align 8
  %slot2 = getelementptr inbounds [4 x i64], ptr %flags, i64 0, i64 2
  store i64 0, ptr %slot2, align 16
  %slot3 = getelementptr inbounds [4 x i64], ptr %flags, i64 0, i64 3
  store i64 1, ptr %slot3, align 8
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc = phi double [ 1.000000e+00, %entry ], [ %prod, %loop ]
  %idx = and i64 %i, 3
  %slot = getelementptr inbounds [4 x i64], ptr %flags, i64 0, i64 %idx
  %flag = load i64, ptr %slot, align 8
  %flipped = xor i64 %flag, 1
  store i64 %flipped, ptr %slot, align 8
  %is_zero = icmp eq i64 %flag, 0
  %factor = select i1 %is_zero, double %x, double 2.000000e+00
  %prod = fmul double %acc, %factor
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %prod
}

define double @grad_stack_loop(double %x, i64 %n) {
entry:
  %g = call double (...) @__enzyme_autodiff(ptr @stack_loop, double %x, metadata !"enzyme_const", i64 %n)
  ret double %g
}

define double @inactive_heap(double %x) {
entry:
  %mem = call noalias ptr @malloc(i64 8)
  store i64 3, ptr %mem, align 8
  %count = load i64, ptr %mem, align 8
  call void @free(ptr %mem)
  %scale = sitofp i64 %count to double
  %res = fmul double %x, %scale
  ret double %res
}

define i32 @main() {
entry:
  %d2 = call double (...) @__enzyme_fwddiff(ptr @grad_stack_loop, double 5.000000e+00, double 1.000000e+00, metadata !"enzyme_const", i64 4)
  %d1 = call double (...) @__enzyme_fwddiff(ptr @inactive_heap, double 5.000000e+00, double 1.000000e+00)
  call i32 (ptr, ...) @printf(ptr @.fmt, double %d2, double %d1)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @fwddiffediffestack_loop(
; CHECK: %[[IS_ZERO_CACHE:.+]] = tail call noalias nonnull ptr @malloc(
; CHECK: tail call void @free(ptr nonnull %[[IS_ZERO_CACHE]]){{.*}}, !enzyme_cache_free

; CHECK-LABEL: define internal double @fwddiffeinactive_heap(
; CHECK: %mem = call noalias ptr @malloc(i64 8)
; CHECK: call void @free(ptr %mem)

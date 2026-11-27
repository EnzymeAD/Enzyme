; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,adce,loop(loop-deletion),correlated-propagation,%simplifycfg)" -opaque-pointers -S | FileCheck %s; fi

; The allocation in the loop is rematerialized in the reverse pass. The loop
; block's switch has an unreachable default destination and an unreachable
; first case, so the rematerialized block must branch on to the second case.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare noalias nonnull ptr @malloc(i64)

declare void @free(ptr)

declare void @user(ptr)

define double @f(double %x, i64 %n, i64 %k) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %latch ]
  %m = tail call noalias nonnull ptr @malloc(i64 8)
  store double %x, ptr %m, align 8
  switch i64 %k, label %err.default [
    i64 0, label %err.zero
    i64 1, label %latch
  ]

err.default:
  call void @user(ptr %m)
  unreachable

err.zero:
  call void @user(ptr %m)
  unreachable

latch:
  %v = load double, ptr %m, align 8
  %sq = fmul double %v, %v
  tail call void @free(ptr %m)
  %acc.next = fadd double %acc, %sq
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %acc.next
}

define double @df(double %x, i64 %n, i64 %k) {
entry:
  %r = tail call double (...) @__enzyme_autodiff(ptr @f, double %x, i64 %n, i64 %k)
  ret double %r
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffef(double %x, i64 %n, i64 %k, double %differeturn)

; CHECK: latch:
; CHECK-NEXT:   tail call void @free(ptr nonnull %m)
; CHECK-NEXT:   %done = icmp eq i64 %iv.next2, %n
; CHECK-NEXT:   br i1 %done, label %remat_enter, label %loop

; CHECK: remat_enter:
; CHECK-NEXT:   %"x'de.0" = phi double [ %9, %incinvertloop ], [ 0.000000e+00, %latch ]
; CHECK-NEXT:   %"iv1'ac.0" = phi i64 [ %2, %incinvertloop ], [ %0, %latch ]
; CHECK-NEXT:   %remat_m = call noalias nonnull ptr @malloc(i64 8)
; CHECK-NEXT:   %"m'mi" = call noalias nonnull ptr @malloc(i64 8)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr nonnull dereferenceable(8) dereferenceable_or_null(8) %"m'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   store double %x, ptr %remat_m, align 8
; CHECK-NEXT:   %v_unwrap = load double, ptr %remat_m, align 8
; CHECK-NEXT:   %3 = fmul fast double %differeturn, %v_unwrap
; CHECK-NEXT:   %4 = fmul fast double %differeturn, %v_unwrap
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   %6 = load double, ptr %"m'mi", align 8
; CHECK-NEXT:   %7 = fadd fast double %6, %5
; CHECK-NEXT:   store double %7, ptr %"m'mi", align 8
; CHECK-NEXT:   %8 = load double, ptr %"m'mi", align 8
; CHECK-NEXT:   store double 0.000000e+00, ptr %"m'mi", align 8
; CHECK-NEXT:   %9 = fadd fast double %"x'de.0", %8
; CHECK-NEXT:   call void @free(ptr nonnull %"m'mi")
; CHECK-NEXT:   call void @free(ptr nonnull %remat_m)
; CHECK-NEXT:   %10 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   %11 = select fast i1 %10, double 0.000000e+00, double %differeturn
; CHECK-NEXT:   br i1 %10, label %invertentry, label %incinvertloop
; CHECK-NEXT: }

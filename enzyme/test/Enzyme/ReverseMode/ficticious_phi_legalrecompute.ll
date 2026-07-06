; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes=enzyme -S -opaque-pointers | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @sub(double %x) {
entry:
  %call = call double @llvm.sin.f64(double %x)
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %sum = phi double [ 0.0, %entry ], [ %sum.next, %loop ]
  %mul = fmul double %call, %call
  %sum.next = fadd double %sum, %mul
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, 10
  br i1 %cmp, label %loop, label %exit

exit:
  %lcssa = phi double [ %sum.next, %loop ]
  ret double %lcssa
}

declare double @llvm.sin.f64(double)
declare double @__enzyme_autodiff(ptr, double)

define double @test_func(double %x) {
  %res = call double @__enzyme_autodiff(ptr @sub, double %x)
  ret double %res
}

; CHECK: define internal { double } @diffesub

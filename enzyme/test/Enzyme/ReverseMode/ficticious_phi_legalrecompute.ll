; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes=enzyme -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @tester(ptr %p, double %x) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %val = phi double [ %x, %entry ], [ %val.next, %loop ]
  %unused = fadd double %val, 1.0
  %val.next = fmul double %val, %val
  %iv.next = add i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, 10
  br i1 %cmp, label %loop, label %exit

exit:
  %lcssa = phi double [ %unused, %loop ]
  store double %lcssa, ptr %p, align 8
  ret double %val.next
}

declare double @__enzyme_autodiff(ptr, ptr, double)

define double @test_func(ptr %p, double %x) {
  %res = call double @__enzyme_autodiff(ptr @tester, ptr %p, double %x)
  ret double %res
}

; CHECK: define internal void @diffetester

; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes=enzyme -S -opaque-pointers | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @gamma_test(double %x) {
top:
  %cmp1 = fcmp uge double %x, 0.0
  br i1 %cmp1, label %L70.preheader, label %exit

L70.preheader:
  %cmp2 = fcmp ult double %x, 3.0
  br i1 %cmp2, label %L77.preheader, label %L74

L74:
  %val.1 = phi double [ %sub, %L74 ], [ %x, %L70.preheader ]
  %val.2 = phi double [ %mul, %L74 ], [ 1.0, %L70.preheader ]
  %sub = fadd double %val.1, -1.0
  %mul = fmul double %val.2, %sub
  %cmp3 = fcmp ult double %sub, 3.0
  br i1 %cmp3, label %L77.preheader, label %L74

L77.preheader:
  %lcssa.1 = phi double [ 1.0, %L70.preheader ], [ %mul, %L74 ]
  %lcssa.2 = phi double [ %x, %L70.preheader ], [ %sub, %L74 ]
  %cmp4 = fcmp uge double %lcssa.2, 2.0
  br i1 %cmp4, label %L84, label %L81

L81:
  %phi.1 = phi double [ %div, %L81 ], [ %lcssa.1, %L77.preheader ]
  %phi.2 = phi double [ %add, %L81 ], [ %lcssa.2, %L77.preheader ]
  %div = fdiv double %phi.1, %phi.2
  %add = fadd double %phi.2, 1.0
  %cmp5 = fcmp uge double %add, 2.0
  br i1 %cmp5, label %L84, label %L81

L84:
  %res1 = phi double [ %lcssa.1, %L77.preheader ], [ %div, %L81 ]
  %res2 = phi double [ %lcssa.2, %L77.preheader ], [ %add, %L81 ]
  %sub2 = fadd double %res2, -2.0
  %mul2 = fmul double %res1, %sub2
  ret double %mul2

exit:
  ret double %x
}

declare double @__enzyme_autodiff(ptr, double)

define double @test_func(double %x) {
  %res = call double @__enzyme_autodiff(ptr @gamma_test, double %x)
  ret double %res
}

; CHECK: define internal { double } @diffegamma_test

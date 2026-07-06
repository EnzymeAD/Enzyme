; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes=enzyme -S -opaque-pointers | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define double @gamma_like(double %x) {
entry:
  %cmp = fcmp ult double %x, 3.000000e+00
  br i1 %cmp, label %L77.preheader, label %L74

L74:                                              ; preds = %entry, %L74
  %val.1 = phi double [ %sub, %L74 ], [ %x, %entry ]
  %val.2 = phi double [ %mul, %L74 ], [ 1.000000e+00, %entry ]
  %sub = fadd double %val.1, -1.000000e+00
  %mul = fmul double %val.2, %sub
  %loop.cmp = fcmp ult double %sub, 3.000000e+00
  br i1 %loop.cmp, label %L77.preheader, label %L74

L77.preheader:                                    ; preds = %L74, %entry
  %lcssa.1 = phi double [ 1.000000e+00, %entry ], [ %mul, %L74 ]
  %lcssa.2 = phi double [ %x, %entry ], [ %sub, %L74 ]
  %cmp2 = fcmp uge double %lcssa.2, 2.000000e+00
  br i1 %cmp2, label %L84, label %L81

L81:                                              ; preds = %L77.preheader, %L81
  %phi.1 = phi double [ %div, %L81 ], [ %lcssa.1, %L77.preheader ]
  %phi.2 = phi double [ %add, %L81 ], [ %lcssa.2, %L77.preheader ]
  %div = fdiv double %phi.1, %phi.2
  %add = fadd double %phi.2, 1.000000e+00
  %loop2.cmp = fcmp uge double %add, 2.000000e+00
  br i1 %loop2.cmp, label %L84, label %L81

L84:                                              ; preds = %L81, %L77.preheader
  %res = phi double [ %lcssa.1, %L77.preheader ], [ %div, %L81 ]
  ret double %res
}

declare double @__enzyme_autodiff(ptr, double)

define double @test_func(double %x) {
  %res = call double @__enzyme_autodiff(ptr @gamma_like, double %x)
  ret double %res
}

; CHECK: define internal { double } @diffegamma_like

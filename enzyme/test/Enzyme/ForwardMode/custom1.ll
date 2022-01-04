; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s 

source_filename = "exer2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__enzyme_register_derivative_add = dso_local local_unnamed_addr global [2 x i8*] [i8* bitcast (double (double, double)* @add to i8*), i8* bitcast ({ double, double } (double, double, double, double)* @add_err to i8*)], align 16

declare double @add(double %x, double %y) #0

declare { double, double } @add_err(double %v1, double %v1err, double %v2, double %v2err)

; Function Attrs: norecurse nounwind readnone uwtable willreturn
define double @f(double %x) {
entry:
  %call = call double @add(double %x, double %x)
  %mul = fmul double %call, %call
  ret double %mul
}

; Function Attrs: nounwind uwtable
define double @caller(double %x, double %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @f to i8*), double %x, double %dx)
  ret double %call
}

declare dso_local double @__enzyme_fwddiff(i8*, ...)

attributes #0 = { norecurse nounwind readnone }

; CHECK: define internal double @fwddiffef(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double, double } @add_err(double %x, double %"x'", double %x, double %"x'")
; CHECK-NEXT:   %1 = extractvalue { double, double } %0, 0
; CHECK-NEXT:   %2 = extractvalue { double, double } %0, 1
; CHECK-NEXT:   %3 = fmul fast double %2, %1
; CHECK-NEXT:   %4 = fadd fast double %3, %3
; CHECK-NEXT:   ret double %4
; CHECK-NEXT: }

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -preserve-nvvm -enzyme -enzyme-preopt=false -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,enzyme,function(early-cse)" -enzyme-preopt=false -S | FileCheck %s

source_filename = "customfwd_double.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__enzyme_register_derivative_square = dso_local local_unnamed_addr global [2 x i8*] [i8* bitcast (double (double)* @square to i8*), i8* bitcast (double (double, double)* @derivative_square to i8*)], align 16

; Function Attrs: norecurse nounwind readnone uwtable willreturn
define double @square(double %x) #0 {
entry:
  %mul = fmul double %x, %x
  ret double %mul
}

define double @derivative_square(double %x, double %dx) #0 {
entry:
  ret double 100.000000e+00
}

; Function Attrs: nounwind uwtable
define double @caller(double %x, double %dx) {
entry:
  %call = call double (i8*, ...) @__enzyme_fwddiff(i8* bitcast (double (double)* @square to i8*), metadata !"enzyme_dup", double %x, double %dx)
  ret double %call
}

declare dso_local double @__enzyme_fwddiff(i8*, ...)

attributes #0 = { norecurse nounwind readnone }

; CHECK: define internal double @fwddiffesquare(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call double @derivative_square(double %x, double %"x'")
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -early-cse -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

declare { double, double } @__muldc3(double, double, double, double)
declare { double, double } @__enzyme_fwddiff(i8*, ...)

define { double, double } @square(double %xre, double %xim, double %yre, double %yim) {
entry:
  %call = call { double, double } @__muldc3(double %xre, double %xim, double %yre, double %yim)
  ret { double, double } %call
}

define { double, double } @dsquare(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double, double, double, double)* @square to i8*), double %x.coerce0, double %x.coerce1, double 1.000000e+00, double 0.000000e+00, double 1.000000e+00, double 0.000000e+00, double 1.000000e+00, double 0.000000e+00)
  ret { double, double } %call
}


; CHECK: define internal { double, double } @fwddiffesquare(double %xre, double %"xre'", double %xim, double %"xim'", double %yre, double %"yre'", double %yim, double %"yim'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %"xre'", %yre
; CHECK-NEXT:   %1 = fmul fast double %"xim'", %yim
; CHECK-NEXT:   %2 = fsub fast double %0, %1
; CHECK-NEXT:   %3 = fmul fast double %"xre'", %yim
; CHECK-NEXT:   %4 = fmul fast double %yre, %"xim'"
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   %6 = fmul fast double %"yre'", %xre
; CHECK-NEXT:   %7 = fmul fast double %"yim'", %xim
; CHECK-NEXT:   %8 = fsub fast double %6, %7
; CHECK-NEXT:   %9 = fmul fast double %"yre'", %xim
; CHECK-NEXT:   %10 = fmul fast double %xre, %"yim'"
; CHECK-NEXT:   %11 = fadd fast double %9, %10
; CHECK-NEXT:   %12 = fadd fast double %2, %8
; CHECK-NEXT:   %13 = fadd fast double %5, %11
; CHECK-NEXT:   %14 = insertvalue { double, double } undef, double %12, 0
; CHECK-NEXT:   %15 = insertvalue { double, double } %14, double %13, 1
; CHECK-NEXT:   ret { double, double } %15
; CHECK-NEXT: }

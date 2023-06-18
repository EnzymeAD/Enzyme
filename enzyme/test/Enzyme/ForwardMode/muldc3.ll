; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare { double, double } @__muldc3(double, double, double, double) readonly
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
; CHECK-NEXT:   %0 = call { double, double } @__muldc3(double %"xre'", double %"xim'", double %yre, double %yim)
; CHECK-NEXT:   %1 = call { double, double } @__muldc3(double %"yre'", double %"yim'", double %xre, double %xim)
; CHECK-NEXT:   %re1 = extractvalue { double, double } %0, 0
; CHECK-NEXT:   %im1 = extractvalue { double, double } %0, 1
; CHECK-NEXT:   %re2 = extractvalue { double, double } %1, 0
; CHECK-NEXT:   %im2 = extractvalue { double, double } %1, 1
; CHECK-NEXT:   %2 = fadd fast double %re1, %re2
; CHECK-NEXT:   %3 = fadd fast double %im1, %im2
; CHECK-NEXT:   %4 = insertvalue { double, double } undef, double %2, 0
; CHECK-NEXT:   %5 = insertvalue { double, double } %4, double %3, 1
; CHECK-NEXT:   ret { double, double } %5
; CHECK-NEXT: }
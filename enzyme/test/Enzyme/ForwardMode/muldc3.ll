; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

declare { double, double } @__muldc3(double, double, double, double)
declare { double, double } @__enzyme_fwddiff(i8*, ...)

define { double, double } @square(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %x.coerce0, double %x.coerce1)
  ret { double, double } %call
}

define { double, double } @dsquare(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double, double)* @square to i8*), double %x.coerce0, double %x.coerce1, double 1.000000e+00, double 0.000000e+00)
  ret { double, double } %call
}


; CHECK: define internal { double, double } @fwddiffesquare(double %x.coerce0, double %"x.coerce0'", double %x.coerce1, double %"x.coerce1'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double, double } @__muldc3(double %"x.coerce0'", double %"x.coerce1'", double %x.coerce0, double %x.coerce1)
; CHECK-NEXT:   %1 = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %"x.coerce0'", double %"x.coerce1'")
; CHECK-NEXT:   %2 = extractvalue { double, double } %0, 0
; CHECK-NEXT:   %3 = extractvalue { double, double } %1, 0
; CHECK-NEXT:   %4 = fadd fast double %2, %3
; CHECK-NEXT:   %5 = extractvalue { double, double } %0, 1
; CHECK-NEXT:   %6 = extractvalue { double, double } %1, 1
; CHECK-NEXT:   %7 = fadd fast double %5, %6
; CHECK-NEXT:   %8 = insertvalue { double, double } undef, double %4, 0
; CHECK-NEXT:   %9 = insertvalue { double, double } %8, double %7, 1
; CHECK-NEXT:   ret { double, double } %9
; CHECK-NEXT: }
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

declare { double, double } @__muldc3(double, double, double, double)
declare { double, double } @__enzyme_autodiff(i8*, ...)

define { double, double } @square(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %x.coerce0, double %x.coerce1) #2
  ret { double, double } %call
}

define { double, double } @dsquare(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_autodiff(i8* bitcast ({ double, double } (double, double)* @square to i8*), double %x.coerce0, double %x.coerce1) #2
  ret { double, double } %call
}


; CHECK: define internal { double, double } @diffesquare(double %x.coerce0, double %x.coerce1, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %1 = extractvalue { double, double } %differeturn, 1
; CHECK-NEXT:   %2 = call { double, double } @__muldc3(double %0, double %1, double %x.coerce0, double %x.coerce1)
; CHECK-NEXT:   %3 = call { double, double } @__muldc3(double %x.coerce0, double %x.coerce1, double %0, double %1)
; CHECK-NEXT:   %4 = extractvalue { double, double } %2, 0
; CHECK-NEXT:   %5 = extractvalue { double, double } %2, 1
; CHECK-NEXT:   %6 = extractvalue { double, double } %3, 0
; CHECK-NEXT:   %7 = fadd fast double %4, %6
; CHECK-NEXT:   %8 = extractvalue { double, double } %3, 1
; CHECK-NEXT:   %9 = fadd fast double %5, %8
; CHECK-NEXT:   %10 = insertvalue { double, double } undef, double %7, 0
; CHECK-NEXT:   %11 = insertvalue { double, double } %10, double %9, 1
; CHECK-NEXT:   ret { double, double } %11
; CHECK-NEXT: }
; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define { double, double } @test(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double } @__divdc3(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
  ret { double, double } %call
}

declare { double, double } @__divdc3(double, double, double, double)

define { double, double, double, double} @dtest(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double, double, double } (i8*, ...) @__enzyme_autodiff(i8* bitcast ({ double, double } (double, double, double, double)* @test to i8*), double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
  ret { double, double, double, double} %call
}

declare { double, double, double, double } @__enzyme_autodiff(i8*, ...)


; CHECK: define internal { double, double, double, double } @diffetest(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1, { double, double } %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call = call { double, double } @__divdc3(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) #0
; CHECK-NEXT:   %0 = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %1 = extractvalue { double, double } %differeturn, 1
; CHECK-NEXT:   %2 = call { double, double } @__divdc3(double %0, double %1, double %y.coerce0, double %y.coerce1)
; CHECK-NEXT:   %3 = call { double, double } @__divdc3(double %0, double %1, double %x.coerce1, double %y.coerce0)
; CHECK-NEXT:   %4 = extractvalue { double, double } %call, 0
; CHECK-NEXT:   %5 = fneg fast double %4
; CHECK-NEXT:   %6 = extractvalue { double, double } %call, 1
; CHECK-NEXT:   %7 = fneg fast double %6
; CHECK-NEXT:   %8 = extractvalue { double, double } %3, 0
; CHECK-NEXT:   %9 = extractvalue { double, double } %3, 1
; CHECK-NEXT:   %10 = call { double, double } @__muldc3(double %5, double %7, double %8, double %9)
; CHECK-NEXT:   %11 = extractvalue { double, double } %2, 0
; CHECK-NEXT:   %12 = extractvalue { double, double } %2, 1
; CHECK-NEXT:   %13 = extractvalue { double, double } %10, 0
; CHECK-NEXT:   %14 = extractvalue { double, double } %10, 1
; CHECK-NEXT:   %15 = insertvalue { double, double, double, double } undef, double %11, 0
; CHECK-NEXT:   %16 = insertvalue { double, double, double, double } %15, double %12, 1
; CHECK-NEXT:   %17 = insertvalue { double, double, double, double } %16, double %13, 2
; CHECK-NEXT:   %18 = insertvalue { double, double, double, double } %17, double %14, 3
; CHECK-NEXT:   ret { double, double, double, double } %18
; CHECK-NEXT: }
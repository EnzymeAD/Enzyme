; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -early-cse -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

declare dso_local { double, double } @__divdc3(double, double, double, double)
declare { double, double } @__enzyme_fwddiff(i8*, ...)

define { double, double } @tester(double %xre, double %xim, double %yre, double %yim) {
entry:
  %call = call { double, double } @__divdc3(double %xre, double %xim, double %yre, double %yim)
  ret { double, double } %call
}

define dso_local { double, double } @test_derivative(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double, double, double, double)* @tester to i8*), double %x.coerce0, double %x.coerce1, double 1.000000e+00, double 0.000000e+00, double %y.coerce0, double %y.coerce1, double 1.000000e+00, double 0.000000e+00)
  ret { double, double } %call
}


; CHECK: define internal { double, double } @fwddiffetester(double %xre, double %"xre'", double %xim, double %"xim'", double %yre, double %"yre'", double %yim, double %"yim'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fmul fast double %"xre'", %yre
; CHECK-NEXT:   %1 = fmul fast double %"xim'", %yim
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   %3 = fmul fast double %yre, %yre
; CHECK-NEXT:   %4 = fmul fast double %yim, %yim
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   %6 = fdiv fast double %2, %5
; CHECK-NEXT:   %7 = fmul fast double %yre, %"xim'"
; CHECK-NEXT:   %8 = fmul fast double %"xre'", %yim
; CHECK-NEXT:   %9 = fsub fast double %7, %8
; CHECK-NEXT:   %10 = fdiv fast double %9, %5
; CHECK-NEXT:   %11 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %xre
; CHECK-NEXT:   %12 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %xim
; CHECK-NEXT:   %13 = fsub fast double %3, %4
; CHECK-NEXT:   %14 = fmul fast double %yre, %yim
; CHECK-NEXT:   %15 = fadd fast double %14, %14
; CHECK-NEXT:   %16 = fmul fast double %11, %13
; CHECK-NEXT:   %17 = fmul fast double %12, %15
; CHECK-NEXT:   %18 = fadd fast double %16, %17
; CHECK-NEXT:   %19 = fmul fast double %13, %13
; CHECK-NEXT:   %20 = fmul fast double %15, %15
; CHECK-NEXT:   %21 = fadd fast double %19, %20
; CHECK-NEXT:   %22 = fdiv fast double %18, %21
; CHECK-NEXT:   %23 = fmul fast double %"yre'", %22
; CHECK-NEXT:   %24 = fmul fast double %"yim'", %15
; CHECK-NEXT:   %25 = fsub fast double %23, %24
; CHECK-NEXT:   %26 = fmul fast double %"yre'", %15
; CHECK-NEXT:   %27 = fmul fast double %22, %"yim'"
; CHECK-NEXT:   %28 = fadd fast double %26, %27
; CHECK-NEXT:   %29 = fadd fast double %6, %25
; CHECK-NEXT:   %30 = fadd fast double %10, %28
; CHECK-NEXT:   %31 = insertvalue { double, double } undef, double %29, 0
; CHECK-NEXT:   %32 = insertvalue { double, double } %31, double %30, 1
; CHECK-NEXT:   ret { double, double } %32
; CHECK-NEXT: }
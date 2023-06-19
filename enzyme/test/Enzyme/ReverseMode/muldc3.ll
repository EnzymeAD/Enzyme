; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -early-cse -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,early-cse,instsimplify)" -S | FileCheck %s

declare { double, double } @__muldc3(double, double, double, double)
declare { double, double, double, double } @__enzyme_autodiff(...)


define { double, double } @square(double %xre, double %xim, double %yre, double %yim) {
entry:
  %call = call { double, double } @__muldc3(double %xre, double %xim, double %yre, double %yim)
  ret { double, double } %call
}

define { double, double, double, double } @dsquare(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double, double, double } (...) @__enzyme_autodiff({ double, double } (double, double, double, double)* @square, double %x.coerce0, double %x.coerce1, double 1.000000e+00, double 0.000000e+00)
  ret { double, double, double, double } %call
}

; CHECK: define internal { double, double, double, double } @diffesquare(double %xre, double %xim, double %yre, double %yim, { double, double } %differeturn) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %re1 = extractvalue { double, double } %differeturn, 0
; CHECK-NEXT:   %im1 = extractvalue { double, double } %differeturn, 1
; CHECK-NEXT:   %0 = fmul fast double %re1, %yre
; CHECK-NEXT:   %1 = fmul fast double %im1, %yim
; CHECK-NEXT:   %2 = fsub fast double %0, %1
; CHECK-NEXT:   %3 = fmul fast double %re1, %yim
; CHECK-NEXT:   %4 = fmul fast double %yre, %im1
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   %6 = fadd fast double %2, %5
; CHECK-NEXT:   %7 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %yre
; CHECK-NEXT:   %8 = fmul fast double %im1, %7
; CHECK-NEXT:   %9 = fsub fast double %3, %8
; CHECK-NEXT:   %10 = fmul fast double %re1, %7
; CHECK-NEXT:   %11 = fadd fast double %10, %1
; CHECK-NEXT:   %12 = fadd fast double %9, %11
; CHECK-NEXT:   %13 = fmul fast double %re1, %xre
; CHECK-NEXT:   %14 = fmul fast double %im1, %xim
; CHECK-NEXT:   %15 = fsub fast double %13, %14
; CHECK-NEXT:   %16 = fmul fast double %re1, %xim
; CHECK-NEXT:   %17 = fmul fast double %xre, %im1
; CHECK-NEXT:   %18 = fadd fast double %16, %17
; CHECK-NEXT:   %19 = fadd fast double %15, %18
; CHECK-NEXT:   %20 = {{(fneg fast double)|(fsub fast double \-0.000000e\+00,)}} %xre
; CHECK-NEXT:   %21 = fmul fast double %im1, %20
; CHECK-NEXT:   %22 = fsub fast double %16, %21
; CHECK-NEXT:   %23 = fmul fast double %re1, %20
; CHECK-NEXT:   %24 = fadd fast double %23, %14
; CHECK-NEXT:   %25 = fadd fast double %22, %24
; CHECK-NEXT:   %26 = insertvalue { double, double, double, double } {{(undef|poison)}}, double %6, 0
; CHECK-NEXT:   %27 = insertvalue { double, double, double, double } %26, double %12, 1
; CHECK-NEXT:   %28 = insertvalue { double, double, double, double } %27, double %19, 2
; CHECK-NEXT:   %29 = insertvalue { double, double, double, double } %28, double %25, 3
; CHECK-NEXT:   ret { double, double, double, double } %29
; CHECK-NEXT: }
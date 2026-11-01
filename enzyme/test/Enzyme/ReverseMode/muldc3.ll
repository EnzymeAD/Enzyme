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
; CHECK-NEXT:   %0 = fmul double %re1, %yre
; CHECK-NEXT:   %1 = fmul double %im1, %yim
; CHECK-NEXT:   %2 = fsub double %0, %1
; CHECK-NEXT:   %3 = fmul double %re1, %yim
; CHECK-NEXT:   %4 = fmul double %yre, %im1
; CHECK-NEXT:   %5 = fadd double %3, %4
; CHECK-NEXT:   %6 = fadd double %2, %5
; CHECK-NEXT:   %7 = fadd double 0.000000e+00, %6
; CHECK-NEXT:   %8 = {{(fneg double)|(fsub double \-0.000000e\+00,)}} %yre
; CHECK-NEXT:   %9 = fmul double %im1, %8
; CHECK-NEXT:   %10 = fsub double %3, %9
; CHECK-NEXT:   %11 = fmul double %re1, %8
; CHECK-NEXT:   %12 = fadd double %11, %1
; CHECK-NEXT:   %13 = fadd double %10, %12
; CHECK-NEXT:   %14 = fadd double 0.000000e+00, %13
; CHECK-NEXT:   %15 = fmul double %re1, %xre
; CHECK-NEXT:   %16 = fmul double %im1, %xim
; CHECK-NEXT:   %17 = fsub double %15, %16
; CHECK-NEXT:   %18 = fmul double %re1, %xim
; CHECK-NEXT:   %19 = fmul double %xre, %im1
; CHECK-NEXT:   %20 = fadd double %18, %19
; CHECK-NEXT:   %21 = fadd double %17, %20
; CHECK-NEXT:   %22 = fadd double 0.000000e+00, %21
; CHECK-NEXT:   %23 = {{(fneg double)|(fsub double \-0.000000e\+00,)}} %xre
; CHECK-NEXT:   %24 = fmul double %im1, %23
; CHECK-NEXT:   %25 = fsub double %18, %24
; CHECK-NEXT:   %26 = fmul double %re1, %23
; CHECK-NEXT:   %27 = fadd double %26, %16
; CHECK-NEXT:   %28 = fadd double %25, %27
; CHECK-NEXT:   %29 = fadd double 0.000000e+00, %28
; CHECK-NEXT:   %30 = insertvalue { double, double, double, double } {{(undef|poison)}}, double %7, 0
; CHECK-NEXT:   %31 = insertvalue { double, double, double, double } %30, double %14, 1
; CHECK-NEXT:   %32 = insertvalue { double, double, double, double } %31, double %22, 2
; CHECK-NEXT:   %33 = insertvalue { double, double, double, double } %32, double %29, 3
; CHECK-NEXT:   ret { double, double, double, double } %33
; CHECK-NEXT: }
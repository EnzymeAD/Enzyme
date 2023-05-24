; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_fwddiff(...)

define double @square(double %arg) {

  ; arg, 1
  %i8 = insertelement <2 x double> <double undef, double 1.000000e+00>, double %arg, i32 0
  
  ; arg
  %i14 = extractelement <2 x double> %i8, i32 1
  
  ret double %i14
}

define double @dsquare(double %x, double %dx) {
  %res = tail call double (...) @__enzyme_fwddiff(double (double)* nonnull @square, double 1.000000e+00, double 1.000000e+00)
  ret double %res
}

; CHECK: define internal double @fwddiffesquare(double %arg, double %"arg'")
; CHECK-NEXT:   %"i8'ipie" = insertelement <2 x double> zeroinitializer, double %"arg'", i32 0
; CHECK-NEXT:   %"i14'ipee" = extractelement <2 x double> %"i8'ipie", i32 1
; CHECK-NEXT:   ret double %"i14'ipee"
; CHECK-NEXT: }

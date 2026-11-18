; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -sroa -instsimplify -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(sroa,instsimplify,instcombine,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_autodiff(ptr, ...)
declare <2 x double> @llvm.fma.v2f64(<2 x double>, <2 x double>, <2 x double>)

define double @square(double %arg) {
  %i8 = insertelement <2 x double> <double poison, double 1.000000e+00>, double %arg, i32 0
  %i12 = fmul <2 x double> <double undef, double 1.000000e+00>, %i8
  %i14 = extractelement <2 x double> %i12, i32 1
  ret double %i14
}

define double @div_test(double %arg) {
  %i8 = insertelement <2 x double> <double poison, double 1.000000e+00>, double %arg, i32 0
  %i12 = fdiv <2 x double> %i8, <double undef, double 2.000000e+00>
  %i14 = extractelement <2 x double> %i12, i32 1
  ret double %i14
}

define double @div_test2(double %arg) {
  %i8 = insertelement <2 x double> <double poison, double 1.000000e+00>, double %arg, i32 0
  %i12 = fdiv <2 x double> <double undef, double 2.000000e+00>, %i8
  %i14 = extractelement <2 x double> %i12, i32 1
  ret double %i14
}

define double @fma_test(double %arg) {
  %i8 = insertelement <2 x double> <double poison, double 1.000000e+00>, double %arg, i32 0
  %i12 = call <2 x double> @llvm.fma.v2f64(<2 x double> <double undef, double 1.000000e+00>, <2 x double> %i8, <2 x double> zeroinitializer)
  %i14 = extractelement <2 x double> %i12, i32 1
  ret double %i14
}

define double @caller(double %x) {
  %r1 = call double (ptr, ...) @__enzyme_autodiff(ptr @square, double %x)
  %r2 = call double (ptr, ...) @__enzyme_autodiff(ptr @div_test, double %x)
  %r3 = call double (ptr, ...) @__enzyme_autodiff(ptr @div_test2, double %x)
  %r4 = call double (ptr, ...) @__enzyme_autodiff(ptr @fma_test, double %x)
  %s1 = fadd double %r1, %r2
  %s2 = fadd double %s1, %r3
  %s3 = fadd double %s2, %r4
  ret double %s3
}

; CHECK: define internal { double } @diffesquare(double %arg, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }

; CHECK: define internal { double } @diffediv_test(double %arg, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }

; CHECK: define internal { double } @diffediv_test2(double %arg, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   ret { double } { double -0.000000e+00 }
; CHECK-NEXT: }

; CHECK: define internal { double } @diffefma_test(double %arg, double %differeturn)
; CHECK-NEXT: invert:
; CHECK-NEXT:   ret { double } zeroinitializer
; CHECK-NEXT: }

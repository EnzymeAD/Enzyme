; RUN: if [ %llvmver -ge 12 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instcombine,%simplifycfg)" -S | FileCheck %s; fi

declare double @llvm.vector.reduce.fmax.v2f64(<4 x double> %v)

; Function Attrs: nounwind readnone uwtable
define double @tester(<4 x double> %v) {
entry:
  %r = tail call double @llvm.vector.reduce.fmax.v2f64(<4 x double> %v)
  ret double %r
}

define <4 x double> @test_derivative(<4 x double> %x) {
entry:
  %r = tail call <4 x double> (...) @__enzyme_autodiff(double (<4 x double>)* nonnull @tester, <4 x double> %x)
  ret <4 x double> %r
}

declare <4 x double> @__enzyme_autodiff(...)

; CHECK: define internal { <4 x double> } @diffetester(<4 x double> %v, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractelement <4 x double> %v, i64 0
; CHECK-NEXT:   %1 = extractelement <4 x double> %v, i64 1
; CHECK-NEXT:   %2 = extractelement <4 x double> %v, i64 2
; CHECK-NEXT:   %3 = extractelement <4 x double> %v, i64 3
; CHECK-NEXT:   %4 = fcmp fast olt double %0, %1
; CHECK-NEXT:   %5 = select fast i1 %4, double %1, double %0
; CHECK-NEXT:   %6 = fcmp fast olt double %5, %2
; CHECK-NEXT:   %7 = select fast i1 %6, double %2, double %5
; CHECK-NEXT:   %8 = fcmp fast olt double %7, %3
; CHECK-NEXT:   %9 = insertelement <4 x double> <double {{(poison|undef)}}, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00>, double %differeturn, i64 0
; CHECK-NEXT:   %10 = insertelement <4 x double> <double 0.000000e+00, double {{(poison|undef)}}, double 0.000000e+00, double 0.000000e+00>, double %differeturn, i64 1
; CHECK-NEXT:   %11 = select fast i1 %4, <4 x double> %10, <4 x double> %9
; CHECK-NEXT:   %12 = insertelement <4 x double> <double 0.000000e+00, double 0.000000e+00, double {{(poison|undef)}}, double 0.000000e+00>, double %differeturn, i64 2
; CHECK-NEXT:   %13 = select fast i1 %6, <4 x double> %12, <4 x double> %11
; CHECK-NEXT:   %14 = insertelement <4 x double> <double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double {{(poison|undef)}}>, double %differeturn, i64 3
; CHECK-NEXT:   %15 = select fast i1 %8, <4 x double> %14, <4 x double> %13
; CHECK-NEXT:   %16 = insertvalue { <4 x double> } undef, <4 x double> %15, 0
; CHECK-NEXT:   ret { <4 x double> } %16
; CHECK-NEXT: }

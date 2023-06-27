; RUN: if [ %llvmver -ge 12 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instcombine,%simplifycfg)" -S | FileCheck %s; fi

declare double @llvm.vector.reduce.fmax.v2f64(<2 x double> %v)

; Function Attrs: nounwind readnone uwtable
define double @tester(<2 x double> %v) {
entry:
  %r = tail call double @llvm.vector.reduce.fmax.v2f64(<2 x double> %v)
  ret double %r
}

define <2 x double> @test_derivative(<2 x double> %x) {
entry:
  %r = tail call <2 x double> (...) @__enzyme_autodiff(double (<2 x double>)* nonnull @tester, <2 x double> %x)
  ret <2 x double> %r
}

declare <2 x double> @__enzyme_autodiff(...)

; CHECK: define internal { <2 x double> } @diffetester(<2 x double> %v, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractelement <2 x double> %v, i64 0
; CHECK-NEXT:   %1 = extractelement <2 x double> %v, i64 1
; CHECK-NEXT:   %2 = fcmp fast olt double %0, %1
; CHECK-NEXT:   %3 = insertelement <2 x double> <double {{(poison|undef)}}, double 0.000000e+00>, double %differeturn, i64 0
; CHECK-NEXT:   %4 = insertelement <2 x double> <double 0.000000e+00, double {{(poison|undef)}}>, double %differeturn, i64 1
; CHECK-NEXT:   %5 = select fast i1 %2, <2 x double> %4, <2 x double> %3
; CHECK-NEXT:   %6 = insertvalue { <2 x double> } undef, <2 x double> %5, 0
; CHECK-NEXT:   ret { <2 x double> } %6
; CHECK-NEXT: }

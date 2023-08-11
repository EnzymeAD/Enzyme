; RUN: if [ %llvmver -ge 12 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare double @llvm.vector.reduce.fmax.v2f64(<4 x double> %v)

; Function Attrs: nounwind readnone uwtable
define double @tester(<4 x double> %v) {
entry:
  %r = tail call double @llvm.vector.reduce.fmax.v2f64(<4 x double> %v)
  ret double %r
}

define double @test_derivative(<4 x double> %x, <4 x double> %dx) {
entry:
  %r = tail call double (...) @__enzyme_fwddiff(double (<4 x double>)* nonnull @tester, <4 x double> %x, <4 x double> %dx)
  ret double %r
}

declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffetester(<4 x double> %v, <4 x double> %"v'")
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
; CHECK-NEXT:   %9 = extractelement <4 x double> %"v'", i64 0
; CHECK-NEXT:   %10 = extractelement <4 x double> %"v'", i64 1
; CHECK-NEXT:   %11 = select fast i1 %4, double %10, double %9
; CHECK-NEXT:   %12 = extractelement <4 x double> %"v'", i64 2
; CHECK-NEXT:   %13 = select fast i1 %6, double %12, double %11
; CHECK-NEXT:   %14 = extractelement <4 x double> %"v'", i64 3
; CHECK-NEXT:   %15 = select fast i1 %8, double %14, double %13
; CHECK-NEXT:   ret double %15
; CHECK-NEXT: }

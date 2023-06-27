; RUN: if [ %llvmver -ge 12 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

declare double @llvm.vector.reduce.fmax.v2f64(<2 x double> %v)

; Function Attrs: nounwind readnone uwtable
define double @tester(<2 x double> %v) {
entry:
  %r = tail call double @llvm.vector.reduce.fmax.v2f64(<2 x double> %v)
  ret double %r
}

define double @test_derivative(<2 x double> %x, <2 x double> %dx) {
entry:
  %r = tail call double (...) @__enzyme_fwddiff(double (<2 x double>)* nonnull @tester, <2 x double> %x, <2 x double> %dx)
  ret double %r
}

declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffetester(<2 x double> %v, <2 x double> %"v'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractelement <2 x double> %v, i64 0
; CHECK-NEXT:   %1 = extractelement <2 x double> %v, i64 1
; CHECK-NEXT:   %2 = fcmp fast olt double %0, %1
; CHECK-NEXT:   %3 = extractelement <2 x double> %"v'", i64 0
; CHECK-NEXT:   %4 = extractelement <2 x double> %"v'", i64 1
; CHECK-NEXT:   %5 = select fast i1 %2, double %4, double %3
; CHECK-NEXT:   ret double %5
; CHECK-NEXT: }

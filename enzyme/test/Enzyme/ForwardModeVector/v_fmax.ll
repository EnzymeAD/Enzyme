; RUN: if [ %llvmver -ge 12 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

%struct.Gradients = type { double, double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(...)

declare double @llvm.vector.reduce.fmax.v2f64(<2 x double> %v)

; Function Attrs: nounwind readnone uwtable
define double @tester(<2 x double> %v) {
entry:
  %r = tail call double @llvm.vector.reduce.fmax.v2f64(<2 x double> %v)
  ret double %r
}

define %struct.Gradients @test_derivative(<2 x double> %x, <2 x double> %dx1, <2 x double> %dx2, <2 x double> %dx3) {
entry:
  %r = tail call %struct.Gradients (...) @__enzyme_fwddiff(double (<2 x double>)* nonnull @tester, metadata !"enzyme_width", i64 3, <2 x double> %x, <2 x double> %dx1, <2 x double> %dx2, <2 x double> %dx3)
  ret %struct.Gradients %r
}

; CHECK: define internal [3 x double] @fwddiffe3tester(<2 x double> %v, [3 x <2 x double>] %"v'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractelement <2 x double> %v, i64 0
; CHECK-NEXT:   %1 = extractelement <2 x double> %v, i64 1
; CHECK-NEXT:   %2 = fcmp fast olt double %0, %1
; CHECK-NEXT:   %3 = extractvalue [3 x <2 x double>] %"v'", 0
; CHECK-NEXT:   %4 = extractelement <2 x double> %3, i64 0
; CHECK-NEXT:   %5 = extractelement <2 x double> %3, i64 1
; CHECK-NEXT:   %6 = select fast i1 %2, double %5, double %4
; CHECK-NEXT:   %7 = insertvalue [3 x double] {{(undef|poison)}}, double %6, 0
; CHECK-NEXT:   %8 = extractvalue [3 x <2 x double>] %"v'", 1
; CHECK-NEXT:   %9 = extractelement <2 x double> %8, i64 0
; CHECK-NEXT:   %10 = extractelement <2 x double> %8, i64 1
; CHECK-NEXT:   %11 = select fast i1 %2, double %10, double %9
; CHECK-NEXT:   %12 = insertvalue [3 x double] %7, double %11, 1
; CHECK-NEXT:   %13 = extractvalue [3 x <2 x double>] %"v'", 2
; CHECK-NEXT:   %14 = extractelement <2 x double> %13, i64 0
; CHECK-NEXT:   %15 = extractelement <2 x double> %13, i64 1
; CHECK-NEXT:   %16 = select fast i1 %2, double %15, double %14
; CHECK-NEXT:   %17 = insertvalue [3 x double] %12, double %16, 2
; CHECK-NEXT:   ret [3 x double] %17
; CHECK-NEXT: }

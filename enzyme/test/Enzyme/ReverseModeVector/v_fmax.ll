; RUN: if [ %llvmver -ge 12 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -adce -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 12 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s; fi

declare double @llvm.vector.reduce.fmax.v2f64(<2 x double> %v)

; Function Attrs: nounwind readnone uwtable
define double @tester(<2 x double> %v) {
entry:
  %r = tail call double @llvm.vector.reduce.fmax.v2f64(<2 x double> %v)
  ret double %r
}

define [3 x <2 x double>] @test_derivative(<2 x double> %x) {
entry:
  %r = tail call [3 x <2 x double>] (...) @__enzyme_autodiff(double (<2 x double>)* nonnull @tester, metadata !"enzyme_width", i64 3, <2 x double> %x)
  ret [3 x <2 x double>] %r
}

declare [3 x <2 x double>] @__enzyme_autodiff(...)

; CHECK: define internal { [3 x <2 x double>] } @diffe3tester(<2 x double> %v, [3 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractelement <2 x double> %v, i64 0
; CHECK-NEXT:   %1 = extractelement <2 x double> %v, i64 1
; CHECK-NEXT:   %2 = fcmp fast olt double %0, %1
; CHECK-NEXT:   %3 = extractvalue [3 x double] %differeturn, 0
; CHECK-NEXT:   %4 = insertelement <2 x double> zeroinitializer, double %3, i64 0
; CHECK-NEXT:   %5 = insertelement <2 x double> zeroinitializer, double %3, i64 1
; CHECK-NEXT:   %6 = select fast i1 %2, <2 x double> %5, <2 x double> %4
; CHECK-NEXT:   %7 = extractvalue [3 x double] %differeturn, 1
; CHECK-NEXT:   %8 = insertelement <2 x double> zeroinitializer, double %7, i64 0
; CHECK-NEXT:   %9 = insertelement <2 x double> zeroinitializer, double %7, i64 1
; CHECK-NEXT:   %10 = select fast i1 %2, <2 x double> %9, <2 x double> %8
; CHECK-NEXT:   %11 = extractvalue [3 x double] %differeturn, 2
; CHECK-NEXT:   %12 = insertelement <2 x double> zeroinitializer, double %11, i64 0
; CHECK-NEXT:   %13 = insertelement <2 x double> zeroinitializer, double %11, i64 1
; CHECK-NEXT:   %14 = select fast i1 %2, <2 x double> %13, <2 x double> %12
; CHECK-NEXT:   %15 = fadd fast <2 x double> zeroinitializer, %6
; CHECK-NEXT:   %16 = fadd fast <2 x double> zeroinitializer, %10
; CHECK-NEXT:   %17 = fadd fast <2 x double> zeroinitializer, %14
; CHECK-NEXT:   %.fca.0.insert6 = insertvalue [3 x <2 x double>] {{(undef|poison)}}, <2 x double> %15, 0
; CHECK-NEXT:   %.fca.1.insert9 = insertvalue [3 x <2 x double>] %.fca.0.insert6, <2 x double> %16, 1
; CHECK-NEXT:   %.fca.2.insert12 = insertvalue [3 x <2 x double>] %.fca.1.insert9, <2 x double> %17, 2
; CHECK-NEXT:   %18 = insertvalue { [3 x <2 x double>] } {{(undef|poison)}}, [3 x <2 x double>] %.fca.2.insert12, 0
; CHECK-NEXT:   ret { [3 x <2 x double>] } %18
; CHECK-NEXT: }

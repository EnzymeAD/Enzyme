; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify,adce)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone willreturn
declare double @cabs([2 x double])

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %agg0 = insertvalue [2 x double] undef, double %x, 0
  %agg1 = insertvalue [2 x double] %agg0, double %y, 1
  %call = call double @cabs([2 x double] %agg1)
  ret double %call
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_const", double %x, double %y, double 1.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal double @fwddiffetester(double %x, double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg0 = insertvalue [2 x double] undef, double %x, 0
; CHECK-NEXT:   %agg1 = insertvalue [2 x double] %agg0, double %y, 1
; CHECK-NEXT:   %0 = fmul fast double %"y'", %y
; CHECK-NEXT:   %1 = call fast double @cabs([2 x double] %agg1) #0
; CHECK-NEXT:   %2 = fdiv fast double %0, %1
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }

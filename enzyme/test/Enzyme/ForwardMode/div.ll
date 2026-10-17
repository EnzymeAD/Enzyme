; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fdiv fast double %x, %y
  ret double %0
}

define double @test_derivative(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, double %x, double 1.0, double %y, double 0.0)
  ret double %0
}

; Function Attrs: nounwind
declare double @__enzyme_fwddiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}double @fwddiffetester(double %x, double %"x'", double %y, double %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = fdiv nnan ninf nsz arcp contract afn double %x, %y
; CHECK-NEXT:   %1 = fmul nnan ninf nsz arcp contract afn double %"y'", %0
; CHECK-NEXT:   %2 = fsub nnan ninf nsz arcp contract afn double %"x'", %1
; CHECK-NEXT:   %3 = fdiv nnan ninf nsz arcp contract afn double %2, %y
; CHECK-NEXT:   ret double %3
; CHECK-NEXT: }

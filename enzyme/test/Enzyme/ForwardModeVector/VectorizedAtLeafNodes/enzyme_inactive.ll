; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

; Function Attrs: nounwind
declare <3 x double> @__enzyme_fwddiff(double (double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x) {
entry:
  tail call void @myprint(double %x)
  ret double %x
}

define <3 x double> @test_derivative(double %x) {
entry:
  %0 = tail call <3 x double> (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, <3 x double> <double 0.0, double 1.0, double 2.0>)
  ret <3 x double> %0
}

declare void @myprint(double %x) #0

attributes #0 = { "enzyme_inactive" }


; CHECK: define internal <3 x double> @fwddiffe3tester(double %x, <3 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @myprint(double %x) #1
; CHECK-NEXT:   ret <3 x double> %"x'"
; CHECK-NEXT: }
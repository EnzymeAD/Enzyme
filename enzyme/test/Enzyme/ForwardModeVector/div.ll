; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s

%struct.Gradients = type { double, double }

; Function Attrs: nounwind
declare %struct.Gradients @__enzyme_fwddiff(double (double,double)*, ...)

; Function Attrs: noinline nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %0 = fdiv fast double %x, %y
  ret double %0
}

define %struct.Gradients @test_derivative(double %x, double %y) {
entry:
  %0 = tail call %struct.Gradients (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 2, double %x, double 0.0, double 1.0, double %y, double 1.0, double 0.0)
  ret %struct.Gradients %0
}


; CHECK: define internal [2 x double] @fwddiffe2tester(double %x, [2 x double] %"x'", double %y, [2 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[q:.+]] = fdiv fast double %x, %y
; CHECK-NEXT:   %[[y0:.+]] = extractvalue [2 x double] %"y'", 0
; CHECK-NEXT:   %[[m0:.+]] = fmul fast double %[[y0]], %[[q]]
; CHECK-NEXT:   %[[y1:.+]] = extractvalue [2 x double] %"y'", 1
; CHECK-NEXT:   %[[m1:.+]] = fmul fast double %[[y1]], %[[q]]
; CHECK-NEXT:   %[[x0:.+]] = extractvalue [2 x double] %"x'", 0
; CHECK-NEXT:   %[[s0:.+]] = fsub fast double %[[x0]], %[[m0]]
; CHECK-NEXT:   %[[x1:.+]] = extractvalue [2 x double] %"x'", 1
; CHECK-NEXT:   %[[s1:.+]] = fsub fast double %[[x1]], %[[m1]]
; CHECK-NEXT:   %[[d0:.+]] = fdiv fast double %[[s0]], %y
; CHECK-NEXT:   %[[r0:.+]] = insertvalue [2 x double] undef, double %[[d0]], 0
; CHECK-NEXT:   %[[d1:.+]] = fdiv fast double %[[s1]], %y
; CHECK-NEXT:   %[[r1:.+]] = insertvalue [2 x double] %[[r0]], double %[[d1]], 1
; CHECK-NEXT:   ret [2 x double] %[[r1]]
; CHECK-NEXT: }

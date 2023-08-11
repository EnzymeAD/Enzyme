; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -instsimplify -simplifycfg -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind readnone willreturn
declare double @cabs([2 x double]) #7

; Function Attrs: nounwind readnone uwtable
define double @tester(double %x, double %y) {
entry:
  %agg0 = insertvalue [2 x double] undef, double %x, 0
  %agg1 = insertvalue [2 x double] %agg0, double %y, 1
  %call = call double @cabs([2 x double] %agg1)
  ret double %call
}

define [3 x double] @test_derivative(double %x, double %y) {
entry:
  %0 = tail call [3 x double] (double (double, double)*, ...) @__enzyme_fwddiff(double (double, double)* nonnull @tester, metadata !"enzyme_width", i64 3, double %x, double 1.0, double 1.3, double 2.0, double %y, double 1.0, double 0.0, double 2.0)
  ret [3 x double] %0
}

; Function Attrs: nounwind
declare [3 x double] @__enzyme_fwddiff(double (double, double)*, ...)


; CHECK: define internal [3 x double] @fwddiffe3tester(double %x, [3 x double] %"x'", double %y, [3 x double] %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %agg0 = insertvalue [2 x double] undef, double %x, 0
; CHECK-NEXT:   %agg1 = insertvalue [2 x double] %agg0, double %y, 1
; CHECK-NEXT:   %[[a0:.+]] = extractvalue [3 x double] %"x'", 0
; CHECK-NEXT:   %[[b0:.+]] = fmul fast double %[[a0]], %x
; CHECK-NEXT:   %[[a1:.+]] = extractvalue [3 x double] %"x'", 1
; CHECK-NEXT:   %[[b1:.+]] = fmul fast double %[[a1]], %x
; CHECK-NEXT:   %[[a2:.+]] = extractvalue [3 x double] %"x'", 2
; CHECK-NEXT:   %[[b2:.+]] = fmul fast double %[[a2]], %x
; CHECK-NEXT:   %[[a3:.+]] = extractvalue [3 x double] %"y'", 0
; CHECK-NEXT:   %[[b3:.+]] = fmul fast double %[[a3]], %y
; CHECK-NEXT:   %[[a4:.+]] = extractvalue [3 x double] %"y'", 1
; CHECK-NEXT:   %[[b4:.+]] = fmul fast double %[[a4]], %y
; CHECK-NEXT:   %[[a5:.+]] = extractvalue [3 x double] %"y'", 2
; CHECK-NEXT:   %[[b5:.+]] = fmul fast double %[[a5]], %y

; CHECK-NEXT:   %[[c0:.+]] = fadd fast double %[[b0]], %[[b3]]
; CHECK-NEXT:   %[[c1:.+]] = fadd fast double %[[b1]], %[[b4]]
; CHECK-NEXT:   %[[c2:.+]] = fadd fast double %[[b2]], %[[b5]]

; CHECK-NEXT:   %[[a6:.+]] = call fast double @cabs([2 x double] %agg1)

; CHECK-NEXT:   %[[r0:.+]] = fdiv fast double %[[c0]], %[[a6]]
; CHECK-NEXT:   %[[a12:.+]] = insertvalue [3 x double] undef, double %[[r0]], 0
; CHECK-NEXT:   %[[r1:.+]] = fdiv fast double %[[c1]], %[[a6]]
; CHECK-NEXT:   %[[a18:.+]] = insertvalue [3 x double] %[[a12]], double %[[r1]], 1
; CHECK-NEXT:   %[[r2:.+]] = fdiv fast double %[[c2]], %[[a6]]
; CHECK-NEXT:   %[[a24:.+]] = insertvalue [3 x double] %[[a18]], double %[[r2]], 2
; CHECK-NEXT:   ret [3 x double] %[[a24]]
; CHECK-NEXT: }
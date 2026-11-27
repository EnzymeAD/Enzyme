; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

%struct.Gradients = type { double, double, double, double }

define <2 x double> @tester(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %absx = and <2 x i64> %cstx, <i64 9223372036854775807, i64 9223372036854775807>
  %csty = bitcast <2 x i64> %absx to <2 x double>
  ret <2 x double> %csty
}

define %struct.Gradients @test_derivative(<2 x double> %x, <2 x double> %dx1, <2 x double> %dx2) {
entry:
  %0 = tail call %struct.Gradients (<2 x double> (<2 x double>)*, ...) @__enzyme_fwddiff(<2 x double> (<2 x double>)* nonnull @tester, metadata !"enzyme_width", i64 2, <2 x double> %x, <2 x double> %dx1, <2 x double> %dx2)
  ret %struct.Gradients %0
}

declare %struct.Gradients @__enzyme_fwddiff(<2 x double> (<2 x double>)*, ...)

; CHECK: define internal [2 x <2 x double>] @fwddiffe2tester(<2 x double> %x, [2 x <2 x double>] %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[E0:.+]] = extractvalue [2 x <2 x double>] %"x'", 0
; CHECK-NEXT:   %[[E1:.+]] = extractvalue [2 x <2 x double>] %"x'", 1
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt <2 x double> %x, zeroinitializer
; CHECK-NEXT:   %[[N0:.+]] = {{(fsub fast <2 x double> .*,|fneg fast <2 x double>)}} %[[E0]]
; CHECK-NEXT:   %[[S0:.+]] = select {{(fast )?}}<2 x i1> %[[CMP]], <2 x double> %[[N0]], <2 x double> %[[E0]]
; CHECK-NEXT:   %[[N1:.+]] = {{(fsub fast <2 x double> .*,|fneg fast <2 x double>)}} %[[E1]]
; CHECK-NEXT:   %[[S1:.+]] = select {{(fast )?}}<2 x i1> %[[CMP]], <2 x double> %[[N1]], <2 x double> %[[E1]]
; CHECK-NEXT:   %[[I0:.+]] = insertvalue [2 x <2 x double>] undef, <2 x double> %[[S0]], 0
; CHECK-NEXT:   %[[I1:.+]] = insertvalue [2 x <2 x double>] %[[I0]], <2 x double> %[[S1]], 1
; CHECK-NEXT:   ret [2 x <2 x double>] %[[I1]]
; CHECK-NEXT: }

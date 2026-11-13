; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

define double @abs_scalar(double %x) {
entry:
  %cstx = bitcast double %x to i64
  %absx = and i64 %cstx, 9223372036854775807
  %csty = bitcast i64 %absx to double
  ret double %csty
}

define <2 x double> @abs_vec(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %absx = and <2 x i64> %cstx, <i64 9223372036854775807, i64 9223372036854775807>
  %csty = bitcast <2 x i64> %absx to <2 x double>
  ret <2 x double> %csty
}

define <2 x double> @abs_mixed(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %absx = and <2 x i64> %cstx, <i64 9223372036854775807, i64 -1>
  %csty = bitcast <2 x i64> %absx to <2 x double>
  ret <2 x double> %csty
}

define <2 x double> @abs_load(<2 x i64>* %p) {
entry:
  %v = load <2 x i64>, <2 x i64>* %p, align 8
  %absx = and <2 x i64> %v, <i64 9223372036854775807, i64 9223372036854775807>
  %csty = bitcast <2 x i64> %absx to <2 x double>
  ret <2 x double> %csty
}

define <2 x double> @abs_insertelem(<2 x double> %x) {
entry:
  %m0 = insertelement <4 x i32> undef, i32 -1, i32 0
  %m1 = insertelement <4 x i32> %m0, i32 2147483647, i32 1
  %m2 = insertelement <4 x i32> %m1, i32 -1, i32 2
  %m3 = insertelement <4 x i32> %m2, i32 2147483647, i32 3
  %mask = bitcast <4 x i32> %m3 to <2 x i64>
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %absx = and <2 x i64> %cstx, %mask
  %csty = bitcast <2 x i64> %absx to <2 x double>
  ret <2 x double> %csty
}

define double @test_scalar(double %x, double %dx) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_fwddiff(double (double)* nonnull @abs_scalar, double %x, double %dx)
  ret double %0
}

define <2 x double> @test_vec(<2 x double> %x, <2 x double> %dx) {
entry:
  %0 = tail call <2 x double> (<2 x double> (<2 x double>)*, ...) @__enzyme_fwddiff2(<2 x double> (<2 x double>)* nonnull @abs_vec, <2 x double> %x, <2 x double> %dx)
  ret <2 x double> %0
}

define <2 x double> @test_mixed(<2 x double> %x, <2 x double> %dx) {
entry:
  %0 = tail call <2 x double> (<2 x double> (<2 x double>)*, ...) @__enzyme_fwddiff2(<2 x double> (<2 x double>)* nonnull @abs_mixed, <2 x double> %x, <2 x double> %dx)
  ret <2 x double> %0
}

define <2 x double> @test_load(<2 x i64>* %p, <2 x i64>* %dp) {
entry:
  %0 = tail call <2 x double> (<2 x double> (<2 x i64>*)*, ...) @__enzyme_fwddiff3(<2 x double> (<2 x i64>*)* nonnull @abs_load, metadata !"enzyme_dup", <2 x i64>* %p, <2 x i64>* %dp)
  ret <2 x double> %0
}

define <2 x double> @test_insertelem(<2 x double> %x, <2 x double> %dx) {
entry:
  %0 = tail call <2 x double> (<2 x double> (<2 x double>)*, ...) @__enzyme_fwddiff2(<2 x double> (<2 x double>)* nonnull @abs_insertelem, <2 x double> %x, <2 x double> %dx)
  ret <2 x double> %0
}

declare double @__enzyme_fwddiff(double (double)*, ...)
declare <2 x double> @__enzyme_fwddiff2(<2 x double> (<2 x double>)*, ...)
declare <2 x double> @__enzyme_fwddiff3(<2 x double> (<2 x i64>*)*, ...)

; CHECK: define internal double @fwddiffeabs_scalar(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %"x'"
; CHECK-NEXT:   %[[SEL:.+]] = select {{(fast )?}}i1 %[[CMP]], double %[[NEG]], double %"x'"
; CHECK-NEXT:   ret double %[[SEL]]
; CHECK-NEXT: }

; CHECK: define internal <2 x double> @fwddiffeabs_vec(<2 x double> %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt <2 x double> %x, zeroinitializer
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast <2 x double> .*,|fneg fast <2 x double>)}} %"x'"
; CHECK-NEXT:   %[[SEL:.+]] = select {{(fast )?}}<2 x i1> %[[CMP]], <2 x double> %[[NEG]], <2 x double> %"x'"
; CHECK-NEXT:   ret <2 x double> %[[SEL]]
; CHECK-NEXT: }

; CHECK: define internal <2 x double> @fwddiffeabs_mixed(<2 x double> %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt <2 x double> %x, zeroinitializer
; CHECK-NEXT:   %[[AND:.+]] = and <2 x i1> <i1 true, i1 false>, %[[CMP]]
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast <2 x double> .*,|fneg fast <2 x double>)}} %"x'"
; CHECK-NEXT:   %[[SEL:.+]] = select {{(fast )?}}<2 x i1> %[[AND]], <2 x double> %[[NEG]], <2 x double> %"x'"
; CHECK-NEXT:   ret <2 x double> %[[SEL]]
; CHECK-NEXT: }

; CHECK: define internal <2 x double> @fwddiffeabs_load(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[DP:.+]] = load <2 x i64>,
; CHECK-NEXT:   %[[P:.+]] = load <2 x i64>,
; CHECK-NEXT:   %[[XFP:.+]] = bitcast <2 x i64> %[[P]] to <2 x double>
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt <2 x double> %[[XFP]], zeroinitializer
; CHECK-NEXT:   %[[DFP:.+]] = bitcast <2 x i64> %[[DP]] to <2 x double>
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast <2 x double> .*,|fneg fast <2 x double>)}} %[[DFP]]
; CHECK-NEXT:   %[[SEL:.+]] = select {{(fast )?}}<2 x i1> %[[CMP]], <2 x double> %[[NEG]], <2 x double> %[[DFP]]
; CHECK-NEXT:   ret <2 x double> %[[SEL]]
; CHECK-NEXT: }

; CHECK: define internal <2 x double> @fwddiffeabs_insertelem(<2 x double> %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt <2 x double> %x, zeroinitializer
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast <2 x double> .*,|fneg fast <2 x double>)}} %"x'"
; CHECK-NEXT:   %[[SEL:.+]] = select {{(fast )?}}<2 x i1> %[[CMP]], <2 x double> %[[NEG]], <2 x double> %"x'"
; CHECK-NEXT:   ret <2 x double> %[[SEL]]
; CHECK-NEXT: }

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

define double @abs_load(<2 x i64>* %p) {
entry:
  %v = load <2 x i64>, <2 x i64>* %p, align 8
  %absx = and <2 x i64> %v, <i64 9223372036854775807, i64 9223372036854775807>
  %csty = bitcast <2 x i64> %absx to <2 x double>
  %e0 = extractelement <2 x double> %csty, i32 0
  %e1 = extractelement <2 x double> %csty, i32 1
  %sum = fadd double %e0, %e1
  ret double %sum
}

define double @test_scalar(double %x) {
entry:
  %0 = tail call double (double (double)*, ...) @__enzyme_autodiff(double (double)* nonnull @abs_scalar, double %x)
  ret double %0
}

define <2 x double> @test_vec(<2 x double> %x) {
entry:
  %0 = tail call <2 x double> (<2 x double> (<2 x double>)*, ...) @__enzyme_autodiff2(<2 x double> (<2 x double>)* nonnull @abs_vec, <2 x double> %x)
  ret <2 x double> %0
}

define <2 x double> @test_mixed(<2 x double> %x) {
entry:
  %0 = tail call <2 x double> (<2 x double> (<2 x double>)*, ...) @__enzyme_autodiff2(<2 x double> (<2 x double>)* nonnull @abs_mixed, <2 x double> %x)
  ret <2 x double> %0
}

define void @test_load(<2 x i64>* %p, <2 x i64>* %dp) {
entry:
  tail call void (double (<2 x i64>*)*, ...) @__enzyme_autodiff3(double (<2 x i64>*)* nonnull @abs_load, metadata !"enzyme_dup", <2 x i64>* %p, <2 x i64>* %dp)
  ret void
}

declare double @__enzyme_autodiff(double (double)*, ...)
declare <2 x double> @__enzyme_autodiff2(<2 x double> (<2 x double>)*, ...)
declare void @__enzyme_autodiff3(double (<2 x i64>*)*, ...)

; CHECK: define internal { double } @diffeabs_scalar(double %x, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt double %x, 0.000000e+00
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast double \-?0.000000e\+00,|fneg fast double)}} %differeturn
; CHECK-NEXT:   %[[SEL:.+]] = select {{(fast )?}}i1 %[[CMP]], double %[[NEG]], double %differeturn
; CHECK-NEXT:   %[[RET:.+]] = insertvalue { double } undef, double %[[SEL]], 0
; CHECK-NEXT:   ret { double } %[[RET]]
; CHECK-NEXT: }

; CHECK: define internal { <2 x double> } @diffeabs_vec(<2 x double> %x, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt <2 x double> %x, zeroinitializer
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast <2 x double> .*,|fneg fast <2 x double>)}} %differeturn
; CHECK-NEXT:   %[[SEL:.+]] = select {{(fast )?}}<2 x i1> %[[CMP]], <2 x double> %[[NEG]], <2 x double> %differeturn
; CHECK-NEXT:   %[[RET:.+]] = insertvalue { <2 x double> } undef, <2 x double> %[[SEL]], 0
; CHECK-NEXT:   ret { <2 x double> } %[[RET]]
; CHECK-NEXT: }

; CHECK: define internal { <2 x double> } @diffeabs_mixed(<2 x double> %x, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CMP:.+]] = fcmp {{(fast )?}}olt <2 x double> %x, zeroinitializer
; CHECK-NEXT:   %[[AND:.+]] = and <2 x i1> <i1 true, i1 false>, %[[CMP]]
; CHECK-NEXT:   %[[NEG:.+]] = {{(fsub fast <2 x double> .*,|fneg fast <2 x double>)}} %differeturn
; CHECK-NEXT:   %[[SEL:.+]] = select {{(fast )?}}<2 x i1> %[[AND]], <2 x double> %[[NEG]], <2 x double> %differeturn
; CHECK-NEXT:   %[[RET:.+]] = insertvalue { <2 x double> } undef, <2 x double> %[[SEL]], 0
; CHECK-NEXT:   ret { <2 x double> } %[[RET]]
; CHECK-NEXT: }

; CHECK: define internal void @diffeabs_load(
; CHECK:   %[[P:.+]] = load <2 x i64>,
; CHECK:   %[[XFP:.+]] = bitcast <2 x i64> %[[P]] to <2 x double>
; CHECK:   %[[CMP:.+]] = fcmp {{(fast )?}}olt <2 x double> %[[XFP]], zeroinitializer
; CHECK:   select {{(fast )?}}<2 x i1> %[[CMP]], <2 x double>
; CHECK:   fadd fast <2 x double>
; CHECK:   store <2 x double>
; CHECK:   ret void
; CHECK-NEXT: }

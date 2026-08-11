; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

define <2 x double> @tester(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %normx = or <2 x i64> %cstx, <i64 4602678819172646912, i64 4602678819172646912>
  %csty = bitcast <2 x i64> %normx to <2 x double>
  ret <2 x double> %csty
}

define <2 x double> @test_derivative(<2 x double> %x) {
entry:
  %0 = tail call <2 x double> (<2 x double> (<2 x double>)*, ...) @__enzyme_autodiff(<2 x double> (<2 x double>)* nonnull @tester, <2 x double> %x)
  ret <2 x double> %0
}

declare <2 x double> @__enzyme_autodiff(<2 x double> (<2 x double>)*, ...)

; CHECK: define internal { <2 x double> } @diffetester(<2 x double> %x, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CSTX:.+]] = bitcast <2 x double> %x to <2 x i64>
; CHECK-NEXT:   %[[OR:.+]] = or <2 x i64> %[[CSTX]], <i64 4602678819172646912, i64 4602678819172646912>
; CHECK-NEXT:   %[[SUB:.+]] = sub nuw <2 x i64> %[[OR]], %[[CSTX]]
; CHECK-NEXT:   %[[ADD:.+]] = add nuw nsw <2 x i64> %[[SUB]], <i64 4607182418800017408, i64 4607182418800017408>
; CHECK-NEXT:   %[[BC:.+]] = bitcast <2 x i64> %[[ADD]] to <2 x double>
; CHECK-NEXT:   %[[MUL:.+]] = fmul fast <2 x double> %differeturn, %[[BC]]
; CHECK-NEXT:   %[[IV:.+]] = insertvalue { <2 x double> } undef, <2 x double> %[[MUL]], 0
; CHECK-NEXT:   ret { <2 x double> } %[[IV]]
; CHECK-NEXT: }

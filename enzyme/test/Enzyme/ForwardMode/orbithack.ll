; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(instsimplify)" -enzyme-preopt=false -S | FileCheck %s

define double @tester(<2 x double> %x) {
entry:
  %cstx = bitcast <2 x double> %x to <2 x i64>
  %normx = or <2 x i64> %cstx, <i64 4602678819172646912, i64 4602678819172646912>
  %csty = bitcast <2 x i64> %normx to <2 x double>
  %res = extractelement <2 x double> %csty, i32 0
  ret double %res
}

define double @test_derivative(<2 x double> %x, <2 x double> %dx) {
entry:
  %0 = tail call double (double (<2 x double>)*, ...) @__enzyme_fwddiff(double (<2 x double>)* nonnull @tester, <2 x double> %x, <2 x double> %dx)
  ret double %0
}

declare double @__enzyme_fwddiff(double (<2 x double>)*, ...)

; CHECK: define internal double @fwddiffetester(<2 x double> %x, <2 x double> %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[CSTX:.+]] = bitcast <2 x double> %x to <2 x i64>
; CHECK-NEXT:   %[[OR:.+]] = or <2 x i64> %[[CSTX]], <i64 4602678819172646912, i64 4602678819172646912>
; CHECK-NEXT:   %[[SUB:.+]] = sub nuw <2 x i64> %[[OR]], %[[CSTX]]
; CHECK-NEXT:   %[[ADD:.+]] = add nuw nsw <2 x i64> %[[SUB]], <i64 4607182418800017408, i64 4607182418800017408>
; CHECK-NEXT:   %[[BC:.+]] = bitcast <2 x i64> %[[ADD]] to <2 x double>
; CHECK-NEXT:   %[[MUL:.+]] = fmul fast <2 x double> %"x'", %[[BC]]
; CHECK-NEXT:   %[[EXT:.+]] = extractelement <2 x double> %[[MUL]], i32 0
; CHECK-NEXT:   ret double %[[EXT]]
; CHECK-NEXT: }

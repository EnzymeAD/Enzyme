; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,%simplifycfg)" -S | FileCheck %s; fi

; Function Attrs: noinline nounwind readnone uwtable
define float @tester(float %x, float %y) {
entry:
  %0 = tail call float @llvm.maximum.f32(float %x, float %y)
  ret float %0
}

define float @test_derivative(float %x, float %y) {
entry:
  %0 = tail call float (...) @__enzyme_fwddiff(ptr nonnull @tester, float %x, float 1.0, float %y, float 1.0)
  ret float %0
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.maximum.f32(float, float)

; Function Attrs: nounwind
declare float @__enzyme_fwddiff(...)

; CHECK: define internal {{(dso_local )?}}float @fwddiffetester(float %x, float %"x'", float %y, float %"y'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[ORD:.+]] = fcmp ord float %x, %x
; CHECK-NEXT:   %[[ULT:.+]] = fcmp ult float %x, %y
; CHECK-NEXT:   %[[OEQ:.+]] = fcmp oeq float %x, %y
; CHECK-NEXT:   %[[XI:.+]] = bitcast float %x to i32
; CHECK-NEXT:   %[[YI:.+]] = bitcast float %y to i32
; CHECK-NEXT:   %[[NOTYI:.+]] = xor i32 %[[YI]], -1
; CHECK-NEXT:   %[[MASK:.+]] = and i32 %[[XI]], %[[NOTYI]]
; CHECK-NEXT:   %[[SIGN:.+]] = icmp slt i32 %[[MASK]], 0
; CHECK-NEXT:   %[[NPZ:.+]] = and i1 %[[OEQ]], %[[SIGN]]
; CHECK-NEXT:   %[[OR:.+]] = or i1 %[[ULT]], %[[NPZ]]
; CHECK-NEXT:   %[[CMP:.+]] = and i1 %[[ORD]], %[[OR]]
; CHECK-NEXT:   %[[RES:.+]] = select {{(fast )?}}i1 %[[CMP]], float %"y'", float %"x'"
; CHECK-NEXT:   ret float %[[RES]]
; CHECK-NEXT: }


; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-activity-analysis" -activity-analysis-func=test -S -o /dev/null | FileCheck %s

; Ensure insertAllFrom preserves Hypothesis.ReEvaluateValueIfInactiveInst so
; that when an inner pointer's store instruction is later proven inactive, the
; inner pointer and its dependent instructions are re-evaluated as constant.

declare ptr @malloc(i64)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

define double @test(double %x) {
entry:
  %outer3 = alloca double
  %outer2 = alloca double
  %outer = alloca double
  %buf = call ptr @malloc(i64 8)
  %ld = load double, ptr %buf
  %ld2 = fadd double %ld, 1.0
  store double %ld2, ptr %outer
  call void @llvm.memcpy.p0.p0.i64(ptr %outer2, ptr %outer, i64 8, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %outer3, ptr %outer2, i64 8, i1 false)
  store double %x, ptr %outer3
  %res = load double, ptr %outer3
  ret double %res
}

; CHECK: double %x: icv:0
; CHECK-NEXT: entry
; CHECK-NEXT:   %outer3 = alloca double, align 8: icv:0 ici:1
; CHECK-NEXT:   %outer2 = alloca double, align 8: icv:1 ici:1
; CHECK-NEXT:   %outer = alloca double, align 8: icv:1 ici:1
; CHECK-NEXT:   %buf = call ptr @malloc(i64 8): icv:1 ici:1
; CHECK-NEXT:   %ld = load double, ptr %buf, align 8: icv:1 ici:1
; CHECK-NEXT:   %ld2 = fadd double %ld, 1.000000e+00: icv:1 ici:1
; CHECK-NEXT:   store double %ld2, ptr %outer, align 8: icv:1 ici:1
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %outer2, ptr %outer, i64 8, i1 false): icv:1 ici:1
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr %outer3, ptr %outer2, i64 8, i1 false): icv:1 ici:1
; CHECK-NEXT:   store double %x, ptr %outer3, align 8: icv:1 ici:0
; CHECK-NEXT:   %res = load double, ptr %outer3, align 8: icv:0 ici:0
; CHECK-NEXT:   ret double %res: icv:1 ici:1

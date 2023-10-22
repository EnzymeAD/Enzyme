; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=caller -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=caller -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @caller(i32* %p) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %mul = phi i64 [ 0, %entry ], [ %indvars.iv.next2, %for.body ]
  %p2 = getelementptr i32, i32* %p, i64 %mul
  %p3 = getelementptr i32, i32* %p2, i64 %indvars.iv
  %ld = load i32, i32* %p3, align 8, !tbaa !2
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %indvars.iv.next2 = add nuw i64 %indvars.iv, 25
  %exitcond = icmp eq i64 %indvars.iv, 4
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0, i64 8}
!3 = !{!4, i64 8, !"float"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; TODO in this test, the float offsets are [0 * 0, 104, 208, 312, 416]
;   incorrectly attributing the same offset in two places will get incorrect answers of [0, 4, 8, 12, 16, 100, 104, 108, 112, 116, 200, 204, 208, 216, ...]

; CHECK: caller - {} |{[-1]:Pointer}:{} 
; CHECK-NEXT: i32* %p: {[-1]:Pointer}
; CHECK-NEXT: entry
; CHECK-NEXT:   br label %for.body: {}
; CHECK-NEXT: for.body
; CHECK-NEXT:   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]: {[-1]:Integer}
; CHECK-NEXT:   %mul = phi i64 [ 0, %entry ], [ %indvars.iv.next2, %for.body ]: {[-1]:Integer}
; CHECK-NEXT:   %p2 = getelementptr i32, i32* %p, i64 %mul: {[-1]:Pointer}
; CHECK-NEXT:   %p3 = getelementptr i32, i32* %p2, i64 %indvars.iv: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %ld = load i32, i32* %p3, align 8, !tbaa !2: {[-1]:Float@float}
; CHECK-NEXT:   %indvars.iv.next = add nuw i64 %indvars.iv, 1: {[-1]:Integer}
; CHECK-NEXT:   %indvars.iv.next2 = add nuw i64 %indvars.iv, 25: {[-1]:Integer}
; CHECK-NEXT:   %exitcond = icmp eq i64 %indvars.iv, 4: {[-1]:Integer}
; CHECK-NEXT:   br i1 %exitcond, label %for.cond.cleanup, label %for.body: {}
; CHECK-NEXT: for.cond.cleanup
; CHECK-NEXT:   ret void: {}

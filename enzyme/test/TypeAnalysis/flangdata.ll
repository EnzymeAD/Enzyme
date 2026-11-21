; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=copy -S | FileCheck %s; fi

; flang tags accesses to Fortran data with TBAA types such as
; "global data/<name>" (here a COMMON block /blk/). Fortran does not
; reinterpret memory, so a floating-point load or store through such a tag
; accesses floating-point data, even when the value is only copied, as in
;
;   REAL*8 A, B
;   COMMON /BLK/ A, B
;   B = A

@blk_ = common global [16 x i8] zeroinitializer, align 8

define void @copy() {
entry:
  %a = load double, ptr @blk_, align 8, !tbaa !0
  %bp = getelementptr inbounds nuw i8, ptr @blk_, i64 8
  store double %a, ptr %bp, align 8, !tbaa !0
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"global data/blk_", !2, i64 0}
!2 = !{!"global data", !3, i64 0}
!3 = !{!"any data access", !4, i64 0}
!4 = !{!"any access", !5, i64 0}
!5 = !{!"Flang function root _QPcopy"}

; CHECK: copy - {} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %a = load double, ptr @blk_, align 8, !tbaa !{{[0-9]+}}: {[-1]:Float@double}
; CHECK-NEXT:   %bp = getelementptr inbounds nuw i8, ptr @blk_, i64 8: {[-1]:Pointer, [-1,-1]:Float@double}
; CHECK-NEXT:   store double %a, ptr %bp, align 8, !tbaa !{{[0-9]+}}: {}

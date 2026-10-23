; RUN: if [ %llvmver -ge 17 ]; then %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=lbound -S | FileCheck %s; fi

; flang tags accesses to the fields of Fortran descriptors with the TBAA type
; "descriptor member". The fields are pointers and integers, never
; floating point: here, the base address and the lower bound of a rank-1
; allocatable module variable, as flang emits them.

@_QMmo_stateEwork = dso_local global { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] } zeroinitializer, align 8

define double @lbound() {
entry:
  %base = load ptr, ptr @_QMmo_stateEwork, align 8, !tbaa !0
  %lbp = getelementptr inbounds nuw i8, ptr @_QMmo_stateEwork, i64 24
  %lb = load i64, ptr %lbp, align 8, !tbaa !0
  %off = sub nsw i64 1, %lb
  %p = getelementptr double, ptr %base, i64 %off
  %v = load double, ptr %p, align 8
  ret double %v
}

!0 = !{!1, !1, i64 0}
!1 = !{!"descriptor member", !2, i64 0}
!2 = !{!"any access", !3, i64 0}
!3 = !{!"Flang function root _QMmo_statePlbound"}

; CHECK: lbound - {[-1]:Float@double} |
; CHECK-NEXT: entry
; CHECK-NEXT:   %base = load ptr, ptr @_QMmo_stateEwork, align 8, !tbaa !{{[0-9]+}}: {[-1]:Pointer}
; CHECK-NEXT:   %lbp = getelementptr inbounds nuw i8, ptr @_QMmo_stateEwork, i64 24: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT:   %lb = load i64, ptr %lbp, align 8, !tbaa !{{[0-9]+}}: {[-1]:Integer}
; CHECK-NEXT:   %off = sub nsw i64 1, %lb: {[-1]:Integer}

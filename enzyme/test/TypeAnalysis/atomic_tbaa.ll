; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -disable-output | FileCheck %s

define void @callee(ptr %int_ptr, ptr %dbl_ptr, ptr %ptr_slot, i64 %cmp_int, i64 %new_int, i64 %delta, double %cmp_dbl_f, double %new_dbl_f) {
entry:
  %cmp_dbl = bitcast double %cmp_dbl_f to i64
  %new_dbl = bitcast double %new_dbl_f to i64
  %rmw_add = atomicrmw add ptr %int_ptr, i64 %delta seq_cst, align 8, !tbaa !4
  %rmw_xchg = atomicrmw xchg ptr %dbl_ptr, i64 %new_dbl seq_cst, align 8, !tbaa !6
  %cas_int = cmpxchg ptr %int_ptr, i64 %cmp_int, i64 %new_int seq_cst monotonic, align 8, !tbaa !4
  %old_int = extractvalue { i64, i1 } %cas_int, 0
  %ok_int = extractvalue { i64, i1 } %cas_int, 1
  %cas_dbl = cmpxchg ptr %dbl_ptr, i64 %cmp_dbl, i64 %new_dbl seq_cst monotonic, align 8, !tbaa !6
  %old_dbl = extractvalue { i64, i1 } %cas_dbl, 0
  %ok_dbl = extractvalue { i64, i1 } %cas_dbl, 1
  %cas_ptr = cmpxchg ptr %ptr_slot, ptr null, ptr %dbl_ptr seq_cst monotonic, align 8, !tbaa !8
  %old_ptr = extractvalue { ptr, i1 } %cas_ptr, 0
  %ok_ptr = extractvalue { ptr, i1 } %cas_ptr, 1
  ret void
}

!0 = !{!"Simple C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!"any pointer", !1, i64 0}
!3 = !{!"long", !1, i64 0}
!4 = !{!3, !3, i64 0}
!5 = !{!"double", !1, i64 0}
!6 = !{!5, !5, i64 0}
!8 = !{!2, !2, i64 0}

; CHECK-LABEL: callee - {}
; CHECK-NEXT: ptr %int_ptr: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer}
; CHECK-NEXT: ptr %dbl_ptr: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT: ptr %ptr_slot: {[-1]:Pointer, [-1,0]:Pointer, [-1,0,0]:Float@double}
; CHECK:   %rmw_add = atomicrmw add ptr %int_ptr, i64 %delta seq_cst, align 8, !tbaa !{{[0-9]+}}: {[-1]:Integer}
; CHECK-NEXT:   %rmw_xchg = atomicrmw xchg ptr %dbl_ptr, i64 %new_dbl seq_cst, align 8, !tbaa !{{[0-9]+}}: {[-1]:Float@double}
; CHECK-NEXT:   %cas_int = cmpxchg ptr %int_ptr, i64 %cmp_int, i64 %new_int seq_cst monotonic, align 8, !tbaa !{{[0-9]+}}: {[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Integer, [5]:Integer, [6]:Integer, [7]:Integer, [8]:Integer}
; CHECK-NEXT:   %old_int = extractvalue { i64, i1 } %cas_int, 0: {[-1]:Integer}
; CHECK-NEXT:   %ok_int = extractvalue { i64, i1 } %cas_int, 1: {[-1]:Integer}
; CHECK-NEXT:   %cas_dbl = cmpxchg ptr %dbl_ptr, i64 %cmp_dbl, i64 %new_dbl seq_cst monotonic, align 8, !tbaa !{{[0-9]+}}: {[0]:Float@double, [8]:Integer}
; CHECK-NEXT:   %old_dbl = extractvalue { i64, i1 } %cas_dbl, 0: {[-1]:Float@double}
; CHECK-NEXT:   %ok_dbl = extractvalue { i64, i1 } %cas_dbl, 1: {[-1]:Integer}
; CHECK-NEXT:   %cas_ptr = cmpxchg ptr %ptr_slot, ptr null, ptr %dbl_ptr seq_cst monotonic, align 8, !tbaa !{{[0-9]+}}: {[0]:Pointer, [0,0]:Float@double, [8]:Integer}
; CHECK-NEXT:   %old_ptr = extractvalue { ptr, i1 } %cas_ptr, 0: {[-1]:Pointer, [-1,0]:Float@double}
; CHECK-NEXT:   %ok_ptr = extractvalue { ptr, i1 } %cas_ptr, 1: {[-1]:Integer}

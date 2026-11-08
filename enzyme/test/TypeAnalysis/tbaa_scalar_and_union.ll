; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=analyze_tbaa -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=analyze_tbaa -disable-output | FileCheck %s

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

define void @analyze_tbaa(ptr %short_ptr, ptr %wchar_ptr, ptr %char16_ptr, ptr %int128_ptr, ptr %bitint_ptr, ptr %f16_ptr, ptr %bf16_ptr, ptr %f128_ptr, ptr %typed_ptr_slot, ptr %any_p2_slot, ptr %union_dst, ptr %union_src, ptr %bitfield_dst, ptr %bitfield_src) {
entry:
  %short_val = load i16, ptr %short_ptr, align 2, !tbaa !2
  %wchar_val = load i32, ptr %wchar_ptr, align 4, !tbaa !4
  %char16_val = load i16, ptr %char16_ptr, align 2, !tbaa !6
  %int128_val = load i128, ptr %int128_ptr, align 16, !tbaa !8
  %bitint_val = load i32, ptr %bitint_ptr, align 4, !tbaa !10
  %f16_raw = load i16, ptr %f16_ptr, align 2, !tbaa !12
  %bf16_raw = load i16, ptr %bf16_ptr, align 2, !tbaa !14
  %f128_raw = load i128, ptr %f128_ptr, align 16, !tbaa !16
  %typed_ptr_raw = load i64, ptr %typed_ptr_slot, align 8, !tbaa !20
  %any_p2_raw = load i64, ptr %any_p2_slot, align 8, !tbaa !22
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %union_dst, ptr align 8 %union_src, i64 24, i1 false), !tbaa.struct !26
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %bitfield_dst, ptr align 4 %bitfield_src, i64 12, i1 false), !tbaa.struct !29
  ret void
}

!0 = !{!"Simple C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!3, !3, i64 0}
!3 = !{!"short", !1, i64 0}
!4 = !{!5, !5, i64 0}
!5 = !{!"wchar_t", !1, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"char16_t", !1, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"__int128", !1, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"_BitInt(24)", !1, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"_Float16", !1, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"__bf16", !1, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"__float128", !1, i64 0}
!18 = !{!0, i64 1, !"omnipotent char"}
!19 = !{!18, i64 8, !"any pointer"}
!20 = !{!21, !21, i64 0, i64 8}
!21 = !{!19, i64 8, !"p1 double"}
!22 = !{!23, !23, i64 0, i64 8}
!23 = !{!19, i64 8, !"any p2 pointer"}
!24 = !{!"double", !1, i64 0}
!25 = !{!24, !24, i64 0}
!27 = !{!"int", !1, i64 0}
!28 = !{!27, !27, i64 0}
!26 = !{i64 0, i64 8, !25, i64 8, i64 8, !25, i64 8, i64 4, !28, i64 12, i64 4, !28, i64 16, i64 2, !2}
!30 = !{!"float", !1, i64 0}
!31 = !{!30, !30, i64 0}
!29 = !{i64 0, i64 4, !28, i64 2, i64 4, !28, i64 4, i64 4, !31, i64 8, i64 4, !31}

; CHECK-LABEL: analyze_tbaa - {}
; CHECK-NEXT: ptr %short_ptr: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer}
; CHECK-NEXT: ptr %wchar_ptr: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; CHECK-NEXT: ptr %char16_ptr: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer}
; CHECK-NEXT: ptr %int128_ptr: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer}
; CHECK-NEXT: ptr %bitint_ptr: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer}
; CHECK-NEXT: ptr %f16_ptr: {[-1]:Pointer, [-1,0]:Float@half}
; CHECK-NEXT: ptr %bf16_ptr: {[-1]:Pointer, [-1,0]:Float@bf16}
; CHECK-NEXT: ptr %f128_ptr: {[-1]:Pointer, [-1,0]:Float@fp128}
; CHECK-NEXT: ptr %typed_ptr_slot: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT: ptr %any_p2_slot: {[-1]:Pointer, [-1,0]:Pointer}
; CHECK-NEXT: ptr %union_dst: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Integer, [-1,17]:Integer}
; CHECK-NEXT: ptr %union_src: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Integer, [-1,17]:Integer}
; CHECK-NEXT: ptr %bitfield_dst: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,8]:Float@float}
; CHECK-NEXT: ptr %bitfield_src: {[-1]:Pointer, [-1,0]:Integer, [-1,1]:Integer, [-1,2]:Integer, [-1,3]:Integer, [-1,8]:Float@float}

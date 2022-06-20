; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -simplifycfg -dce -S | FileCheck %s
; Function Attrs: nounwind

declare void @__enzyme_fwddiff.f64(...)

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_ptr(i8* nocapture %dst, i8* nocapture readonly %src, i64 %num) {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 %num, i1 false), !tbaa !17, !tbaa.struct !19
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #0

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_ptr(i8* %dst, <3 x i8>* %dstp, i8* %src, <3 x i8>* %dsrcp, i64 %n) {
entry:
  tail call void (...) @__enzyme_fwddiff.f64(void (i8*, i8*, i64)* nonnull @memcpy_ptr, metadata !"enzyme_width", i64 3, metadata !"enzyme_dup", i8* %dst, <3 x i8>* %dstp, metadata !"enzyme_dup", i8* %src, <3 x i8>* %dsrcp, i64 %n)
  ret void
}

attributes #0 = { argmemonly nounwind }

!17 = !{!18, !18, i64 0, i64 32}
!18 = !{!4, i64 32, !"_ZTSSt5arrayIlLm4EE", !9, i64 0, i64 32}

!19 = !{i64 0, i64 32, !20}
!20 = !{!9, !9, i64 0, i64 32}
!9 = !{!4, i64 8, !"long"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}


; CHECK: define internal void @fwddiffe3memcpy_ptr(i8* nocapture %dst, <3 x i8>* %"dst'", i8* nocapture readonly %src, <3 x i8>* %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* align 8 %src, i64 %num, i1 false) #1, !tbaa !0, !tbaa.struct !5
; CHECK-NEXT:   %num.vecsize = mul i64 3, %num
; CHECK-NEXT:   %0 = bitcast <3 x i8>* %"dst'" to i8*
; CHECK-NEXT:   %1 = bitcast <3 x i8>* %"src'" to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 %num.vecsize, i1 false) #1, !tbaa !0, !tbaa.struct !5
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
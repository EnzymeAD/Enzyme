; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme" -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_float(double* nocapture %dst, double* nocapture readonly %src, i64 %num) #0 {
entry:
  %dummy1 = load double, double* %dst, align 8, !enzyme_type !0
  %dummy2 = load double, double* %src, align 8, !enzyme_type !0
  %0 = bitcast double* %dst to i8*
  %1 = bitcast double* %src to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_float(double* %dst, double* %dstp, double* %src, double* %srcp, i64 %n) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_fwddiff.f64(void (double*, double*, i64)* nonnull @memcpy_float, metadata !"enzyme_runtime_activity", double* %dst, double* %dstp, double* %src, double* %srcp, i64 %n) #3
  ret void
}

declare void @__enzyme_fwddiff.f64(...) local_unnamed_addr

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind }

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Float@double"}

; CHECK: define internal void @fwddiffememcpy_float(ptr nocapture %dst, ptr nocapture %"dst'", ptr nocapture readonly %src, ptr nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[dst:.+]] = bitcast ptr %dst to ptr
; CHECK-NEXT:   %[[src:.+]] = bitcast ptr %src to ptr
; CHECK-NEXT:   %[[inactive:.+]] = icmp eq ptr %"src'", %[[src]]
; CHECK-NEXT:   %[[cpylen:.+]] = select i1 %[[inactive]], i64 0, i64 %num
; CHECK-NEXT:   %[[zerolen:.+]] = select i1 %[[inactive]], i64 %num, i64 0
; CHECK-NEXT:   tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %"dst'", ptr align 1 %"src'", i64 %[[cpylen]], i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 1 %"dst'", i8 0, i64 %[[zerolen]], i1 false)
; CHECK-NEXT:   tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[dst]], ptr align 1 %[[src]], i64 %num, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

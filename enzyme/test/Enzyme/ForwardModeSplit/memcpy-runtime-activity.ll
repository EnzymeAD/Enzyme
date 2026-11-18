; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -opaque-pointers -enzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -mem2reg -instsimplify -simplifycfg -dce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -opaque-pointers -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg,dce)" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s

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
  tail call void (...) @__enzyme_fwdsplit.f64(void (double*, double*, i64)* nonnull @memcpy_float, metadata !"enzyme_runtime_activity", double* %dst, double* %dstp, double* %src, double* %srcp, i64 %n, i8* null) #3
  ret void
}

define dso_local void @dmemcpy_const_src(double* %dst, double* %dstp, double* %src, i64 %n) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_fwdsplit.f64(void (double*, double*, i64)* nonnull @memcpy_float, metadata !"enzyme_runtime_activity", double* %dst, double* %dstp, metadata !"enzyme_const", double* %src, i64 %n, i8* null) #3
  ret void
}

declare void @__enzyme_fwdsplit.f64(...) local_unnamed_addr

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind }

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Float@double"}

; CHECK: define internal {{(dso_local )?}}void @fwddiffememcpy_float(ptr nocapture %dst, ptr nocapture %"dst'", ptr nocapture readonly %src, ptr nocapture %"src'", i64 %num, ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[dst_in:%.*]] = icmp eq ptr %"dst'", %dst
; CHECK-NEXT:   [[src_in:%.*]] = icmp eq ptr %"src'", %src
; CHECK-NEXT:   [[nocpy:%.*]] = or i1 [[dst_in]], [[src_in]]
; CHECK-NEXT:   [[cpylen:%.*]] = select i1 [[nocpy]], i64 0, i64 %num
; CHECK-NEXT:   [[dst_act:%.*]] = xor i1 [[dst_in]], true
; CHECK-NEXT:   [[dozero:%.*]] = and i1 [[dst_act]], [[src_in]]
; CHECK-NEXT:   [[zerolen:%.*]] = select i1 [[dozero]], i64 %num, i64 0
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %"dst'", ptr align 1 %"src'", i64 [[cpylen]], i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 1 %"dst'", i8 0, i64 [[zerolen]], i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @fwddiffememcpy_float.2(ptr nocapture %dst, ptr nocapture %"dst'", ptr nocapture readonly %src, i64 %num, ptr %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[dst_in_c:%.*]] = icmp eq ptr %"dst'", %dst
; CHECK-NEXT:   [[len_c:%.*]] = select i1 [[dst_in_c]], i64 0, i64 %num
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 1 %"dst'", i8 0, i64 [[len_c]], i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

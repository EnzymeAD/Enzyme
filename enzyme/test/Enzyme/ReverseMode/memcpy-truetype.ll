; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_float(i8* nocapture %dst, i8* nocapture readonly %src) #0 {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 100000, i1 false), !enzyme_truetype !0
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_float(i8* %dst, i8* %dstp, i8* %src, i64 %n) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_autodiff.f64(void (i8*, i8*)* nonnull @memcpy_float, metadata !"enzyme_dup", i8* %dst, i8* %dstp, metadata !"enzyme_dup", i8* %src, i8* %src) #3
  ret void
}

declare void @__enzyme_autodiff.f64(...) local_unnamed_addr


attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }

!0 = !{!"Float@float", i64 0, !"Integer", i64 8, !"Float@float", i64 50000, !"Integer", i64 50008}


; CHECK: define internal void @diffememcpy_float(i8* nocapture %dst, i8* nocapture %"dst'", i8* nocapture readonly %src, i8* nocapture %"src'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = getelementptr inbounds i8, i8* %"dst'", i64 8
; CHECK-NEXT:   %1 = getelementptr inbounds i8, i8* %"src'", i64 8
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 49992, i1 false)
; CHECK-NEXT:   %2 = getelementptr inbounds i8, i8* %"dst'", i64 50008
; CHECK-NEXT:   %3 = getelementptr inbounds i8, i8* %"src'", i64 50008
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %2, i8* align 1 %3, i64 49992, i1 false)
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %dst, i8* align 1 %src, i64 100000, i1 false) #{{[0-9]+}}, !enzyme_truetype
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %4 = bitcast i8* %"dst'" to float*
; CHECK-NEXT:   %5 = bitcast i8* %"src'" to float*
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %invertentry
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %invertentry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds float, float* %4, i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load float, float* %dst.i.i, align 1
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds float, float* %5, i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load float, float* %src.i.i, align 1
; CHECK-NEXT:   %6 = fadd fast float %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store float %6, float* %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %7 = icmp eq i64 2, %idx.next.i
; CHECK-NEXT:   br i1 %7, label %__enzyme_memcpyadd_floatda1sa1.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit:              ; preds = %for.body.i
; CHECK-NEXT:   %8 = getelementptr inbounds i8, i8* %"dst'", i64 50000
; CHECK-NEXT:   %9 = bitcast i8* %8 to float*
; CHECK-NEXT:   %10 = getelementptr inbounds i8, i8* %"src'", i64 50000
; CHECK-NEXT:   %11 = bitcast i8* %10 to float*
; CHECK-NEXT:   br label %for.body.i7

; CHECK: for.body.i7:                                      ; preds = %for.body.i7, %__enzyme_memcpyadd_floatda1sa1.exit
; CHECK-NEXT:   %idx.i1 = phi i64 [ 0, %__enzyme_memcpyadd_floatda1sa1.exit ], [ %idx.next.i6, %for.body.i7 ]
; CHECK-NEXT:   %dst.i.i2 = getelementptr inbounds float, float* %9, i64 %idx.i1
; CHECK-NEXT:   %dst.i.l.i3 = load float, float* %dst.i.i2, align 1
; CHECK-NEXT:   store float 0.000000e+00, float* %dst.i.i2, align 1
; CHECK-NEXT:   %src.i.i4 = getelementptr inbounds float, float* %11, i64 %idx.i1
; CHECK-NEXT:   %src.i.l.i5 = load float, float* %src.i.i4, align 1
; CHECK-NEXT:   %12 = fadd fast float %src.i.l.i5, %dst.i.l.i3
; CHECK-NEXT:   store float %12, float* %src.i.i4, align 1
; CHECK-NEXT:   %idx.next.i6 = add nuw i64 %idx.i1, 1
; CHECK-NEXT:   %13 = icmp eq i64 2, %idx.next.i6
; CHECK-NEXT:   br i1 %13, label %__enzyme_memcpyadd_floatda1sa1.exit8, label %for.body.i7

; CHECK: __enzyme_memcpyadd_floatda1sa1.exit8:             ; preds = %for.body.i7
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

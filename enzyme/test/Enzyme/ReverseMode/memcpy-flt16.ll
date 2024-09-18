; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_float(double* nocapture %dst, double* nocapture readonly %src, i64 %num) #0 {
entry:
  %0 = bitcast double* %dst to i8*
  %1 = bitcast double* %src to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 %1, i64 %num, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_float(double* %dst, double* %dstp, double* %src, double* %srcp, i64 %n) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_autodiff.f64(void (double*, double*, i64)* nonnull @memcpy_float, double* %dst, double* %dstp, double* %src, double* %srcp, i64 %n) #3
  ret void
}

declare void @__enzyme_autodiff.f64(...) local_unnamed_addr

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }
attributes #3 = { nounwind }

; CHECK: define internal {{(dso_local )?}}void @diffememcpy_float(double* nocapture %dst, double* nocapture %"dst'", double* nocapture readonly %src, double* nocapture %"src'", i64 %num) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast double* %dst to i8*
; CHECK-NEXT:   %1 = bitcast double* %src to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 %1, i64 %num, i1 false)
; CHECK-NEXT:   %2 = udiv i64 %num, 8
; CHECK-NEXT:   %3 = {{(icmp eq i64 %2, 0|icmp ult i64 %num, 8)}}
; CHECK-NEXT:   br i1 %3, label %__enzyme_memcpyadd_doubleda16sa16.exit, label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %entry
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %"dst'", i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load double, double* %dst.i.i, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %dst.i.i, align 8
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, double* %"src'", i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load double, double* %src.i.i, align 8
; CHECK-NEXT:   %4 = fadd fast double %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store double %4, double* %src.i.i, align 8
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %5 = icmp eq i64 %2, %idx.next.i
; CHECK-NEXT:   br i1 %5, label %__enzyme_memcpyadd_doubleda16sa16.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_doubleda16sa16.exit:             ; preds = %entry, %for.body.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

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
  tail call void (...) @__enzyme_autodiff.f64(void (double*, double*, i64)* nonnull @memcpy_float, metadata !"enzyme_runtime_activity", double* %dst, double* %dstp, double* %src, double* %srcp, i64 %n) #3
  ret void
}

declare void @__enzyme_autodiff.f64(...) local_unnamed_addr

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #3 = { nounwind }

!0 = !{!"Unknown", i32 -1, !1}
!1 = !{!"Float@double"}

; CHECK: define internal {{(dso_local )?}}void @diffememcpy_float({{(double\*|ptr)}} nocapture %dst, {{(double\*|ptr)}} nocapture %"dst'", {{(double\*|ptr)}} nocapture readonly %src, {{(double\*|ptr)}} nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK:        tail call void @{{(llvm\.memcpy\.p0i8\.p0i8\.i64|llvm\.memcpy\.p0\.p0\.i64)}}({{(i8\*|ptr)}} align 1 %{{.*}}, {{(i8\*|ptr)}} align 1 %{{.*}}, i64 %num, i1 false)
; CHECK-NEXT:   %0 = icmp eq {{(double\*|ptr)}} %"dst'", %dst
; CHECK-NEXT:   %1 = icmp eq {{(double\*|ptr)}} %"src'", %src
; CHECK-NEXT:   %2 = udiv i64 %num, 8
; CHECK-NEXT:   %3 = icmp eq i64 %2, 0
; CHECK-NEXT:   %4 = or i1 %3, %0
; CHECK-NEXT:   br i1 %4, label %__enzyme_memcpyadd_doubleda1sa1_runtime_activity.exit, label %check_src.i

; CHECK: check_src.i:                                      ; preds = %entry
; CHECK-NEXT:   br i1 %1, label %memset_dst.i, label %for.body.i

; CHECK: memset_dst.i:                                     ; preds = %check_src.i
; CHECK-NEXT:   %5 = mul nuw nsw i64 %2, 8
; CHECK:        call void @{{(llvm\.memset\.p0i8\.i64|llvm\.memset\.p0\.i64)}}({{(i8\*|ptr)}} align 1 %{{.*}}, i8 0, i64 %5, i1 false)
; CHECK-NEXT:   br label %__enzyme_memcpyadd_doubleda1sa1_runtime_activity.exit

; CHECK: for.body.i:                                       ; preds = %for.body.i, %check_src.i
; CHECK-NEXT:   %idx.i = phi i64 [ 0, %check_src.i ], [ %idx.next.i, %for.body.i ]
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, {{(double\*|ptr)}} %"dst'", i64 %idx.i
; CHECK-NEXT:   %dst.i.l.i = load double, {{(double\*|ptr)}} %dst.i.i, align 1
; CHECK-NEXT:   store double 0.000000e+00, {{(double\*|ptr)}} %dst.i.i, align 1
; CHECK-NEXT:   %src.i.i = getelementptr inbounds double, {{(double\*|ptr)}} %"src'", i64 %idx.i
; CHECK-NEXT:   %src.i.l.i = load double, {{(double\*|ptr)}} %src.i.i, align 1
; CHECK-NEXT:   %6 = fadd fast double %src.i.l.i, %dst.i.l.i
; CHECK-NEXT:   store double %6, {{(double\*|ptr)}} %src.i.i, align 1
; CHECK-NEXT:   %idx.next.i = add nuw i64 %idx.i, 1
; CHECK-NEXT:   %7 = icmp eq i64 %2, %idx.next.i
; CHECK-NEXT:   br i1 %7, label %__enzyme_memcpyadd_doubleda1sa1_runtime_activity.exit, label %for.body.i

; CHECK: __enzyme_memcpyadd_doubleda1sa1_runtime_activity.exit: ; preds = %entry, %memset_dst.i, %for.body.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @__enzyme_memcpyadd_doubleda1sa1_runtime_activity({{(double\*|ptr)}} nocapture %dst, {{(double\*|ptr)}} nocapture %src, i64 %num, i1 %dst_inactive, i1 %src_inactive) #[[mymemattrs:.+]] {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = icmp eq i64 %num, 0
; CHECK-NEXT:   %1 = or i1 %0, %dst_inactive
; CHECK-NEXT:   br i1 %1, label %for.end, label %check_src

; CHECK: check_src:                                        ; preds = %entry
; CHECK-NEXT:   br i1 %src_inactive, label %memset_dst, label %for.body

; CHECK: memset_dst:                                       ; preds = %check_src
; CHECK-NEXT:   %2 = mul nuw nsw i64 %num, 8
; CHECK:        call void @{{(llvm\.memset\.p0i8\.i64|llvm\.memset\.p0\.i64)}}({{(i8\*|ptr)}} align 1 %{{.*}}, i8 0, i64 %2, i1 false)
; CHECK-NEXT:   br label %for.end

; CHECK: for.body:                                         ; preds = %for.body, %check_src
; CHECK-NEXT:   %idx = phi i64 [ 0, %check_src ], [ %idx.next, %for.body ]
; CHECK-NEXT:   %dst.i = getelementptr inbounds double, {{(double\*|ptr)}} %dst, i64 %idx
; CHECK-NEXT:   %dst.i.l = load double, {{(double\*|ptr)}} %dst.i, align 1
; CHECK-NEXT:   store double 0.000000e+00, {{(double\*|ptr)}} %dst.i, align 1
; CHECK-NEXT:   %src.i = getelementptr inbounds double, {{(double\*|ptr)}} %src, i64 %idx
; CHECK-NEXT:   %src.i.l = load double, {{(double\*|ptr)}} %src.i, align 1
; CHECK-NEXT:   %3 = fadd fast double %src.i.l, %dst.i.l
; CHECK-NEXT:   store double %3, {{(double\*|ptr)}} %src.i
; CHECK-NEXT:   %idx.next = add nuw i64 %idx, 1
; CHECK-NEXT:   %4 = icmp eq i64 %num, %idx.next
; CHECK-NEXT:   br i1 %4, label %for.end, label %for.body

; CHECK: for.end:                                          ; preds = %for.body, %memset_dst, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: attributes #[[mymemattrs]] = { alwaysinline {{(argmemonly nounwind|nounwind memory\(argmem: readwrite\))}} }

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

define dso_local void @memcpy_float(i8* %dst, i8* %src, i64 %num) {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %dst, i8* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %src, i64 %num, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

define dso_local void @dmemcpy_float(i8* %dst, i8* %dstp, i8* %src, i8* %srcp, i64 %n) {
entry:
  tail call void (...) @__enzyme_autodiff.f64(void (double*, double*, i64)* nonnull @memcpy_float, metadata !"enzyme_dup", i8* %dst, i8* %dstp, metadata !"enzyme_dup", i8* %src, i8* %srcp, i64 %n)
  ret void
}

declare void @__enzyme_autodiff.f64(...)

define internal void @diffememcpy_float(ptr %dst, ptr %"dst'", ptr %src, ptr %"src'", i64 %num) #1 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr  %dst, ptr "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %src, i64 %num, i1 false) #3
  %0 = udiv i64 %num, 8
  %1 = icmp eq i64 %0, 0
  br i1 %1, label %__enzyme_memcpyadd_doubleda1sa1.exit, label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  %idx.i = phi i64 [ 0, %entry ], [ %idx.next.i, %for.body.i ]
  %dst.i.i = getelementptr inbounds double, ptr %"dst'", i64 %idx.i
  %dst.i.l.i = load double, ptr %dst.i.i, align 1
  store double 0.000000e+00, ptr %dst.i.i, align 1
  %src.i.i = getelementptr inbounds double, ptr %"src'", i64 %idx.i
  %src.i.l.i = load double, ptr %src.i.i, align 1
  %2 = fadd fast double %src.i.l.i, %dst.i.l.i
  store double %2, ptr %src.i.i, align 1
  %idx.next.i = add nuw i64 %idx.i, 1
  %3 = icmp eq i64 %0, %idx.next.i
  br i1 %3, label %__enzyme_memcpyadd_doubleda1sa1.exit, label %for.body.i

__enzyme_memcpyadd_doubleda1sa1.exit:             ; preds = %entry, %for.body.i
  ret void
}

; CHECK: define internal {{(dso_local )?}}void @diffememcpy_float(double* nocapture %dst, double* nocapture %"dst'", double* nocapture readonly %src, double* nocapture %"src'", i64 %num) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %dst, i8* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %src, i64 %num, i1 false)
; CHECK-NEXT:   %2 = udiv i64 %num, 8
; CHECK-NEXT:   %3 = {{(icmp eq i64 %2, 0|icmp ult i64 %num, 8)}}
; CHECK-NEXT:   br i1 %3, label %__enzyme_memcpyadd_doubleda1sa1.exit, label %for.body.i

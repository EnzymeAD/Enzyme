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
  tail call void (...) @__enzyme_autodiff.f64(void (i8*, i8*, i64)* nonnull @memcpy_float, metadata !"enzyme_dup", i8* %dst, i8* %dstp, metadata !"enzyme_dup", i8* %src, i8* %srcp, i64 %n)
  ret void
}

declare void @__enzyme_autodiff.f64(...)



; CHECK: define internal {{(dso_local )?}}void @diffememcpy_float(i8* nocapture writeonly %dst, i8* nocapture %"dst'", i8* nocapture readonly %src, i8* nocapture %"src'", i64 %num) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %dst, i8* "enzyme_type"="{[-1]:Pointer, [-1,0]:Float@double}" %src, i64 %num, i1 false)
; CHECK:   label %__enzyme_memcpyadd_doubleda1sa1.exit

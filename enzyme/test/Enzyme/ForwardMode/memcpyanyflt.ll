; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

@_j_const2 = private unnamed_addr constant { i64, double } { i64 1, double 1.000000e+00 }, align 8

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_ptr(i8* nocapture %dst, i8* nocapture readonly %src, i64 %num) {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* bitcast ({ i64, double }* @_j_const2 to i8*), i64 16, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #0

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_ptr(i8* %dst, i8* %dstp, i8* %src, i8* %srcp, i64 %n) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff.f64(void (i8*, i8*, i64)* nonnull @memcpy_ptr, metadata !"enzyme_dup", i8* %dst, i8* %dstp, metadata !"enzyme_dup", i8* %src, i8* %srcp, i64 %n)
  ret void
}

declare double @__enzyme_fwddiff.f64(...) local_unnamed_addr

attributes #0 = { argmemonly nounwind }

; CHECK: define internal void @fwddiffememcpy_ptr(i8* nocapture %dst, i8* nocapture %"dst'", i8* nocapture readonly %src, i8* nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   tail call void @llvm.memset.p0i8.i64(i8* align 1 %"dst'", i8 0, i64 16, i1 true) 
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %dst, i8* bitcast ({ i64, double }* @_j_const2 to i8*), i64 16, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

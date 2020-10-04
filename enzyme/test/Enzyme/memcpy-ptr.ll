; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -S | FileCheck %s

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_ptr(double** nocapture %dst, double** nocapture readonly %src, i64 %num) #0 {
entry:
  %0 = bitcast double** %dst to i8*
  %1 = bitcast double** %src to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_ptr(double** %dst, double** %dstp, double** %src, double** %srcp, i64 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (...) @__enzyme_autodiff.f64(void (double**, double**, i64)* nonnull @memcpy_ptr, double** %dst, double** %dstp, double** %src, double** %srcp, i64 %n) #3
  ret void
}

declare double @__enzyme_autodiff.f64(...) local_unnamed_addr

; Function Attrs: noinline nounwind uwtable
define dso_local void @submemcpy_ptr(double** nocapture %dst, double** nocapture readonly %src, i64 %num) local_unnamed_addr #2 {
entry:
  %0 = bitcast double** %dst to i8*
  %1 = bitcast double** %src to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @memcpyaugment_ptr(double** nocapture %dst, double** nocapture readonly %src, i64 %num) #0 {
entry:
  tail call void @submemcpy_ptr(double** %dst, double** %src, i64 %num)
  store double* null, double** %dst
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpyaugment_ptr(double** %dst, double** %dstp, double** %src, double** %srcp, i64 %n) local_unnamed_addr #0 {
entry:
  %0 = tail call double (...) @__enzyme_autodiff.f64(void (double**, double**, i64)* nonnull @memcpyaugment_ptr, double** %dst, double** %dstp, double** %src, double** %srcp, i64 %n) #3
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }
attributes #3 = { nounwind }

; CHECK: define internal {{(dso_local )?}}void @diffememcpy_ptr(double** nocapture %dst, double** nocapture %"dst'", double** nocapture readonly %src, double** nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'ipc" = bitcast double** %"dst'" to i8*
; CHECK-NEXT:   %0 = bitcast double** %dst to i8*
; CHECK-NEXT:   %"'ipc1" = bitcast double** %"src'" to i8*
; CHECK-NEXT:   %1 = bitcast double** %src to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %"'ipc", i8* align 1 %"'ipc1", i64 %num, i1 false)
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @diffememcpyaugment_ptr(double** nocapture %dst, double** nocapture %"dst'", double** nocapture readonly %src, double** nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @augmented_submemcpy_ptr(double** %dst, double** %"dst'", double** %src, double** %"src'", i64 %num)
; CHECK-NEXT:   store double* null, double** %"dst'"
; CHECK-NEXT:   store double* null, double** %dst
; CHECK-NEXT:   call void @diffesubmemcpy_ptr(double** %dst, double** %"dst'", double** %src, double** %"src'", i64 %num)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @augmented_submemcpy_ptr(double** nocapture %dst, double** nocapture %"dst'", double** nocapture readonly %src, double** nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'ipc" = bitcast double** %"dst'" to i8*
; CHECK-NEXT:   %0 = bitcast double** %dst to i8*
; CHECK-NEXT:   %"'ipc1" = bitcast double** %"src'" to i8*
; CHECK-NEXT:   %1 = bitcast double** %src to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %"'ipc", i8* align 1 %"'ipc1", i64 %num, i1 false)
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}void @diffesubmemcpy_ptr(double** nocapture %dst, double** nocapture %"dst'", double** nocapture readonly %src, double** nocapture %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

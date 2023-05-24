; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,dce,instcombine)" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: nounwind
declare void @__enzyme_autodiff.f64(...)

; Function Attrs: nounwind uwtable
define dso_local void @memcpy_float(double* nocapture %dst, double* nocapture readonly %src, i64 %num) #0 {
entry:
  %0 = bitcast double* %dst to i8*
  %1 = bitcast double* %src to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define dso_local void @dmemcpy_float(double* %dst, double* %dstp1, double* %dstp2, double* %dstp3, double* %src, double* %srcp1, double* %dsrcp2, double* %dsrcp3, i64 %n) local_unnamed_addr #0 {
entry:
  tail call void (...) @__enzyme_autodiff.f64(void (double*, double*, i64)* nonnull @memcpy_float, metadata !"enzyme_width", i64 3, double* %dst, double* %dstp1, double* %dstp2, double* %dstp3, metadata !"enzyme_const", double* %src, i64 %n) #3
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }

; CHECK: define internal void @diffe3memcpy_float(double* nocapture %dst, [3 x double*] %"dst'", double* nocapture readonly %src, i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [3 x double*] %"dst'", 0
; CHECK-NEXT:   %"'ipc" = bitcast double* %0 to i8*
; CHECK-NEXT:   %1 = extractvalue [3 x double*] %"dst'", 1
; CHECK-NEXT:   %"'ipc1" = bitcast double* %1 to i8*
; CHECK-NEXT:   %2 = extractvalue [3 x double*] %"dst'", 2
; CHECK-NEXT:   %"'ipc2" = bitcast double* %2 to i8*
; CHECK-NEXT:   %3 = bitcast double* %dst to i8*
; CHECK-NEXT:   %4 = bitcast double* %src to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %3, i8* align 1 %4, i64 %num, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* align 1 %"'ipc", i8 0, i64 %num, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* align 1 %"'ipc1", i8 0, i64 %num, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* align 1 %"'ipc2", i8 0, i64 %num, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

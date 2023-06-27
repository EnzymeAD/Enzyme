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
  tail call void (...) @__enzyme_autodiff.f64(void (double*, double*, i64)* nonnull @memcpy_float, metadata !"enzyme_width", i64 3, metadata !"enzyme_const", double* %dst, double* %src, double* %srcp1, double* %dsrcp2, double* %dsrcp3, i64 %n) #3
  ret void
}

attributes #0 = { nounwind uwtable }
attributes #1 = { argmemonly nounwind }
attributes #2 = { noinline nounwind uwtable }


; CHECK: define internal void @diffe3memcpy_float(double* nocapture %dst, double* nocapture readonly %src, [3 x double*] %"src'", i64 %num)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast double* %dst to i8*
; CHECK-NEXT:   %1 = bitcast double* %src to i8*
; CHECK-NEXT:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %0, i8* align 1 %1, i64 %num, i1 false)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

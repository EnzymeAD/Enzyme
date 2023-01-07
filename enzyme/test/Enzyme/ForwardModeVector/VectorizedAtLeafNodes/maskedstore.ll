; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s

declare <6 x double> @__enzyme_fwddiff.f64(...) 

declare void @llvm.masked.store.v2f64.p0v2f64  (<2 x double>, <2 x double>*, i32, <2 x i1>)

; Function Attrs: nounwind uwtable
define dso_local void @loader(<2 x double>* %ptr, <2 x i1> %mask, <2 x double> %val) {
entry:
  call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 16, <2 x i1> %mask)
  ret void
}


; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define <6 x double> @dloader(i8* %ptr, <3 x i8>* %dptr, <2 x i1> %mask, <2 x double> %other, <6 x double> %dother) {
entry:
  %res = tail call <6 x double> (...) @__enzyme_fwddiff.f64(void (<2 x double>*, <2 x i1>, <2 x double>)* @loader, metadata !"enzyme_width", i64 3, i8* %ptr, <3 x i8>* %dptr, <2 x i1> %mask, <2 x double> %other, <6 x double> %dother)
  ret <6 x double> %res
}


; CHECK: define internal void @fwddiffe3loader(<2 x double>* %ptr, <6 x double>* %"ptr'", <2 x i1> %mask, <2 x double> %val, <6 x double> %"val'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 16, <2 x i1> %mask) #2, !alias.scope !0, !noalias !3
; CHECK-NEXT:   %0 = bitcast <6 x double>* %"ptr'" to <2 x double>*
; CHECK-NEXT:   %"val'.subvector.0" = shufflevector <6 x double> %"val'", <6 x double> undef, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %"val'.subvector.0", <2 x double>* %0, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   %1 = getelementptr inbounds <6 x double>, <6 x double>* %"ptr'", i64 0, i64 2
; CHECK-NEXT:   %2 = bitcast double* %1 to <2 x double>*
; CHECK-NEXT:   %"val'.subvector.1" = shufflevector <6 x double> %"val'", <6 x double> undef, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %"val'.subvector.1", <2 x double>* %2, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   %3 = getelementptr inbounds <6 x double>, <6 x double>* %"ptr'", i64 0, i64 4
; CHECK-NEXT:   %4 = bitcast double* %3 to <2 x double>*
; CHECK-NEXT:   %"val'.subvector.2" = shufflevector <6 x double> %"val'", <6 x double> undef, <2 x i32> <i32 4, i32 5>
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %"val'.subvector.2", <2 x double>* %4, i32 16, <2 x i1> %mask)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
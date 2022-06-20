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
; CHECK-NEXT:   call void @llvm.masked.store.v2f64.p0v2f64(<2 x double> %val, <2 x double>* %ptr, i32 16, <2 x i1> %mask) #2
; CHECK-NEXT:   %mask.vecsplat = shufflevector <2 x i1> %mask, <2 x i1> poison, <6 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
; CHECK-NEXT:   call void @llvm.masked.store.v6f64.p0v6f64(<6 x double> %"val'", <6 x double>* %"ptr'", i32 16, <6 x i1> %mask.vecsplat)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
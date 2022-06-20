; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-vectorize-at-leaf-nodes -mem2reg -simplifycfg -dce -instcombine -S | FileCheck %s

declare <6 x double> @__enzyme_fwddiff.f64(...) 

declare <2 x double>  @llvm.masked.load.v2f64.p0v2f64  (<2 x double>*, i32, <2 x i1>, <2 x double>)

; Function Attrs: nounwind uwtable
define dso_local <2 x double> @loader(<2 x double>* %ptr, <2 x i1> %mask, <2 x double> %other) {
entry:
  %res = call <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %ptr, i32 16, <2 x i1> %mask, <2 x double> %other)
  ret <2 x double> %res
}


; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

; Function Attrs: nounwind uwtable
define <6 x double> @dloader(i8* %ptr, <3 x i8>* %dptr, <2 x i1> %mask, <2 x double> %other,  <6 x double> %dother) {
entry:
  %res = tail call <6 x double> (...) @__enzyme_fwddiff.f64(<2 x double> (<2 x double>*, <2 x i1>, <2 x double>)* @loader, metadata !"enzyme_width", i64 3, i8* %ptr, <3 x i8>* %dptr, <2 x i1> %mask, <2 x double> %other, <6 x double> %dother)
  ret <6 x double> %res
}


; CHECK: define internal <6 x double> @fwddiffe3loader(<2 x double>* %ptr, <6 x double>* %"ptr'", <2 x i1> %mask, <2 x double> %other, <6 x double> %"other'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mask.vecsplat = shufflevector <2 x i1> %mask, <2 x i1> poison, <6 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>
; CHECK-NEXT:   %0 = call fast <6 x double> @llvm.masked.load.v6f64.p0v6f64(<6 x double>* %"ptr'", i32 16, <6 x i1> %mask.vecsplat, <6 x double> %"other'")
; CHECK-NEXT:   ret <6 x double> %0
; CHECK-NEXT: }
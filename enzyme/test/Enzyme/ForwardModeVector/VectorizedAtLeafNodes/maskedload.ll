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


; CHECK: define internal <6 x double> @fwddiffe3loader(<2 x double>* %ptr, <6 x double>* %"ptr'", <2 x i1> %mask, <2 x double> %other, <6 x double> %"other'") #2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = bitcast <6 x double>* %"ptr'" to <2 x double>*
; CHECK-NEXT:   %"other'.subvector.0" = shufflevector <6 x double> %"other'", <6 x double> poison, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT:   %1 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* %0, i32 16, <2 x i1> %mask, <2 x double> %"other'.subvector.0")
; CHECK-NEXT:   %2 = getelementptr inbounds <6 x double>, <6 x double>* %"ptr'", i64 0, i64 2
; CHECK-NEXT:   %3 = bitcast double* %2 to <2 x double>*
; CHECK-NEXT:   %"other'.subvector.1" = shufflevector <6 x double> %"other'", <6 x double> poison, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT:   %4 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* nonnull %3, i32 16, <2 x i1> %mask, <2 x double> %"other'.subvector.1")
; CHECK-NEXT:   %.vecconcat = shufflevector <2 x double> %1, <2 x double> %4, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:   %5 = getelementptr inbounds <6 x double>, <6 x double>* %"ptr'", i64 0, i64 4
; CHECK-NEXT:   %6 = bitcast double* %5 to <2 x double>*
; CHECK-NEXT:   %"other'.subvector.2" = shufflevector <6 x double> %"other'", <6 x double> poison, <2 x i32> <i32 4, i32 5>
; CHECK-NEXT:   %7 = call fast <2 x double> @llvm.masked.load.v2f64.p0v2f64(<2 x double>* nonnull %6, i32 16, <2 x i1> %mask, <2 x double> %"other'.subvector.2")
; CHECK-NEXT:   %.vecpad = shufflevector <2 x double> %7, <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CHECK-NEXT:   %.vecpad.vecconcat = shufflevector <4 x double> %.vecconcat, <4 x double> %.vecpad, <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
; CHECK-NEXT:   ret <6 x double> %.vecpad.vecconcat
; CHECK-NEXT: }
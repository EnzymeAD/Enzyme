; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_fwddiff(...)

declare void @cblas_dtrsm(i32, i32, i32, i32, i32, i32, i32, double, double*, i32, double*, i32)

define void @f(i32 %layout, double* %A, double* %B) {
entry:
  call void @cblas_dtrsm(i32 %layout, i32 141, i32 121, i32 111, i32 131, i32 4, i32 3, double 2.000000e+00, double* %A, i32 4, double* %B, i32 4)
  ret void
}

define void @active(i32 %layout, double* %A, double* %dA, double* %B, double* %dB) {
entry:
  call void (...) @__enzyme_fwddiff(void (i32, double*, double*)* @f, metadata !"enzyme_const", i32 %layout, metadata !"enzyme_dup", double* %A, double* %dA, metadata !"enzyme_dup", double* %B, double* %dB)
  ret void
}

; CHECK-LABEL: define internal void @fwddiffef(
; CHECK: call void @cblas_dtrsm
; CHECK: icmp eq i32 %layout, 101
; CHECK: __enzyme_dmemcpy_double_mat_8_32_32_zero
; CHECK: call void @cblas_dtrsm
; CHECK-NOT: call void bitcast
; CHECK: define internal void @__enzyme_dmemcpy_double_mat_8_32_32_zero(i8 %uplo, i32 %layout,

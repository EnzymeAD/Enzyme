; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-strict-aliasing=0 -enzyme-loose-types=0 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Extents = type { [2 x i64], i32, [4 x i8] }

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare double @__enzyme_autodiff(...)

define double @kernel(double %x, ptr %active_view) {
entry:
  %ext = alloca %struct.Extents, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %ext, i8 0, i64 24, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %active_view, ptr align 8 %ext, i64 24, i1 false)
  %val_ptr = getelementptr inbounds i8, ptr %active_view, i64 24
  %val = load double, ptr %val_ptr, align 8
  %prod = fmul double %x, %val
  ret double %prod
}

define double @grad_kernel(double %x, ptr %view, ptr %dview) {
entry:
  %res = call double (...) @__enzyme_autodiff(ptr @kernel, double %x, metadata !"enzyme_dup", ptr %view, ptr %dview)
  ret double %res
}

; CHECK-LABEL: define internal { double } @diffekernel(
; CHECK: fmul fast double
; CHECK: fadd fast double
; CHECK: ret { double }

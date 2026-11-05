; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.TrackedBuf = type { ptr, i64 }

declare void @__enzyme_autodiff(...)

define void @tagged_ptr_kernel(ptr %data, ptr %ctrl, i1 %untracked) {
entry:
  %raw_int = ptrtoint ptr %ctrl to i64
  %bits = select i1 %untracked, i64 1, i64 %raw_int
  %slot = alloca %struct.TrackedBuf, align 8
  %p0 = getelementptr inbounds %struct.TrackedBuf, ptr %slot, i64 0, i32 0
  store ptr %data, ptr %p0, align 8
  %p1 = getelementptr inbounds %struct.TrackedBuf, ptr %slot, i64 0, i32 1
  store i64 %bits, ptr %p1, align 8
  %buf = load ptr, ptr %p0, align 8
  %x = load double, ptr %buf, align 8
  %sq = fmul double %x, %x
  store double %sq, ptr %buf, align 8
  ret void
}

define void @caller(ptr %data, ptr %ddata, ptr %ctrl, i1 %untracked) {
entry:
  call void (...) @__enzyme_autodiff(ptr @tagged_ptr_kernel, metadata !"enzyme_dup", ptr %data, ptr %ddata, metadata !"enzyme_const", ptr %ctrl, metadata !"enzyme_const", i1 %untracked)
  ret void
}

; CHECK-LABEL: define internal void @diffetagged_ptr_kernel(
; CHECK-NOT: MixedActivityError
; CHECK: fadd fast double

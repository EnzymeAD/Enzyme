; RUN: %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s
; REQUIRES: amdgpu-registered-target

; AMDGPU device code has no realloc: the cache of a loop whose trip count is not
; known on entry grows by malloc and a copy, as on NVPTX.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define double @f(ptr %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %acc.next, %loop ]
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %p
  store double 0.000000e+00, ptr %p
  %sq = fmul double %v, %v
  %acc.next = fadd double %acc, %sq
  %i.next = add nuw i64 %i, 1
  %c = fcmp ogt double %acc.next, 1.000000e+01
  br i1 %c, label %exit, label %loop

exit:
  ret double %acc.next
}

define void @df(ptr %x, ptr %dx, i64 %n) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, i64 %n)
  ret void
}

declare double @__enzyme_autodiff(...)

; CHECK-NOT: @realloc(
; CHECK: define internal ptr @__enzyme_exponentialallocation(
; CHECK-NOT: @realloc(
; CHECK: call {{.*}}ptr @malloc(
; CHECK-NOT: @realloc(
; CHECK: call void @llvm.memcpy
; CHECK-NOT: @realloc(

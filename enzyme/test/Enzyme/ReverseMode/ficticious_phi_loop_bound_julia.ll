; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; The trip count of %loop comes from a size loaded out of a GC allocation. The
; loop has no loop context yet when the min-cut cache analysis runs, so the load
; used to be classified as an unnecessary intermediate, the allocation as not
; needed in the reverse pass, and replaced by a placeholder that the reverse
; pass still used to expand the trip count ("Illegal replace ficticious phi").
; Julia emits this pattern under --check-bounds=yes, where filling a new array
; bounds-checks against sizes loaded back from it.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-unknown-linux-gnu"

declare ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64)

define fastcc ptr addrspace(10) @julia_loop_bound() {
top:
  %arr = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 0)
  %arr11 = addrspacecast ptr addrspace(10) %arr to ptr addrspace(11)
  %size = load i64, ptr addrspace(11) %arr11, align 8
  %len = mul i64 %size, 1
  br label %loop

loop:
  %i = phi i64 [ 0, %top ], [ %i.next, %loop ]
  %i.next = add i64 %i, 1
  %done = icmp eq i64 %i, %len
  br i1 %done, label %exit, label %loop

exit:
  ret ptr addrspace(10) %arr
}

declare ptr @__enzyme_virtualreverse(...)

define void @test_enzyme() {
  %z = call ptr (...) @__enzyme_virtualreverse(ptr @julia_loop_bound)
  ret void
}

; The loaded size is saved on the tape for the trip count of the reverse loop.

; CHECK: define internal fastcc { ptr, ptr addrspace(10), ptr addrspace(10) } @augmented_julia_loop_bound()
; CHECK:   %size = load i64, ptr addrspace(11) %arr11
; CHECK:   store i64 %size, ptr

; CHECK: define internal fastcc void @diffejulia_loop_bound(ptr %tapeArg)
; CHECK:   %truetape = load { ptr addrspace(10), i64 }, ptr %tapeArg
; CHECK:   %size = extractvalue { ptr addrspace(10), i64 } %truetape, 1
; CHECK-NOT: %arr
; CHECK: }

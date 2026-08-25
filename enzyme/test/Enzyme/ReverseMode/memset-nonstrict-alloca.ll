; Zeroing memset of a stack slot whose element type TypeAnalysis cannot determine.
;
; Reduced with llvm-reduce from rustc output. Frontends for untyped-memory languages --
; notably Rust, whose allocas are byte arrays and which must therefore run with
; -enzyme-strict-aliasing=0 -- emit this shape for a tiled reduction: an accumulator on
; the stack, memset to zero at the top of each iteration, accumulated into, then read
; back out.
;
; visitMemSetCommon splits on whether the element type is known. With a null secret type
; it emitted the shadow memset only into the forward sweep, so the shadow accumulator
; was never re-zeroed between reverse iterations and any gradient accumulated into it
; grew with the trip count. A zeroing memset kills every prior value in the region, so
; no adjoint can flow past it and the shadow must be zeroed in the reverse sweep too.
;
; Written with opaque pointers, so it is restricted to LLVM 16 and later.

; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-strict-aliasing=0 -passes="enzyme" -S | FileCheck %s; fi

declare void @__enzyme_autodiff(...)

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

define void @f(ptr %x, i64 %n) {
entry:
  %acc = alloca [32 x i8], align 4
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %latch ]
  call void @llvm.memset.p0.i64(ptr align 4 %acc, i8 0, i64 32, i1 false)
  %v = load float, ptr %x, align 4
  store float %v, ptr %acc, align 4
  %a = load float, ptr %acc, align 4
  store float %a, ptr %x, align 4
  br label %latch

latch:
  %inext = add nuw i64 %i, 1
  %done = icmp eq i64 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @df(ptr %x, ptr %dx, i64 %n) {
  call void (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", i64 %n)
  ret void
}

; The shadow accumulator must be re-zeroed in the reverse sweep. Before the fix no
; memset was emitted into the invert blocks at all.

; CHECK: define internal void @diffef
; CHECK: invertloop:
; CHECK: call void @llvm.memset.p0.i64(ptr {{.*}}%"acc'ipa", i8 0, i64 32, i1 false)

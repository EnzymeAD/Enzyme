; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-strict-aliasing=0 -passes="enzyme" -S | FileCheck %s; fi

; Zeroing memset, inside a loop, of a stack slot whose element type
; TypeAnalysis cannot determine (https://github.com/EnzymeAD/Enzyme/issues/3174).
;
; The memset used to be treated as zeroing never-written memory, because the
; backwards walk looking for a prior write stopped at the top of the loop body
; without ever reaching the alloca. The shadow was then only zeroed in the
; forward sweep, so adjoints accumulated into it leaked between reverse
; iterations. A zeroing memset kills every prior value in the region, so no
; adjoint can flow past it and the shadow must be zeroed in the reverse sweep
; too.

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

; CHECK: define internal void @diffef(ptr{{.*}} %x, ptr{{.*}} %"x'", i64 %n)
; CHECK: loop:
; CHECK:   call void @llvm.memset.p0.i64(ptr align 4 %acc, i8 0, i64 32, i1 false)
; CHECK-NEXT:   call void @llvm.memset.p0.i64(ptr align 4 %"acc'ipa", i8 0, i64 32, i1 false)

; CHECK: invertloop:
; CHECK:   call void @llvm.memset.p0.i64(ptr align 4 %"acc'ipa", i8 0, i64 32, i1 false)
; CHECK: incinvertloop:

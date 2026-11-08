; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s; fi

declare void @__enzyme_autodiff(ptr, ...)
declare void @llvm.assume(i1)

define internal void @scale_square(ptr noalias %out, ptr %inp, i64 %n) noinline {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %inp.ptr = getelementptr inbounds float, ptr %inp, i64 %i
  %x = load float, ptr %inp.ptr, align 4
  %sq = fmul float %x, %x
  %out.ptr = getelementptr inbounds float, ptr %out, i64 %i
  store float %sq, ptr %out.ptr, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define float @step(ptr noalias %out, ptr noalias readonly %inp, ptr noalias readonly %n_ptr) {
entry:
  %n = load i64, ptr %n_ptr, align 8
  call void @scale_square(ptr %out, ptr %inp, i64 %n)
  %cmp = icmp sgt i64 %n, 0
  call void @llvm.assume(i1 %cmp)
  %res = load float, ptr %out, align 4
  ret float %res
}

define void @caller(ptr %out, ptr %dout, ptr %inp, ptr %dinp, ptr %n_ptr) {
entry:
  call void (ptr, ...) @__enzyme_autodiff(ptr @step, ptr %out, ptr %dout, ptr %inp, ptr %dinp, metadata !"enzyme_const", ptr %n_ptr)
  ret void
}

; Verify that @llvm.assume after @scale_square does not mark %inp or %n as overwritten,
; so @diffescale_square needs no @malloc cache and reloads %x directly from %inp.
; CHECK: define internal void @diffescale_square(ptr {{.*}}%out, ptr {{.*}}%"out'", ptr {{.*}}%inp, ptr {{.*}}%"inp'", i64 %n)
; CHECK-NOT: @malloc(
; CHECK: invertloop:
; CHECK:   %inp.ptr_unwrap = getelementptr inbounds float, ptr %inp, i64 %"iv'ac.0"
; CHECK-NEXT:   %x_unwrap = load float, ptr %inp.ptr_unwrap, align 4

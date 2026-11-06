; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; @f is differentiated in combined mode. Its `noalias readonly` %w reaches
; @kernel through a ptrtoint/inttoptr round trip and is then passed to an
; opaque call with no memory attributes. Nothing in the invocation can write
; the memory %w points to, yet @kernel's loads from it are taped today.

declare void @__enzyme_autodiff(...)
declare void @opaque(ptr) nofree

define internal void @kernel(ptr noalias readonly %w, ptr noalias %out, double %x, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %pw = getelementptr inbounds double, ptr %w, i64 %i
  %wv = load double, ptr %pw
  %m = fmul double %wv, %x
  %po = getelementptr inbounds double, ptr %out, i64 %i
  store double %m, ptr %po
  %inc = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @f(ptr noalias readonly %w, ptr noalias %out, double %x, i64 %n) {
entry:
  %wi = ptrtoint ptr %w to i64
  %w2 = inttoptr i64 %wi to ptr
  call void @kernel(ptr %w2, ptr %out, double %x, i64 %n)
  call void @opaque(ptr %w)
  ret void
}

define double @test(ptr %w, ptr %out, ptr %dout, double %x, i64 %n) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_const", ptr %w, metadata !"enzyme_dup", ptr %out, ptr %dout, double %x, i64 %n)
  ret double %r
}

; The augmented form tapes the loads from %w and hands the tape to the reverse.
; CHECK: define internal ptr @augmented_kernel(
; CHECK: %wv_malloccache = tail call noalias nonnull ptr @malloc(
; CHECK: }

; CHECK: define internal { double } @diffekernel(
; CHECK-SAME: ptr %tapeArg)

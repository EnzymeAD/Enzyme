; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

declare noalias ptr @malloc(i64)
declare double @__enzyme_autodiff(...)

define double @f(double %x, i64 %n) {
entry:
  %init = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) ptr @malloc(i64 8)
  store double %x, ptr %init, align 8
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %cur = phi ptr [ %init, %entry ], [ %next, %loop ]
  %val = load double, ptr %cur, align 8
  %next = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) ptr @malloc(i64 8)
  %sq = fmul double %val, %val
  store double %sq, ptr %next, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp eq i64 %iv.next, %n
  br i1 %cmp, label %exit, label %loop

exit:
  %res = load double, ptr %next, align 8
  ret double %res
}

define double @df(double %x, i64 %n) {
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x, i64 %n)
  ret double %r
}

; CHECK-LABEL: define internal { double } @diffef(double %x, i64 %n, double %differeturn)
; CHECK: entry:
; CHECK: %"next'mi_malloccache" = tail call noalias nonnull ptr @malloc(i64 %mallocsize)
; CHECK: store ptr %"next'mi_malloccache", ptr %"next'mi_cache", align 8
; CHECK: %next_malloccache = tail call noalias nonnull ptr @malloc(i64 %mallocsize{{.*}})
; CHECK: store ptr %next_malloccache, ptr %next_cache, align 8
; CHECK: loop:
; CHECK: %[[IV:.+]] = phi i64
; CHECK: %[[NC:.+]] = load ptr, ptr %next_cache, align 8
; CHECK: %[[NEXT_IPC:.+]] = getelementptr inbounds ptr, ptr %[[NC]], i64 %[[IV]]
; CHECK: store ptr %next, ptr %[[NEXT_IPC]], align 8
; CHECK: %[[NMC:.+]] = load ptr, ptr %"next'mi_cache", align 8
; CHECK: %[[NEXTMI_IPC:.+]] = getelementptr inbounds ptr, ptr %[[NMC]], i64 %[[IV]]
; CHECK: store ptr %"next'mi", ptr %[[NEXTMI_IPC]], align 8
; CHECK: invertloop:


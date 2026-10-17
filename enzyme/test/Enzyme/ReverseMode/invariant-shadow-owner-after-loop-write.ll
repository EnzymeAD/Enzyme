; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; A result header is selected before the loop. Its owner pointer is read on
; each iteration, and the selected header is written to the slot afterwards.
; The owner pointer has one value for the whole loop even if the slot may alias
; the selected header, so combined reverse needs at most one cached copy.

declare double @__enzyme_autodiff(...)
declare void @bounds_error(i64) noreturn

define double @loss(ptr %slot, ptr %alternate, ptr %q, i64 %n, i1 %reuse) {
entry:
  %old = load ptr, ptr %slot, align 8
  %selected = select i1 %reuse, ptr %old, ptr %alternate
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %acc = phi double [ 0.0, %entry ], [ %acc.next, %body ]
  %owner = load ptr, ptr %selected, align 8
  %valid = icmp ult i64 %i, %n
  br i1 %valid, label %body, label %oob

oob:
  call void @bounds_error(i64 %i)
  unreachable

body:
  %element = getelementptr inbounds double, ptr %owner, i64 %i
  %value = load double, ptr %element, align 8
  %qvalue = load double, ptr %q, align 8
  %term = fmul double %value, %qvalue
  %acc.next = fadd double %acc, %term
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  store ptr %selected, ptr %slot, align 8
  ret double %acc.next
}

define double @driver(ptr %slot, ptr %dslot, ptr %alternate,
                      ptr %dalternate, ptr %q, ptr %dq, i64 %n, i1 %reuse) {
entry:
  %gradient = call double (...) @__enzyme_autodiff(
      ptr @loss, ptr %slot, ptr %dslot, ptr %alternate, ptr %dalternate,
      ptr %q, ptr %dq, i64 %n, i1 %reuse)
  ret double %gradient
}

; CHECK-LABEL: define internal void @diffeloss(
; CHECK: %"owner'ipl_cache" = alloca ptr
; CHECK-NOT: %"owner'ipl_malloccache"
; CHECK: %"owner'ipl" = load ptr
; CHECK-NEXT: store ptr %"owner'ipl", ptr %"owner'ipl_cache"

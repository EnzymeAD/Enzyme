; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; A callee that Enzyme's own analysis marked enzyme_ReadOnlyOrThrow only writes
; pre-existing memory on paths that then throw. LLVM knows nothing about it
; (it calls an opaque function, inactive for Enzyme), and the buffer it receives is a phi of an
; allocation and null, so alias analysis cannot rule out a write either. The
; loads of q in the loop before the call must nevertheless not be cached: from
; a plain call a throw leaves f, and no reverse pass runs.

declare void @opaque(ptr) #1

define double @tail(i64 %k, ptr %q) #0 {
entry:
  call void @opaque(ptr %q)
  %v = load double, ptr %q, align 8
  ret double %v
}

define void @f(i64 %k, ptr noalias %x, ptr noalias %out) {
entry:
  %empty = icmp eq i64 %k, 0
  br i1 %empty, label %pre, label %alloc

alloc:
  %bytes = shl i64 %k, 3
  %m = call noalias ptr @malloc(i64 %bytes)
  br label %pre

pre:
  %q = phi ptr [ null, %entry ], [ %m, %alloc ]
  br label %fill

fill:
  %i = phi i64 [ 0, %pre ], [ %i.next, %fill ]
  %x.p = getelementptr inbounds double, ptr %x, i64 %i
  %xv = load double, ptr %x.p, align 8
  %e = call double @llvm.exp.f64(double %xv)
  %q.p = getelementptr inbounds double, ptr %q, i64 %i
  store double %e, ptr %q.p, align 8
  %i.next = add nuw nsw i64 %i, 1
  %fill.cond = icmp ult i64 %i.next, %k
  br i1 %fill.cond, label %fill, label %use

use:
  %j = phi i64 [ 0, %fill ], [ %j.next, %use ]
  %acc = phi double [ 0.0, %fill ], [ %acc.next, %use ]
  %q.p2 = getelementptr inbounds double, ptr %q, i64 %j
  %qv = load double, ptr %q.p2, align 8
  %x.p2 = getelementptr inbounds double, ptr %x, i64 %j
  %xv2 = load double, ptr %x.p2, align 8
  %prod = fmul double %qv, %xv2
  %acc.next = fadd double %acc, %prod
  %j.next = add nuw nsw i64 %j, 1
  %use.cond = icmp ult i64 %j.next, %k
  br i1 %use.cond, label %use, label %done

done:
  %t = call double @tail(i64 %k, ptr %q)
  %r = fadd double %acc.next, %t
  store double %r, ptr %out, align 8
  call void @free(ptr %q)
  ret void
}

declare double @llvm.exp.f64(double)
declare noalias ptr @malloc(i64)
declare void @free(ptr)

define void @df(i64 %k, ptr %x, ptr %dx, ptr %out, ptr %dout) {
  call void (ptr, ...) @__enzyme_autodiff(ptr @f, i64 %k, ptr %x, ptr %dx, ptr %out, ptr %dout)
  ret void
}

declare void @__enzyme_autodiff(ptr, ...)

attributes #0 = { noinline "enzyme_ReadOnlyOrThrow" }
attributes #1 = { nofree "enzyme_inactive" }

; The reverse of the use loop needs q[j] and x[j]. x is never written, and q is
; not written after the fill loop, so both are reloaded rather than taped.
; CHECK: define internal void @diffef(
; CHECK-NOT: qv_cache
; CHECK-NOT: mallocsize
; CHECK: %qv_unwrap = load double, ptr %q.p2_unwrap
; CHECK-NOT: qv_cache
; CHECK-NOT: mallocsize
; CHECK: define internal { double } @diffetail(

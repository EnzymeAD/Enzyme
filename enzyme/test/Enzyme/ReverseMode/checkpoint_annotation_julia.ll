; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi

; A Julia loop annotated with Expr(:loopinfo, (Symbol("enzyme.checkpoint"),
; :revolve, 3)) over an array %x. The array's data pointer is loaded from the
; array (and made a pointer the array roots with julia.gc_loaded); the hint
; before the loop loads it from the array as well, which the loop does not
; write, so the region is the hint's. The loop leaves to throw when the
; array is too short, and allocates the exception there: those blocks are
; part of the step, and the exception is not part of a snapshot. The derived
; pointers the step uses are made again in the step from the array.

declare void @__enzyme_ptr_size_hint(ptr, i64)
declare ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10), ptr) memory(none)
declare noalias ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
declare void @ijl_throw(ptr addrspace(12)) noreturn
declare ptr @julia.get_pgcstack()

define void @run(ptr addrspace(10) %x, i64 %n) {
entry:
  %pgcstack = call ptr @julia.get_pgcstack()
  %a = addrspacecast ptr addrspace(10) %x to ptr addrspace(11)
  %data0 = load ptr, ptr addrspace(11) %a
  %lenp = getelementptr inbounds i8, ptr addrspace(11) %a, i64 16
  %len = load i64, ptr addrspace(11) %lenp
  %bytes = shl i64 %len, 3
  call void @__enzyme_ptr_size_hint(ptr %data0, i64 %bytes)
  %go = icmp sgt i64 %n, 0
  br i1 %go, label %pre, label %exit

pre:
  %data = load ptr, ptr addrspace(11) %a
  %memp = getelementptr inbounds i8, ptr addrspace(11) %a, i64 8
  %mem = load ptr addrspace(10), ptr addrspace(11) %memp
  %d = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %mem, ptr %data)
  br label %loop

loop:
  %i = phi i64 [ 0, %pre ], [ %i.next, %latch ]
  %short = icmp slt i64 %len, 1
  br i1 %short, label %throw, label %latch

latch:
  %v = load double, ptr addrspace(13) %d
  %w = fmul double %v, %v
  store double %w, ptr addrspace(13) %d
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

throw:
  %ptls = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %e = call noalias ptr addrspace(10) @julia.gc_alloc_obj(ptr %ptls, i64 8, ptr addrspace(10) null)
  %ea = addrspacecast ptr addrspace(10) %e to ptr addrspace(11)
  store i64 %len, ptr addrspace(11) %ea
  %ec = addrspacecast ptr addrspace(10) %e to ptr addrspace(12)
  call void @ijl_throw(ptr addrspace(12) %ec)
  unreachable

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"enzyme.checkpoint", !"revolve", i64 3}

; The region is the hint's pointer and size; the step gets the array object
; and the raw data pointer, and makes the derived pointer itself.
; CHECK: define void @run(
; CHECK: call void @enzyme.ckpt.for.run.ckpt.step(i64 0, i64 %n, ptr %0, ptr %ckpt.config, ptr %data0, i64 %bytes, i64 %len, i64 %n, ptr %pgcstack, ptr addrspace(10) %mem, ptr %data)
; CHECK-NOT: __enzyme_ptr_size_hint(

; CHECK: define internal void @run.ckpt.step(i64 %k, i64 %len, i64 %n, ptr %pgcstack, ptr addrspace(10) %mem, ptr %data)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %d = call ptr addrspace(13) @julia.gc_loaded(ptr addrspace(10) %mem, ptr %data)
; CHECK:      latch:
; CHECK:        store double %w, ptr addrspace(13) %d
; CHECK:      throw:
; CHECK:        %e = call noalias ptr addrspace(10) @julia.gc_alloc_obj(
; CHECK:        call void @ijl_throw(
; CHECK-NEXT:   unreachable

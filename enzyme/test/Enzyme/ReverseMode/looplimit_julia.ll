; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-julia-addr-load -S | FileCheck %s; fi

; The trip count of @check's loop is a load from %tmp, a GC allocation the
; reverse pass does not otherwise need. The min-cut cache must see that load as
; a loop bound requirement and cache it; otherwise the split reverse pass
; recomputes the load from an erased %tmp ("Illegal replace ficticious phi").

declare ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))

declare double @__enzyme_autodiff(...)

define double @f(double %x) {
entry:
  %mem = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 8, ptr addrspace(10) null)
  %arr = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 8, ptr addrspace(10) null)
  %arr11 = addrspacecast ptr addrspace(10) %arr to ptr addrspace(11)
  store ptr addrspace(10) %mem, ptr addrspace(11) %arr11
  call void @check(ptr addrspace(10) %arr)
  %v = load volatile double, ptr null
  ret double 0.0
}

define void @check(ptr addrspace(10) %w) {
entry:
  %w11 = addrspacecast ptr addrspace(10) %w to ptr addrspace(11)
  %wlen = load i64, ptr addrspace(11) %w11
  %tmp = call ptr addrspace(10) @julia.gc_alloc_obj(ptr null, i64 8, ptr addrspace(10) null)
  %tmp11 = addrspacecast ptr addrspace(10) %tmp to ptr addrspace(11)
  store ptr addrspace(10) %tmp, ptr addrspace(11) null
  %n = load i64, ptr addrspace(11) %tmp11
  %lim = add i64 %n, 0
  br label %loop

loop:
  %i = phi i64 [ %inc, %loop ], [ 0, %entry ]
  %done = icmp eq i64 %lim, %i
  %inc = add i64 %i, 1
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define double @driver(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @f, double %x)
  ret double %r
}

; CHECK: define internal i64 @augmented_check(
; CHECK:   %n = load i64, ptr addrspace(11) %tmp11
; CHECK:   store i64 %n, ptr %0
; CHECK:   %[[r:.+]] = load i64, ptr %0
; CHECK-NEXT:   ret i64 %[[r]]

; CHECK: define internal void @diffecheck(ptr addrspace(10) nocapture readonly %w, ptr addrspace(10) nocapture %"w'", i64 %n)
; CHECK-NOT: @julia.gc_alloc_obj
; CHECK: mergeinvertloop_exit:
; CHECK-NEXT:   store i64 %n, ptr %"iv'ac"

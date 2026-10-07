; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-julia-addr-load -S | FileCheck %s; fi

; A Julia allocation that came from the stack (`enzyme_fromstack`) is moved
; back there after differentiation, both it and its shadow. A phi of pointers
; derived from them cannot be recomputed in the reverse pass, so it is cached
; in a stack slot. That slot, and the loads from it, take the moved pointer's
; address space rather than keeping a tracked one.

declare ptr addrspace(10) @jl_gc_alloc_typed(ptr, i64, ptr addrspace(10))
declare double @__enzyme_autodiff(...)
declare i1 @cond()

define double @g(i1 %c2, double %x) {
entry:
  %al = call noalias nonnull dereferenceable(8) ptr addrspace(10) @jl_gc_alloc_typed(ptr null, i64 8, ptr addrspace(10) addrspacecast (ptr inttoptr (i64 139806792221568 to ptr) to ptr addrspace(10))), !enzyme_fromstack !0
  %ald = addrspacecast ptr addrspace(10) %al to ptr addrspace(11)
  store double %x, ptr addrspace(11) %ald, align 8
  br i1 %c2, label %region, label %exit

region:
  %c1 = call i1 @cond()
  br i1 %c1, label %A, label %B

A:
  %c3 = call i1 @cond()
  br i1 %c3, label %T, label %M

B:
  %c4 = call i1 @cond()
  br i1 %c4, label %M, label %exit

T:
  %p1 = getelementptr inbounds i8, ptr addrspace(10) %al, i64 0
  br label %merge

M:
  %p2 = getelementptr inbounds i8, ptr addrspace(10) %al, i64 0
  br label %merge

merge:
  %phi = phi ptr addrspace(10) [ %p1, %T ], [ %p2, %M ]
  %pd = addrspacecast ptr addrspace(10) %phi to ptr addrspace(11)
  %v = load double, ptr addrspace(11) %pd, align 8
  %sq = fmul double %v, %v
  br label %exit

exit:
  %r = phi double [ %sq, %merge ], [ 0.000000e+00, %entry ], [ 0.000000e+00, %B ]
  ret double %r
}

define double @dg(i1 %c2, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @g, metadata !"enzyme_const", i1 %c2, double %x)
  ret double %r
}

!0 = !{i64 8}

; CHECK: define internal { double } @diffeg(
; CHECK-NOT: jl_gc_alloc_typed
; CHECK:        %al = alloca i8, i64 8, align 8
; CHECK:        %phi_cache = alloca ptr, align 8
; CHECK:        %_cache = alloca ptr, align 8
; CHECK:        %"al'mi" = alloca i8, i64 8, align 8
; CHECK:      merge:
; CHECK-NEXT:   %[[sphi:.+]] = phi ptr [ %"p1'ipg", %T ], [ %"p2'ipg", %M ]
; CHECK-NEXT:   %phi = phi ptr [ %p1, %T ], [ %p2, %M ]
; CHECK-NEXT:   store ptr %[[sphi]], ptr %_cache, align 8, !invariant.group
; CHECK-NEXT:   store ptr %phi, ptr %phi_cache, align 8, !invariant.group
; CHECK:      invertmerge:
; CHECK:        %[[pc:.+]] = load ptr, ptr %phi_cache, align 8, !invariant.group
; CHECK-NEXT:   %v_unwrap = load double, ptr %[[pc]], align 8
; CHECK:        %[[sc:.+]] = load ptr, ptr %_cache, align 8, !invariant.group
; CHECK-NEXT:   load double, ptr %[[sc]], align 8

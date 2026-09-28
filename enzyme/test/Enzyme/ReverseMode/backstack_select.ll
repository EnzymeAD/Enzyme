; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-julia-addr-load -S | FileCheck %s; fi

; An allocation marked enzyme_backstack is moved back to the default address
; space after differentiation, and the users of its Julia pointer rewritten.

declare ptr addrspace(10) @jl_gc_alloc_typed(ptr, i64, ptr addrspace(10))

declare double @__enzyme_autodiff(...)

; A Julia allocation that came from the stack (`enzyme_fromstack`) is moved back
; there, both it and its shadow. The reverse pass recomputes the two-way phi as
; a select between two pointers into the moved object, which moves as well.
define double @f(i1 %c, i1 %c2, double %x) {
entry:
  %al = call noalias nonnull dereferenceable(16) ptr addrspace(10) @jl_gc_alloc_typed(ptr null, i64 16, ptr addrspace(10) addrspacecast (ptr inttoptr (i64 139806792221568 to ptr) to ptr addrspace(10))), !enzyme_fromstack !0
  %ald = addrspacecast ptr addrspace(10) %al to ptr addrspace(11)
  store double %x, ptr addrspace(11) %ald, align 8
  %ald8 = getelementptr inbounds i8, ptr addrspace(11) %ald, i64 8
  %x2 = fmul double %x, %x
  store double %x2, ptr addrspace(11) %ald8, align 8
  br i1 %c2, label %region, label %exit

region:
  br i1 %c, label %left, label %right

left:
  br label %merge

right:
  br label %merge

merge:
  %p = phi ptr addrspace(11) [ %ald, %left ], [ %ald8, %right ]
  %v = load double, ptr addrspace(11) %p, align 8
  %sq = fmul double %v, %v
  br label %exit

exit:
  %r = phi double [ %sq, %merge ], [ 0.000000e+00, %entry ]
  ret double %r
}

define double @df(i1 %c, i1 %c2, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @f, metadata !"enzyme_const", i1 %c, metadata !"enzyme_const", i1 %c2, double %x)
  ret double %r
}

; A select that also chooses a pointer from elsewhere keeps its (derived)
; address space and takes the moved object cast back to it.
define double @g(ptr addrspace(11) %other, i1 %c, double %x) {
entry:
  %a = alloca double, align 8
  %pa = addrspacecast ptr %a to ptr addrspace(10), !enzyme_backstack !1
  %d = addrspacecast ptr addrspace(10) %pa to ptr addrspace(11)
  store double %x, ptr addrspace(11) %d, align 8
  %p = select i1 %c, ptr addrspace(11) %d, ptr addrspace(11) %other
  %v = load double, ptr addrspace(11) %p, align 8
  %sq = fmul double %v, %v
  ret double %sq
}

define double @dg(ptr addrspace(11) %other, i1 %c, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @g, metadata !"enzyme_const", ptr addrspace(11) %other, metadata !"enzyme_const", i1 %c, double %x)
  ret double %r
}

!0 = !{i64 16}
!1 = !{}

; CHECK-LABEL: define internal { double } @diffef(
; CHECK:        %al = alloca i8, i64 16
; CHECK:        %"al'mi" = alloca i8, i64 16
; CHECK:        %p_unwrap = select i1 %c, ptr %al, ptr %ald8
; CHECK:        select i1 %c, ptr %"al'mi", ptr %"ald8'ipg"

; CHECK-LABEL: define internal { double } @diffeg(
; CHECK-NOT:    addrspacecast ptr %a to ptr addrspace(10)
; CHECK:        %[[cast:.+]] = addrspacecast ptr %a to ptr addrspace(11)
; CHECK-NEXT:   %p = select i1 %c, ptr addrspace(11) %[[cast]], ptr addrspace(11) %other

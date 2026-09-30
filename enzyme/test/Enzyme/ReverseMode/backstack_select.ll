; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-julia-addr-load -S | FileCheck %s; fi

declare ptr addrspace(10) @jl_gc_alloc_typed(ptr, i64, ptr addrspace(10))

declare double @__enzyme_autodiff(...)

; A Julia allocation that came from the stack (`enzyme_fromstack`) is moved back
; there after differentiation, both it and its shadow, and the users of its
; Julia pointer rewritten. The reverse pass recomputes the two-way phi as a
; select between two pointers into the moved object, which moves as well.
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

!0 = !{i64 16}

; CHECK-LABEL: define internal { double } @diffef(
; CHECK:        %al = alloca i8, i64 16
; CHECK:        %"al'mi" = alloca i8, i64 16
; CHECK:        %p_unwrap = select i1 %c, ptr %al, ptr %ald8
; CHECK:        select i1 %c, ptr %"al'mi", ptr %"ald8'ipg"


; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-julia-addr-load -S | FileCheck %s; fi

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

define double @linear(i1 %c2, double %x) {
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
  br label %exit

exit:
  %r = phi double [ %v, %merge ], [ 0.000000e+00, %entry ], [ 0.000000e+00, %B ]
  ret double %r
}

define double @dlinear(i1 %c2, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @linear, metadata !"enzyme_const", i1 %c2, double %x)
  ret double %r
}

define double @split_caller(i1 %c2, double %x) {
entry:
  %m = call double @g(i1 %c2, double %x)
  %res = fmul double %m, %m
  ret double %res
}

define double @dsplit_caller(i1 %c2, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @split_caller, metadata !"enzyme_const", i1 %c2, double %x)
  ret double %r
}

!0 = !{i64 8}

; CHECK-LABEL: define internal { double } @diffeg(
; CHECK:   %phi_cache = alloca ptr addrspace(10), align 8
; CHECK:   %[[SHADOW_CACHE:.+]] = alloca ptr addrspace(10), align 8
; CHECK:   %"al'mi" = call noalias nonnull dereferenceable(8) ptr addrspace(10) @jl_gc_alloc_typed(
; CHECK:   %al = call noalias nonnull dereferenceable(8) ptr addrspace(10) @jl_gc_alloc_typed(
; CHECK: merge:
; CHECK:   %[[SHADOW_PHI:.+]] = phi ptr addrspace(10) [ %"p1'ipg", %T ], [ %"p2'ipg", %M ]
; CHECK:   %phi = phi ptr addrspace(10) [ %p1, %T ], [ %p2, %M ]
; CHECK:   store ptr addrspace(10) %[[SHADOW_PHI]], ptr %[[SHADOW_CACHE]], align 8
; CHECK:   store ptr addrspace(10) %phi, ptr %phi_cache, align 8
; CHECK: invertmerge:
; CHECK:   %[[LOADED_PHI:.+]] = load ptr addrspace(10), ptr %phi_cache, align 8
; CHECK:   %[[UNWRAPPED_PTR:.+]] = addrspacecast ptr addrspace(10) %[[LOADED_PHI]] to ptr addrspace(11)
; CHECK:   load double, ptr addrspace(11) %[[UNWRAPPED_PTR]], align 8
; CHECK:   %[[LOADED_SHADOW:.+]] = load ptr addrspace(10), ptr %[[SHADOW_CACHE]], align 8
; CHECK:   %[[UNWRAPPED_SHADOW:.+]] = addrspacecast ptr addrspace(10) %[[LOADED_SHADOW]] to ptr addrspace(11)
; CHECK:   load double, ptr addrspace(11) %[[UNWRAPPED_SHADOW]], align 8

; CHECK-LABEL: define internal { double } @diffelinear(
; CHECK:   %al = alloca i8, i64 8, align 8
; CHECK:   %[[LINEAR_SHADOW_CACHE:.+]] = alloca ptr addrspace(10), align 8
; CHECK:   %"al'mi" = call noalias nonnull dereferenceable(8) ptr addrspace(10) @jl_gc_alloc_typed(
; CHECK:   store double %x, ptr %al, align 8
; CHECK: merge:
; CHECK:   %[[LINEAR_SHADOW_PHI:.+]] = phi ptr addrspace(10) [ %"p1'ipg", %T ], [ %"p2'ipg", %M ]
; CHECK:   store ptr addrspace(10) %[[LINEAR_SHADOW_PHI]], ptr %[[LINEAR_SHADOW_CACHE]], align 8

; CHECK-LABEL: define internal { { i1, i1, i1 }, double } @augmented_g(
; CHECK-NOT: @jl_gc_alloc_typed
; CHECK:   %al = alloca i8, i64 8, align 8
; CHECK:   store double %x, ptr %al, align 8
; CHECK: merge:
; CHECK:   %phi = phi ptr [ %p1, %T ], [ %p2, %M ]

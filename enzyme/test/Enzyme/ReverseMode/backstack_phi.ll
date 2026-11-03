; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -enzyme-julia-addr-load -S | FileCheck %s; fi

; A Julia allocation that came from the stack (`enzyme_fromstack`) is moved
; back there after differentiation, both it and its shadow. A pointer derived
; from them reaches a phi that also merges pointers from elsewhere, so the phi
; keeps its (derived) address space and takes the moved pointer cast back to it.
; With three incoming values the reverse pass recomputes it as a phi as well.

declare ptr addrspace(10) @jl_gc_alloc_typed(ptr, i64, ptr addrspace(10))

declare double @__enzyme_autodiff(...)

define double @g(ptr addrspace(10) %other, ptr addrspace(10) %other2, i8 %c, i1 %c2, double %x) {
entry:
  %al = call noalias nonnull dereferenceable(8) ptr addrspace(10) @jl_gc_alloc_typed(ptr null, i64 8, ptr addrspace(10) addrspacecast (ptr inttoptr (i64 139806792221568 to ptr) to ptr addrspace(10))), !enzyme_fromstack !0
  %ald = addrspacecast ptr addrspace(10) %al to ptr addrspace(11)
  store double %x, ptr addrspace(11) %ald, align 8
  %od = addrspacecast ptr addrspace(10) %other to ptr addrspace(11)
  %od2 = addrspacecast ptr addrspace(10) %other2 to ptr addrspace(11)
  br i1 %c2, label %region, label %exit

region:
  switch i8 %c, label %a [ i8 1, label %b
                           i8 2, label %d ]

a:
  br label %merge

b:
  br label %merge

d:
  br label %merge

merge:
  %phi = phi ptr addrspace(11) [ %ald, %a ], [ %od, %b ], [ %od2, %d ]
  %v = load double, ptr addrspace(11) %phi, align 8
  %sq = fmul double %v, %v
  br label %exit

exit:
  %r = phi double [ %sq, %merge ], [ 0.000000e+00, %entry ]
  ret double %r
}

define double @dg(ptr addrspace(10) %other, ptr addrspace(10) %dother, ptr addrspace(10) %other2, ptr addrspace(10) %dother2, i8 %c, i1 %c2, double %x) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @g, metadata !"enzyme_dup", ptr addrspace(10) %other, ptr addrspace(10) %dother, metadata !"enzyme_dup", ptr addrspace(10) %other2, ptr addrspace(10) %dother2, metadata !"enzyme_const", i8 %c, metadata !"enzyme_const", i1 %c2, double %x)
  ret double %r
}

!0 = !{i64 8}

; CHECK: define internal { double } @diffeg(
; CHECK-NOT: jl_gc_alloc_typed
; CHECK:        %al = alloca i8, i64 8, align 8
; CHECK:        %"al'mi" = alloca i8, i64 8, align 8
; CHECK:        %[[gpal:.+]] = addrspacecast ptr %al to ptr addrspace(11)
; CHECK-NEXT:   %[[gsal:.+]] = addrspacecast ptr %"al'mi" to ptr addrspace(11)
; CHECK:        phi ptr addrspace(11) [ %[[gsal]], %a ], [ %"od'ipc", %b ], [ %"od2'ipc", %d ]
; CHECK-NEXT:   %phi = phi ptr addrspace(11) [ %[[gpal]], %a ], [ %od, %b ], [ %od2, %d ]
; CHECK:        %[[gpal2:.+]] = addrspacecast ptr %al to ptr addrspace(11)
; CHECK:        phi ptr addrspace(11) [ %[[gpal2]], %invertmerge_phirc ], [ %od, %invertmerge_phirc1 ], [ %od2, %invertmerge_phirc2 ]
; CHECK:        %[[gsal2:.+]] = addrspacecast ptr %"al'mi" to ptr addrspace(11)
; CHECK:        phi ptr addrspace(11) [ %[[gsal2]], %{{.+}} ], [ %"od'ipc", %{{.+}} ], [ %"od2'ipc", %{{.+}} ]

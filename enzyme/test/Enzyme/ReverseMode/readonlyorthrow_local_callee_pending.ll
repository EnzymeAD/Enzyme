; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Like readonlyorthrow_local_callee.ll, but every caller is defined before the
; callee whose sret it passes on. The inference visits functions in module
; order, so a caller is analyzed while its callee is still unclassified and is
; only legal subject to the callee ("pending"). When the callee then turns out
; to be only local read-only-or-throw, the caller must be analyzed again: the
; callee writes through the sret the caller passed it, which the caller's
; first analysis could not account for. Resolving the caller with what that
; analysis found instead marks `passthrough` (and `outer`, two levels up)
; fully read-only-or-throw rather than local, and `intoarg`, which lets the
; callee write through an ordinary pointer argument, read-only-or-throw rather
; than not at all. Julia emits callers before callees routinely, e.g. for
; `f(y) = collect(SVector(...))` ahead of `_collect_indices`
; (EnzymeAD/Enzyme.jl#3776).

declare void @julia.safepoint(ptr) #1

define void @outer(ptr noalias nocapture noundef sret({ i64, i64 }) %out, ptr addrspace(11) nocapture %r) {
top:
  call void @passthrough(ptr sret({ i64, i64 }) %out, ptr addrspace(11) %r)
  ret void
}

define void @passthrough(ptr noalias nocapture noundef sret({ i64, i64 }) %out, ptr addrspace(11) nocapture %r) {
top:
  call void @callee(ptr sret({ i64, i64 }) %out, ptr addrspace(11) %r)
  ret void
}

define i64 @viaalloca(ptr addrspace(11) nocapture %r) {
top:
  %tmp = alloca { i64, i64 }, align 8
  call void @callee(ptr sret({ i64, i64 }) %tmp, ptr addrspace(11) %r)
  %v = load i64, ptr %tmp, align 8
  ret i64 %v
}

define void @intoarg(ptr nocapture %dst, ptr addrspace(11) nocapture %r) {
top:
  call void @callee(ptr sret({ i64, i64 }) %dst, ptr addrspace(11) %r)
  ret void
}

define void @callee(ptr noalias nocapture noundef sret({ i64, i64 }) %out, ptr addrspace(11) nocapture %r) {
top:
  %datap = load ptr, ptr addrspace(11) %r, align 8
  %el = load i64, ptr %datap, align 8
  store i64 %el, ptr %out, align 8
  ret void
}

attributes #1 = { "enzyme_ReadOnlyOrThrow" }

; CHECK: define void @outer(ptr noalias {{.*}}sret({ i64, i64 }) %out, ptr addrspace(11) {{.*}}readonly{{.*}} %r) #[[LOCAL:[0-9]+]]
; CHECK: define void @passthrough(ptr noalias {{.*}}sret({ i64, i64 }) %out, ptr addrspace(11) {{.*}}readonly{{.*}} %r) #[[LOCAL]]
; CHECK: define i64 @viaalloca(ptr addrspace(11) {{.*}}readonly{{.*}} %r) #[[RO:[0-9]+]]
; CHECK: define void @intoarg(ptr {{.*}} %dst, ptr addrspace(11) {{.*}} %r) #[[NONE:[0-9]+]]
; CHECK: define void @callee(ptr noalias {{.*}}sret({ i64, i64 }) %out, ptr addrspace(11) {{.*}}readonly{{.*}} %r) #[[LOCAL]]
; CHECK-DAG: attributes #[[LOCAL]] = { {{.*}}memory(read, argmem: readwrite, inaccessiblemem: readwrite) "enzyme_LocalReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[RO]] = { {{.*}}memory(read, inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[NONE]] = { nounwind }

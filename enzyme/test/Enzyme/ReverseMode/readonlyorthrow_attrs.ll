; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; A Julia-style function: safepoint prologue, a bounds check whose failing path
; allocates an exception and throws, and a result written through an sret. It
; only writes memory on the path that throws, plus its sret, so besides the
; string attribute the analysis states that in LLVM's own terms.

declare ptr @julia.get_pgcstack() #2
; Enzyme.jl marks the safepoint as read-only-or-throw itself.
declare void @julia.safepoint(ptr) #1
declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
declare void @ijl_throw(ptr addrspace(12)) #0

define void @lookup(ptr noalias nocapture noundef sret({ i64, i64 }) %out, ptr addrspace(11) nocapture %r, i64 %i) {
top:
  %pgcstack = call ptr @julia.get_pgcstack()
  %ptls_field = getelementptr inbounds i8, ptr %pgcstack, i64 16
  %ptls_load = load ptr, ptr %ptls_field, align 8
  %sp = getelementptr inbounds i8, ptr %ptls_load, i64 16
  %safepoint = load atomic ptr, ptr %sp monotonic, align 8
  fence syncscope("singlethread") seq_cst
  call void @julia.safepoint(ptr %safepoint)
  fence syncscope("singlethread") seq_cst
  %lenp = getelementptr inbounds i8, ptr addrspace(11) %r, i64 8
  %len = load i64, ptr addrspace(11) %lenp, align 8
  %inbounds = icmp ult i64 %i, %len
  br i1 %inbounds, label %ok, label %oob

oob:
  %current_task = getelementptr inbounds i8, ptr %pgcstack, i64 -152
  %exc = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %current_task, i64 16, ptr addrspace(10) null)
  %excd = addrspacecast ptr addrspace(10) %exc to ptr addrspace(11)
  store i64 %i, ptr addrspace(11) %excd, align 8
  %exc12 = addrspacecast ptr addrspace(10) %exc to ptr addrspace(12)
  call void @ijl_throw(ptr addrspace(12) %exc12)
  unreachable

ok:
  %datap = load ptr, ptr addrspace(11) %r, align 8
  %elp = getelementptr inbounds i64, ptr %datap, i64 %i
  %el = load i64, ptr %elp, align 8
  store i64 %el, ptr %out, align 8
  %out2 = getelementptr inbounds i8, ptr %out, i64 8
  store i64 1, ptr %out2, align 8
  ret void
}

; A function that writes through an ordinary pointer argument on a returning
; path must not be marked.

define void @setfirst(ptr addrspace(11) nocapture %r, i64 %v) {
top:
  %datap = load ptr, ptr addrspace(11) %r, align 8
  store i64 %v, ptr %datap, align 8
  ret void
}

attributes #0 = { noreturn }
attributes #1 = { "enzyme_ReadOnlyOrThrow" }
attributes #2 = { memory(none) }

; `nocapture` prints as `captures(none)` from LLVM 21 on, so the arguments
; are matched loosely. `setfirst` only gets what Enzyme infers for any
; function (`nounwind`; its argument is itself only read).
; CHECK: define void @lookup(ptr noalias {{.*}}sret({ i64, i64 }) %out, ptr addrspace(11) {{.*}}readonly{{.*}} %r, i64 %i) #[[LOOKUP:[0-9]+]]
; CHECK: define void @setfirst(ptr addrspace(11) {{.*}} %r, i64 %v) #[[SETFIRST:[0-9]+]]
; CHECK: attributes #[[LOOKUP]] = { memory(read, argmem: readwrite, inaccessiblemem: readwrite) "enzyme_LocalReadOnlyOrThrow" }
; CHECK: attributes #[[SETFIRST]] = { nounwind }

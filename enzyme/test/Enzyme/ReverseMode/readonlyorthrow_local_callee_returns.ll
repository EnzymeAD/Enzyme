; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; A local read-only-or-throw callee may write memory it allocates and hand it
; to its caller, through its return value or an sret-like argument. A caller
; that can get such a pointer may hand it on in turn, so it is only local
; read-only-or-throw too. Marking `wrap` fully read-only-or-throw would let
; activity analysis treat a call to it as unable to propagate derivatives,
; although the memory it returns was written from its arguments.

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))

define ptr addrspace(10) @fresh(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %v = load double, ptr addrspace(11) %x, align 8
  %a = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store double %v, ptr addrspace(11) %a11, align 8
  ret ptr addrspace(10) %a
}

define ptr addrspace(10) @wrap(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %r = call ptr addrspace(10) @fresh(ptr %task, ptr addrspace(11) %x)
  ret ptr addrspace(10) %r
}

; The same through an sret holding a pointer.

define void @fresh_sret(ptr noalias nocapture noundef sret({ ptr addrspace(10) }) %out, ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %v = load double, ptr addrspace(11) %x, align 8
  %a = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store double %v, ptr addrspace(11) %a11, align 8
  store ptr addrspace(10) %a, ptr %out, align 8
  ret void
}

define ptr addrspace(10) @wrap_sret(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %tmp = alloca { ptr addrspace(10) }, align 8
  call void @fresh_sret(ptr sret({ ptr addrspace(10) }) %tmp, ptr %task, ptr addrspace(11) %x)
  %r = load ptr addrspace(10), ptr %tmp, align 8
  ret ptr addrspace(10) %r
}

; The same for a call marked local read-only-or-throw by metadata, rather than
; by an attribute of the callee. A caller whose call returns no pointer stays
; fully read-only-or-throw.

declare ptr addrspace(10) @make(ptr addrspace(11))
declare i64 @makeint(ptr addrspace(11))

define ptr addrspace(10) @wrap_md(ptr addrspace(11) nocapture %x) {
top:
  %r = call ptr addrspace(10) @make(ptr addrspace(11) %x), !enzyme_LocalReadOnlyOrThrow !0
  ret ptr addrspace(10) %r
}

define i64 @wrap_md_int(ptr addrspace(11) nocapture %x) {
top:
  %r = call i64 @makeint(ptr addrspace(11) %x), !enzyme_LocalReadOnlyOrThrow !0
  ret i64 %r
}

!0 = !{}

; CHECK: define ptr addrspace(10) @fresh({{.*}}) #[[LOCAL:[0-9]+]]
; CHECK: define ptr addrspace(10) @wrap({{.*}}) #[[LOCAL]]
; CHECK: define void @fresh_sret({{.*}}) #[[LOCAL]]
; CHECK: define ptr addrspace(10) @wrap_sret({{.*}}) #[[LOCAL]]
; CHECK: define ptr addrspace(10) @wrap_md({{.*}}) #[[LOCAL]]
; CHECK: define i64 @wrap_md_int({{.*}}) #[[RO:[0-9]+]]
; CHECK-DAG: attributes #[[LOCAL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[RO]] = { {{.*}}"enzyme_ReadOnlyOrThrow" }

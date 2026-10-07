; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; A local read-only-or-throw callee may write memory it allocates and hand it
; to its caller, through its return value or an sret-like argument. A caller
; that lets that memory escape, directly or through a pointer loaded from it,
; is only local read-only-or-throw too. Marking `wrap` fully
; read-only-or-throw would let activity analysis treat a call to it as unable
; to propagate derivatives, although the memory it returns was written from
; its arguments.

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

; A caller that only uses the memory a local callee returned internally does
; not let it reach its return value, so it writes no memory that outlives it
; and stays fully read-only-or-throw.

define void @use_only(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %r = call ptr addrspace(10) @fresh(ptr %task, ptr addrspace(11) %x)
  %r11 = addrspacecast ptr addrspace(10) %r to ptr addrspace(11)
  %v = load double, ptr addrspace(11) %r11, align 8
  ret void
}

; Returning a value loaded through that memory may return a pointer to it
; unless type info rules that out: here the return value is only known to be a
; double by the function's enzyme_type, without which the caller is local.

define double @use_ret(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %r = call ptr addrspace(10) @fresh(ptr %task, ptr addrspace(11) %x)
  %r11 = addrspacecast ptr addrspace(10) %r to ptr addrspace(11)
  %v = load double, ptr addrspace(11) %r11, align 8
  ret double %v
}

define "enzyme_type"="{[-1]:Float@double}" double @use_ret_typed(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %r = call ptr addrspace(10) @fresh(ptr %task, ptr addrspace(11) %x)
  %r11 = addrspacecast ptr addrspace(10) %r to ptr addrspace(11)
  %v = load double, ptr addrspace(11) %r11, align 8
  ret double %v
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
; by an attribute of the callee. A caller whose call returns nothing, or
; returns an integer that type info says is not a pointer, stays fully
; read-only-or-throw; an integer without such info may be a pointer.

declare ptr addrspace(10) @make(ptr addrspace(11))
declare void @makevoid(ptr addrspace(11))
declare i64 @makeint(ptr addrspace(11))

define ptr addrspace(10) @wrap_md(ptr addrspace(11) nocapture %x) {
top:
  %r = call ptr addrspace(10) @make(ptr addrspace(11) %x), !enzyme_LocalReadOnlyOrThrow !0
  ret ptr addrspace(10) %r
}

define void @wrap_md_void(ptr addrspace(11) nocapture %x) {
top:
  call void @makevoid(ptr addrspace(11) %x), !enzyme_LocalReadOnlyOrThrow !0
  ret void
}

define i64 @wrap_md_int(ptr addrspace(11) nocapture %x) {
top:
  %r = call i64 @makeint(ptr addrspace(11) %x), !enzyme_LocalReadOnlyOrThrow !0
  ret i64 %r
}

define i64 @wrap_md_int_typed(ptr addrspace(11) nocapture %x) {
top:
  %r = call "enzyme_type"="{[-1]:Integer}" i64 @makeint(ptr addrspace(11) %x), !enzyme_LocalReadOnlyOrThrow !0
  ret i64 %r
}

!0 = !{}

; CHECK: define ptr addrspace(10) @fresh({{.*}}) #[[LOCAL:[0-9]+]]
; CHECK: define ptr addrspace(10) @wrap({{.*}}) #[[LOCAL]]
; CHECK: define void @use_only({{.*}}) #[[RO:[0-9]+]]
; CHECK: define double @use_ret({{.*}}) #[[LOCAL]]
; CHECK: define "enzyme_type"="{[-1]:Float@double}" double @use_ret_typed({{.*}}) #[[RO]]
; CHECK: define void @fresh_sret({{.*}}) #[[LOCAL]]
; CHECK: define ptr addrspace(10) @wrap_sret({{.*}}) #[[LOCAL]]
; CHECK: define ptr addrspace(10) @wrap_md({{.*}}) #[[LOCAL]]
; CHECK: define void @wrap_md_void({{.*}}) #[[RO]]
; CHECK: define i64 @wrap_md_int({{.*}}) #[[LOCAL]]
; CHECK: define i64 @wrap_md_int_typed({{.*}}) #[[RO]]
; CHECK-DAG: attributes #[[LOCAL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[RO]] = { {{.*}}"enzyme_ReadOnlyOrThrow" }

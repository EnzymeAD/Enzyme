; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Like readonlyorthrow_local_callee_returns.ll and readonlyorthrow_local_callee.ll,
; but each caller comes before its callee, so the callee is not yet known to be
; read-only-or-throw when the caller is first checked. Once the callee turns out
; to be only local, the caller must be checked again: @wrap returns memory the
; callee wrote and is only local, and @intoarg has the callee write through its
; ordinary pointer argument and must not be marked at all. @use_only, which
; only reads the memory, stays fully read-only-or-throw.

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))

define ptr addrspace(10) @wrap(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %r = call ptr addrspace(10) @fresh(ptr %task, ptr addrspace(11) %x)
  ret ptr addrspace(10) %r
}

define void @use_only(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %r = call ptr addrspace(10) @fresh(ptr %task, ptr addrspace(11) %x)
  %r11 = addrspacecast ptr addrspace(10) %r to ptr addrspace(11)
  %v = load double, ptr addrspace(11) %r11, align 8
  ret void
}

define void @intoarg(ptr %dst, ptr addrspace(11) nocapture %x) {
top:
  call void @fresh_sret(ptr sret({ i64, i64 }) %dst, ptr addrspace(11) %x)
  ret void
}

define ptr addrspace(10) @fresh(ptr %task, ptr addrspace(11) nocapture %x) {
top:
  %v = load double, ptr addrspace(11) %x, align 8
  %a = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store double %v, ptr addrspace(11) %a11, align 8
  ret ptr addrspace(10) %a
}

define void @fresh_sret(ptr noalias nocapture noundef sret({ i64, i64 }) %out, ptr addrspace(11) nocapture %x) {
top:
  %v = load i64, ptr addrspace(11) %x, align 8
  store i64 %v, ptr %out, align 8
  ret void
}

; CHECK: define ptr addrspace(10) @wrap({{.*}}) #[[LOCAL:[0-9]+]]
; CHECK: define void @use_only({{.*}}) #[[RO:[0-9]+]]
; CHECK: define void @intoarg({{.*}}) #[[NONE:[0-9]+]]
; CHECK: define ptr addrspace(10) @fresh({{.*}}) #[[LOCAL]]
; CHECK: define void @fresh_sret({{.*}}) #[[LOCALSRET:[0-9]+]]
; CHECK-DAG: attributes #[[LOCAL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[LOCALSRET]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[RO]] = { {{.*}}"enzyme_ReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[NONE]] = { nounwind }

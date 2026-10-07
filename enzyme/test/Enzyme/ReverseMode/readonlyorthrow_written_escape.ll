; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Writing an object allocated in the function writes memory that did not
; exist before the call. That memory outlives the call, so that the function
; is only local read-only-or-throw, if it may reach the function's return
; value: directly, through another allocation holding it that does, through a
; call that may hand it back, or through an alloca it is stored into. Escaping
; otherwise does not matter: storing it into memory existing before the call
; is a write that disqualifies the function anyway.

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))
declare void @sink(ptr addrspace(10)) #0

define ptr addrspace(10) @id(ptr addrspace(10) %p) {
top:
  ret ptr addrspace(10) %p
}

; Captured by a call that cannot hand it back.

define void @captured_not_returned(ptr %task, double %v) {
top:
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store double %v, ptr addrspace(11) %a11, align 8
  call void @sink(ptr addrspace(10) %a)
  ret void
}

; Stored into another allocation, which is not returned.

define void @into_local(ptr %task, double %v) {
top:
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store double %v, ptr addrspace(11) %a11, align 8
  %b = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %b11 = addrspacecast ptr addrspace(10) %b to ptr addrspace(11)
  store ptr addrspace(10) %a, ptr addrspace(11) %b11, align 8
  ret void
}

; Stored into another allocation, which is returned.

define ptr addrspace(10) @into_returned(ptr %task, double %v) {
top:
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store double %v, ptr addrspace(11) %a11, align 8
  %b = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %b11 = addrspacecast ptr addrspace(10) %b to ptr addrspace(11)
  store ptr addrspace(10) %a, ptr addrspace(11) %b11, align 8
  ret ptr addrspace(10) %b
}

; Returned through a call that hands it back.

define ptr addrspace(10) @through_call(ptr %task, double %v) {
top:
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store double %v, ptr addrspace(11) %a11, align 8
  %r = call ptr addrspace(10) @id(ptr addrspace(10) %a)
  ret ptr addrspace(10) %r
}

; Returned through an alloca it is stored into.

define ptr addrspace(10) @through_alloca(ptr %task, double %v) {
top:
  %t = alloca ptr addrspace(10), align 8
  %a = call ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  %a11 = addrspacecast ptr addrspace(10) %a to ptr addrspace(11)
  store double %v, ptr addrspace(11) %a11, align 8
  store ptr addrspace(10) %a, ptr %t, align 8
  %r = load ptr addrspace(10), ptr %t, align 8
  ret ptr addrspace(10) %r
}

attributes #0 = { "enzyme_ReadOnlyOrThrow" }

; CHECK: define ptr addrspace(10) @id({{.*}}) #[[ID:[0-9]+]] {
; CHECK: define void @captured_not_returned({{.*}}) #[[RO:[0-9]+]] {
; CHECK: define void @into_local({{.*}}) #[[RO]] {
; CHECK: define ptr addrspace(10) @into_returned({{.*}}) #[[LOCAL:[0-9]+]] {
; CHECK: define ptr addrspace(10) @through_call({{.*}}) #[[LOCAL]] {
; CHECK: define ptr addrspace(10) @through_alloca({{.*}}) #[[LOCAL]] {
; CHECK-DAG: attributes #[[ID]] = { {{.*}}"enzyme_ReadOnlyOrThrow"{{.*}} }
; CHECK-DAG: attributes #[[RO]] = { {{.*}}"enzyme_ReadOnlyOrThrow"{{.*}} }
; CHECK-DAG: attributes #[[LOCAL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow"{{.*}} }

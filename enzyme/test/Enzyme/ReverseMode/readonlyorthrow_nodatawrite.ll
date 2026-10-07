; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Read-only-or-throw functions that write no data to memory at all, not even
; memory of their own, are also marked enzyme_NoDataWrite. Allocating memory
; does not count as writing data, nor does calling such a function. Writing
; memory of one's own, or calling a function that does (including one checked
; after the caller), does.

declare noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr, i64, ptr addrspace(10))

define ptr @reader(ptr %p) {
top:
  %q = load ptr, ptr %p, align 8
  ret ptr %q
}

define ptr addrspace(10) @allocates(ptr %task) {
top:
  %a = call noalias nonnull ptr addrspace(10) @julia.gc_alloc_obj(ptr %task, i64 8, ptr addrspace(10) null)
  ret ptr addrspace(10) %a
}

define ptr @calls_reader(ptr %p) {
top:
  %q = call ptr @reader(ptr %p)
  ret ptr %q
}

define ptr @writes_alloca(ptr %p) {
top:
  %t = alloca double, align 8
  %q = load ptr, ptr %p, align 8
  %v = load double, ptr %q, align 8
  store double %v, ptr %t, align 8
  ret ptr %q
}

define ptr @calls_writer(ptr %p) {
top:
  %q = call ptr @writes_alloca(ptr %p)
  ret ptr %q
}

; Checked before its callee is known to be read only or throw.

define ptr @calls_later_writer(ptr %p) {
top:
  %q = call ptr @writes_alloca_later(ptr %p)
  ret ptr %q
}

define ptr @writes_alloca_later(ptr %p) {
top:
  %t = alloca double, align 8
  %q = load ptr, ptr %p, align 8
  %v = load double, ptr %q, align 8
  store double %v, ptr %t, align 8
  ret ptr %q
}

; CHECK: define ptr @reader({{.*}}) #[[NOWRITE:[0-9]+]]
; CHECK: define ptr addrspace(10) @allocates({{.*}}) #[[NOWRITEALLOC:[0-9]+]]
; CHECK: define ptr @calls_reader({{.*}}) #[[NOWRITE]]
; CHECK: define ptr @writes_alloca({{.*}}) #[[WRITE:[0-9]+]]
; CHECK: define ptr @calls_writer({{.*}}) #[[WRITE]]
; CHECK: define ptr @calls_later_writer({{.*}}) #[[WRITE]]
; CHECK: define ptr @writes_alloca_later({{.*}}) #[[WRITE]]
; CHECK-DAG: attributes #[[NOWRITE]] = { {{.*}}"enzyme_NoDataWrite" "enzyme_ReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[NOWRITEALLOC]] = { {{.*}}"enzyme_NoDataWrite" "enzyme_ReadOnlyOrThrow" }
; CHECK-DAG: attributes #[[WRITE]] = { {{.*}}memory({{.*}}) "enzyme_ReadOnlyOrThrow" }

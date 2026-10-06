; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; A local read-only-or-throw callee may return memory it allocated and wrote.
; A caller that lets that memory escape (here, by returning it) wrote it too
; and is only local read-only-or-throw. Marking `passthrough` fully read-only
; let activity analysis treat a call to it as a constant instruction, so the
; reverse pass got an undefined shadow for its argument (Enzyme.jl #3776).
; A caller that only reads the fresh memory stays fully read-only.

declare noalias ptr @malloc(i64)

define ptr @callee(ptr nocapture %x) {
top:
  %v = load double, ptr %x, align 8
  %m = call noalias ptr @malloc(i64 8)
  store double %v, ptr %m, align 8
  ret ptr %m
}

define ptr @passthrough(ptr nocapture %x) {
top:
  %r = call ptr @callee(ptr %x)
  ret ptr %r
}

define double @readsfresh(ptr nocapture %x) {
top:
  %r = call ptr @callee(ptr %x)
  %v = load double, ptr %r, align 8
  ret double %v
}

; `callee` and `passthrough` end up in the same attribute group.
; CHECK: define ptr @callee(ptr {{.*}}%x) #[[LOCAL:[0-9]+]]
; CHECK: define ptr @passthrough(ptr {{.*}}%x) #[[LOCAL]]
; CHECK: define double @readsfresh(ptr {{.*}}%x) #[[RO:[0-9]+]]
; CHECK-DAG: attributes #[[LOCAL]] = { {{.*}}"enzyme_LocalReadOnlyOrThrow"{{.*}} }
; CHECK-DAG: attributes #[[RO]] = { {{.*}}"enzyme_ReadOnlyOrThrow"{{.*}} }

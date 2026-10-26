; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; The callers below are visited before @fill, so when they are first checked
; @fill is not yet known to be read only or throw and they wait on it. @fill
; then turns out only local: it writes its sret. Once it is marked, each caller
; must be checked again with that knowledge rather than marked from its first
; check, which could not see what @fill writes.

; Passes its own (non-sret) argument as @fill's sret, so @fill writes memory
; the caller can see: not read only or throw at all.
define void @through_arg(ptr %out, ptr %x) {
entry:
  call void @fill(ptr sret(double) %out, ptr %x)
  ret void
}

; Passes its own sret through to @fill: local, like @fill.
define void @through_sret(ptr noalias nocapture sret(double) %out, ptr %x) {
entry:
  call void @fill(ptr sret(double) %out, ptr %x)
  ret void
}

; Passes an alloca: @fill only writes memory local to the caller, so the
; caller is fully read only or throw.
define double @into_alloca(ptr %x) {
entry:
  %a = alloca double, align 8
  call void @fill(ptr sret(double) %a, ptr %x)
  %v = load double, ptr %a, align 8
  ret double %v
}

define void @fill(ptr noalias nocapture sret(double) %out, ptr %x) {
entry:
  %v = load double, ptr %x, align 8
  store double %v, ptr %out, align 8
  ret void
}

; CHECK: define void @through_arg(ptr {{.*}}writeonly %out, ptr {{.*}}%x) #[[ARG:[0-9]+]]
; CHECK: define void @through_sret(ptr {{.*}}sret(double) {{.*}}%out, ptr {{.*}}%x) #[[SRET:[0-9]+]]
; CHECK: define double @into_alloca(ptr {{.*}}%x) #[[FULL:[0-9]+]]
; CHECK: define void @fill(ptr {{.*}}sret(double) {{.*}}%out, ptr {{.*}}%x) #[[SRET]]
; CHECK: attributes #[[ARG]] = { nounwind }
; CHECK: attributes #[[SRET]] = { {{.*}}memory(read, argmem: readwrite, inaccessiblemem: readwrite) "enzyme_LocalReadOnlyOrThrow" }
; CHECK: attributes #[[FULL]] = { {{.*}}memory(read, inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" }

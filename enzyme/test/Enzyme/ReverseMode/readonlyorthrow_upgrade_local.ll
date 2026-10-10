; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; Enzyme.jl runs the read-only-or-throw inference several times. A function an
; earlier run marked local (e.g. while it still called a callee that could hand
; back memory it wrote) can turn out fully read-only-or-throw once optimized.
; The upgrade replaces the local attribute rather than adding to it.

; Only reads: upgraded to fully read-only-or-throw.
define double @reads(ptr %x) "enzyme_LocalReadOnlyOrThrow" {
entry:
  %v = load double, ptr %x, align 8
  ret double %v
}

; Writes its sret: stays local.
define void @writes_sret(ptr noalias nocapture sret(double) %out, ptr %x) "enzyme_LocalReadOnlyOrThrow" {
entry:
  %v = load double, ptr %x, align 8
  store double %v, ptr %out, align 8
  ret void
}

; CHECK: define double @reads(ptr {{.*}}%x) #[[READS:[0-9]+]]
; CHECK: define void @writes_sret(ptr {{.*}}sret(double) {{.*}}%out, ptr {{.*}}%x) #[[SRET:[0-9]+]]
; CHECK: attributes #[[READS]] = { {{.*}}memory(read, inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" }
; CHECK: attributes #[[SRET]] = { {{.*}}memory(read, argmem: readwrite, inaccessiblemem: readwrite) "enzyme_LocalReadOnlyOrThrow" }

; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-julia-addr-load -passes="enzyme" -S | FileCheck %s; fi

; A call to a local read-only-or-throw callee writes through the callee's
; sret. The caller must account for where that sret points: its own sret
; (passed straight through, as call-slot optimization leaves it) makes the
; caller local too, an alloca keeps it fully read-only, and an ordinary
; pointer argument disqualifies it. Marking `passthrough` fully read-only
; would let LLVM turn its `writeonly` sret into `readnone` and drop the
; result.

declare void @julia.safepoint(ptr) #1

define void @callee(ptr noalias nocapture noundef sret({ i64, i64 }) %out, ptr addrspace(11) nocapture %r) {
top:
  %datap = load ptr, ptr addrspace(11) %r, align 8
  %el = load i64, ptr %datap, align 8
  store i64 %el, ptr %out, align 8
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

attributes #1 = { "enzyme_ReadOnlyOrThrow" }

; `callee` and `passthrough` end up in the same attribute group.
; CHECK: define void @callee(ptr noalias {{.*}}sret({ i64, i64 }) %out, ptr addrspace(11) {{.*}}readonly{{.*}} %r) #[[LOCAL:[0-9]+]]
; CHECK: define void @passthrough(ptr noalias {{.*}}sret({ i64, i64 }) %out, ptr addrspace(11) {{.*}}readonly{{.*}} %r) #[[LOCAL]]
; CHECK: define i64 @viaalloca(ptr addrspace(11) {{.*}}readonly{{.*}} %r) #[[RO:[0-9]+]]
; CHECK: define void @intoarg(ptr {{.*}} %dst, ptr addrspace(11) {{.*}} %r) #[[NONE:[0-9]+]]
; CHECK: attributes #[[LOCAL]] = { {{.*}}memory(read, argmem: readwrite, inaccessiblemem: readwrite) "enzyme_LocalReadOnlyOrThrow" }
; CHECK: attributes #[[RO]] = { {{.*}}memory(read, inaccessiblemem: readwrite) "enzyme_ReadOnlyOrThrow" }
; CHECK: attributes #[[NONE]] = { nounwind }

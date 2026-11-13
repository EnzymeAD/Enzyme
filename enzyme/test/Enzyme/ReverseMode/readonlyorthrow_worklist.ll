; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; Verify that nounwind and pointer-argument attributes propagate completely
; through multi-callee diamond call graphs and multi-wave callees defined after
; their callers.

define double @top(ptr %p) {
entry:
  %a = call double @mid_a(ptr %p)
  %b = call double @mid_b(ptr %p)
  %s = fadd double %a, %b
  ret double %s
}

define double @mid_a(ptr %p) {
entry:
  %a = call double @leaf_a(ptr %p)
  %b = call double @leaf_b(ptr %p)
  %s = fadd double %a, %b
  ret double %s
}

define double @mid_b(ptr %p) {
entry:
  %a = call double @leaf_a(ptr %p)
  %b = call double @leaf_b(ptr %p)
  %s = fadd double %a, %b
  ret double %s
}

; Multi-wave pointer-arg propagation: @callee_multi gains attributes on %a
; immediately, and on %b only after @leaf_b is analyzed. @caller_multi must
; still be woken on the second wave.
define double @caller_multi(ptr %x, ptr %y) {
entry:
  %r = call double @callee_multi(ptr %x, ptr %y)
  ret double %r
}

define double @callee_multi(ptr %a, ptr %b) {
entry:
  %va = load double, ptr %a, align 8
  %vb = call double @leaf_b(ptr %b)
  %s = fadd double %va, %vb
  ret double %s
}

define double @leaf_a(ptr %p) {
entry:
  %v = load double, ptr %p, align 8
  ret double %v
}

define double @leaf_b(ptr %p) {
entry:
  %v = load double, ptr %p, align 8
  ret double %v
}

; CHECK: define double @top(ptr {{.*}}readonly{{.*}} %p) #[[RO:[0-9]+]]
; CHECK: define double @mid_a(ptr {{.*}}readonly{{.*}} %p) #[[RO]]
; CHECK: define double @mid_b(ptr {{.*}}readonly{{.*}} %p) #[[RO]]
; CHECK: define double @caller_multi(ptr {{.*}}readonly{{.*}} %x, ptr {{.*}}readonly{{.*}} %y) #[[RO]]
; CHECK: define double @callee_multi(ptr {{.*}}readonly{{.*}} %a, ptr {{.*}}readonly{{.*}} %b) #[[RO]]
; CHECK: define double @leaf_a(ptr {{.*}}readonly{{.*}} %p) #[[RO]]
; CHECK: define double @leaf_b(ptr {{.*}}readonly{{.*}} %p) #[[RO]]
; CHECK: attributes #[[RO]] = { nounwind "enzyme_ReadOnlyOrThrow" }

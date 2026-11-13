; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -enzyme-postopt=true -enzyme-detect-readthrow=true -passes="enzyme" -S | FileCheck %s; fi

; Attribute inference in DetectReadonlyOrThrow alone must not trigger the full
; O2 PostOpt simplification pipeline when no Enzyme calls were lowered.

define internal double @helper(ptr %p) {
entry:
  %v = load double, ptr %p, align 8
  %r = fadd double %v, 1.0
  ret double %r
}

define double @caller(ptr %p) {
entry:
  %r = call double @helper(ptr %p)
  ret double %r
}

; CHECK: define internal double @helper(ptr {{.*}}readonly{{.*}} %p) #[[ATTR:[0-9]+]]
; CHECK: define double @caller(ptr {{.*}}readonly{{.*}} %p) #[[ATTR]]
; CHECK:   %r = call double @helper(ptr %p)
; CHECK:   ret double %r
; CHECK: attributes #[[ATTR]] = { nounwind "enzyme_ReadOnlyOrThrow" }

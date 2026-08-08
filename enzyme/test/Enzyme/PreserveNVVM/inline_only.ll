; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm" -S | FileCheck %s --check-prefix=PROMOTE
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm-inline-only" -S | FileCheck %s --check-prefix=INLINEONLY
; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm-inline-only,preserve-nvvm-end" -S | FileCheck %s --check-prefix=RESTORED

; Preserving a libdevice definition normally promotes it to external linkage
; so nothing internalizes or drops it before Enzyme has seen it. A pipeline
; that keeps the definitions alive its own way only needs the calls to stay
; recognizable -- noinline -- and the definition's linkage left alone, so a
; formerly-internal function is not exported as a strong symbol by every
; translation unit. preserve-nvvm-end undoes the inlining toggle either way.

define internal double @__nv_sin(double %x) alwaysinline {
  ret double %x
}

define double @caller(double %x) {
  %r = call double @__nv_sin(double %x)
  ret double %r
}

; PROMOTE: define dso_local double @__nv_sin(double %x) #[[ATTR:[0-9]+]]
; PROMOTE: attributes #[[ATTR]] = { noinline {{.*}}"prev_fixup" "prev_linkage"="7" }

; INLINEONLY: define internal double @__nv_sin(double %x) #[[ATTR:[0-9]+]]
; INLINEONLY-NOT: prev_linkage
; INLINEONLY: attributes #[[ATTR]] = { noinline {{.*}}"prev_fixup" }

; RESTORED: define internal double @__nv_sin(double %x) #[[ATTR:[0-9]+]]
; RESTORED-NOT: prev_fixup
; RESTORED: attributes #[[ATTR]] = { alwaysinline

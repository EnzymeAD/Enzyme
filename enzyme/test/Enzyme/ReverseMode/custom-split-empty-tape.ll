; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; @square has a custom augmented forward pass that returns only an empty tape
; ({}) and writes to %out. @top reads %out after calling @caller, so Enzyme
; differentiates @caller in split mode. The augmented forward pass of @caller
; returns the tape of @square directly, and must keep the call to
; @augment_square for its store to %out.

define internal {} @augment_square(ptr %x, ptr %dx, ptr %out, ptr %dout) {
entry:
  %v = load double, ptr %x
  %m = fmul double %v, %v
  store double %m, ptr %out
  ret {} undef
}

define internal {} @gradient_square(ptr %x, ptr %dx, ptr %out, ptr %dout, {} %tape) {
entry:
  %v = load double, ptr %x
  %d = load double, ptr %dout
  store double 0.000000e+00, ptr %dout
  %t = fmul double %d, %v
  %t2 = fmul double %t, 2.000000e+00
  %o = load double, ptr %dx
  %n = fadd double %o, %t2
  store double %n, ptr %dx
  ret {} undef
}

declare !enzyme_augment !{ptr @augment_square} !enzyme_gradient !{ptr @gradient_square} void @square(ptr %x, ptr %out)

define void @caller(ptr %x, ptr %out) {
entry:
  call void @square(ptr %x, ptr %out)
  ret void
}

define void @top(ptr %x, ptr %out) {
entry:
  call void @caller(ptr %x, ptr %out)
  %v = load double, ptr %out
  %m = fmul double %v, %v
  store double %m, ptr %out
  ret void
}

define void @dtop(ptr %x, ptr %dx, ptr %out, ptr %dout) {
entry:
  call void (...) @__enzyme_autodiff(ptr @top, ptr %x, ptr %dx, ptr %out, ptr %dout)
  ret void
}

declare void @__enzyme_autodiff(...)

; CHECK: define internal {} @augmented_caller(ptr %x, ptr %"x'", ptr %out, ptr %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca {}, align 8
; CHECK-NEXT:   %_augmented = call {} @augment_square(ptr %x, ptr %"x'", ptr %out, ptr %"out'")
; CHECK-NEXT:   %1 = load {}, ptr %0, align 1
; CHECK-NEXT:   ret {} %1
; CHECK-NEXT: }

; CHECK: define internal void @diffecaller(ptr %x, ptr %"x'", ptr %out, ptr %"out'", {} %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertentry
; CHECK: invertentry:
; CHECK-NEXT:   %0 = call {} @gradient_square(ptr %x, ptr %"x'", ptr %out, ptr %"out'", {} undef)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

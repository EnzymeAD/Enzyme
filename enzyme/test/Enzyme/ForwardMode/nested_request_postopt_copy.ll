; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; Lowering the nested request in @inner inlines it into @wrap. @wrap must not
; be inlined further into @h or @g while its own request is still unlowered.

define double @g(double %x) {
  %a = call double @wrap(double %x)
  ret double %a
}

define double @h(double %x) {
  %b = call double (...) @__enzyme_fwddiff(ptr @wrap, double %x, double 1.0)
  %a = call double @wrap(double %x)
  %r = fadd double %a, %b
  ret double %r
}

define double @wrap(double %x) {
  %a = call double @inner(double %x)
  %b = call double (...) @__enzyme_fwddiff(ptr @inner, double %x, double 1.0)
  %r = fadd double %a, %b
  ret double %r
}

define double @inner(double %x) {
  %r = call double (...) @__enzyme_autodiff(ptr @sq, double %x)
  ret double %r
}

define double @sq(double %x) {
  %r = fmul double %x, %x
  ret double %r
}

declare double @__enzyme_fwddiff(...)
declare double @__enzyme_autodiff(...)

; CHECK: define double @g(double %x)
; CHECK-NOT: call {{.*}}@__enzyme_fwddiff(
; CHECK-NOT: call {{.*}}@__enzyme_autodiff(
; CHECK: declare double @__enzyme_fwddiff(...)
; CHECK-NOT: call {{.*}}@__enzyme_fwddiff(
; CHECK-NOT: call {{.*}}@__enzyme_autodiff(

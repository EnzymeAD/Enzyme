; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s; fi

; The frontend marked the pointer handed to the type hint readonly (it was a
; const argument). Enzyme marks a hint's pointer arguments readnone, and the
; verifier rejects a parameter that carries both.

declare void @__enzyme_integer(ptr, i64)

declare double @__enzyme_autodiff(ptr, ...)

@enzyme_const = global i32 0

define double @f(ptr %x, ptr %idx) {
entry:
  call void @__enzyme_integer(ptr readonly %idx, i64 8)
  %i = load i64, ptr %idx, align 8
  %p = getelementptr inbounds double, ptr %x, i64 %i
  %v = load double, ptr %p, align 8
  %r = fmul double %v, %v
  ret double %r
}

define double @df(ptr %x, ptr %dx, ptr %idx) {
entry:
  %r = call double (ptr, ...) @__enzyme_autodiff(ptr @f, ptr %x, ptr %dx, ptr @enzyme_const, ptr %idx)
  ret double %r
}

; CHECK: define internal void @diffef(ptr nocapture readonly %x, ptr nocapture %"x'", ptr nocapture readonly %idx, double %differeturn)
; CHECK-NOT: __enzyme_integer
; CHECK:   %"p'ipg" = getelementptr inbounds double, ptr %"x'", i64 %i
; CHECK:   ret void

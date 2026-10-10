; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme" -enzyme-preopt=false -S | FileCheck %s; fi

; A custom forward derivative is used even if an argument is a constant
; pointer (e.g. a Fortran by-reference flag). Its shadow is passed as null.

@__enzyme_register_derivative_g = global [2 x ptr] [ptr @g, ptr @g_fwd]

declare double @g(ptr %x, ptr %n)

declare { double, double } @g_fwd(ptr %x, ptr %dx, ptr %n, ptr %dn)

define double @f(ptr %x, ptr %n) {
entry:
  %r = call double @g(ptr %x, ptr %n)
  ret double %r
}

define double @caller(ptr %x, ptr %dx, ptr %n) {
entry:
  %r = call double (...) @__enzyme_fwddiff(ptr @f, ptr %x, ptr %dx, metadata !"enzyme_const", ptr %n)
  ret double %r
}

declare double @__enzyme_fwddiff(...)

; CHECK: define internal double @fwddiffef(ptr %x, ptr %"x'", ptr %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @fixderivative_g(ptr %x, ptr %"x'", ptr %n)
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

; CHECK: define internal double @fixderivative_g(ptr %x, ptr %"x'", ptr %n)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call { double, double } @g_fwd(ptr %x, ptr %"x'", ptr %n, ptr null)
; CHECK-NEXT:   %1 = extractvalue { double, double } %0, 1
; CHECK-NEXT:   ret double %1
; CHECK-NEXT: }

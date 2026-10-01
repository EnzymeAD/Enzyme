; RUN: %opt < %s %newLoadEnzyme -passes="preserve-nvvm,globaldce,strip-dead-prototypes,enzyme" -enzyme-preopt=false -S | FileCheck %s

; A custom derivative that is only declared in this module (defined in another
; translation unit, as for a Fortran module procedure used from a module file)
; is kept until Enzyme runs: once the registration is erased only the
; enzyme_derivative metadata refers to it, and the optimizations between
; preserve-nvvm and enzyme would otherwise drop it as a dead prototype.

declare double @f(double)
declare { double, double } @df(double, double)

@f.__enzyme_register_derivative = internal global { ptr, ptr } { ptr @f, ptr @df }

declare double @__enzyme_fwddiff(ptr, ...)

define double @g(double %x) {
entry:
  %r = call double @f(double %x)
  ret double %r
}

define double @dg(double %x) {
entry:
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @g, double %x, double 1.0)
  ret double %r
}

; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @df]
; CHECK: define internal double @fwddiffeg(double %x, double %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:.+]] = call fast double @fixderivative_f(double %x, double %"x'")
; CHECK-NEXT:   ret double %[[r]]

; CHECK: define internal double @fixderivative_f(double %0, double %1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %2 = call { double, double } @df(double %0, double %1)

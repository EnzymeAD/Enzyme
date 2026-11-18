; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -preserve-nvvm -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme,verify" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 100.000000 2

; A registered forward-mode derivative may return just the derivative (the
; signature of the function Enzyme would generate when the primal is unused)
; or, for a function with a constant return, the primal itself; neither form
; must be mistaken for the {primal, derivative} pair, and the second is wrapped
; so its unused result is dropped.

@__enzyme_register_derivative_square = global [2 x ptr] [ptr @square, ptr @derivative_square]
@__enzyme_register_derivative_count = global [2 x ptr] [ptr @count, ptr @derivative_count]
@calls = global i32 0
@.fmt = private unnamed_addr constant [7 x i8] c"%f %d\0A\00"

declare double @__enzyme_fwddiff(...)
declare i32 @__enzyme_fwddiff_int(...)
declare i32 @printf(ptr, ...)

define double @square(ptr %x) {
entry:
  %v = load double, ptr %x
  %r = fmul double %v, %v
  ret double %r
}

define double @derivative_square(ptr %x, ptr %dx) {
entry:
  %c = load i32, ptr @calls
  %inc = add i32 %c, 1
  store i32 %inc, ptr @calls
  ret double 1.000000e+02
}

define i32 @count(ptr %x) {
entry:
  %v = load double, ptr %x
  %i = fptosi double %v to i32
  ret i32 %i
}

define i32 @derivative_count(ptr %x, ptr %dx) {
entry:
  %c = load i32, ptr @calls
  %inc = add i32 %c, 1
  store i32 %inc, ptr @calls
  ret i32 7
}

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  %x = alloca double
  %dx = alloca double
  store double 3.000000e+00, ptr %x
  store double 1.000000e+00, ptr %dx
  ; CHECK: call fast double @derivative_square(ptr %x, ptr %dx)
  %d = call double (...) @__enzyme_fwddiff(ptr @square, metadata !"enzyme_dup", ptr %x, ptr %dx)
  ; CHECK: call void @fixderivative_count(ptr %x, ptr %dx)
  %n = call i32 (...) @__enzyme_fwddiff_int(ptr @count, metadata !"enzyme_dup", ptr %x, ptr %dx)
  %calls = load i32, ptr @calls
  call i32 (ptr, ...) @printf(ptr @.fmt, double %d, i32 %calls)
  ret i32 0
}

; CHECK-LABEL: define internal void @fixderivative_count(ptr %x, ptr %dx)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call i32 @derivative_count(ptr %x, ptr %dx)
; CHECK-NEXT:   ret void

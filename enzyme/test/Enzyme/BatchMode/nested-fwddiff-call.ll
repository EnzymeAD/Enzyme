; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | %lli - | FileCheck %s --check-prefix=EXEC

; Batching a function whose body calls another defined function rebuilds each
; vector argument of the inner call lane by lane. The partially built aggregate
; must be threaded through the inserts; passing the initial undef instead hands
; the batched callee no lanes at all. The inner call here is itself a forward
; mode derivative, as in the report.

declare [2 x double] @__enzyme_batch(...)
declare double @__enzyme_fwddiff(...)
declare i32 @printf(ptr, ...)

@format = private constant [11 x i8] c"%.1f %.1f\0A\00"

define double @cube(double %scale, double %x) {
entry:
  %square = fmul double %x, %x
  %cube = fmul double %square, %x
  %scaled = fmul double %scale, %cube
  ret double %scaled
}

define double @cube_derivative(double %scale, double %x) {
entry:
  %derivative = call double (...) @__enzyme_fwddiff(ptr @cube, double %scale, double 0.0, double %x, double 1.0)
  ret double %derivative
}

define [2 x double] @batched_derivative(double %scale, double %x0, double %x1) {
entry:
  %result = call [2 x double] (...) @__enzyme_batch(ptr @cube_derivative, metadata !"enzyme_width", i64 2, metadata !"enzyme_scalar", double %scale, metadata !"enzyme_vector", double %x0, double %x1)
  ret [2 x double] %result
}

define i32 @main() {
entry:
  %result = call [2 x double] @batched_derivative(double 2.0, double 1.0, double 3.0)
  %lane0 = extractvalue [2 x double] %result, 0
  %lane1 = extractvalue [2 x double] %result, 1
  call i32 (ptr, ...) @printf(ptr @format, double %lane0, double %lane1)
  ret i32 0
}

; CHECK-LABEL: define internal [2 x double] @batch_cube_derivative(double %scale, [2 x double] %x)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %unwrap.x0 = extractvalue [2 x double] %x, 0
; CHECK-NEXT:   %unwrap.x1 = extractvalue [2 x double] %x, 1
; CHECK-NEXT:   %[[LANE0:.+]] = insertvalue [2 x double] undef, double %unwrap.x0, 0
; CHECK-NEXT:   %[[LANES:.+]] = insertvalue [2 x double] %[[LANE0]], double %unwrap.x1, 1
; CHECK-NEXT:   %[[CALL:.+]] = call [2 x double] @batch_fwddiffecube(double %scale, double 0.000000e+00, [2 x double] %[[LANES]], double 1.000000e+00)

; EXEC: 6.0 54.0

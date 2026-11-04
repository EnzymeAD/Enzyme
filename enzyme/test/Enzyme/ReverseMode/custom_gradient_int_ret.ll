; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -preserve-nvvm -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; Verify that __enzyme_register_gradient works on functions returning a constant
; non-void integer status code (retType == CONSTANT while subretType == DUP_ARG
; under intAreConstant=false), both when the caller ignores the status return,
; when the caller checks the status return, and when differentiated directly.

@.str = private unnamed_addr constant [16 x i8] c"%.4f %.4f %.4f\0A\00", align 1
@enzyme_dup = global i32 0, align 4
@enzyme_const = global i32 0, align 4

@__enzyme_register_gradient_sq = global [3 x ptr] [ptr @my_square_status, ptr @aug_my_square_status, ptr @rev_my_square_status], align 16

declare void @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)
declare noalias ptr @malloc(i64)
declare void @free(ptr)

define i32 @my_square_status(ptr nocapture readonly %x, ptr nocapture writeonly %out) noinline {
entry:
  %v = load double, ptr %x, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %out, align 8
  ret i32 0
}

define { ptr, i32 } @aug_my_square_status(ptr nocapture readonly %x, ptr nocapture readnone %dx, ptr nocapture writeonly %out, ptr nocapture readnone %dout) {
entry:
  %tape = call noalias ptr @malloc(i64 8)
  %v = load double, ptr %x, align 8
  store double %v, ptr %tape, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %out, align 8
  %s0 = insertvalue { ptr, i32 } undef, ptr %tape, 0
  %s1 = insertvalue { ptr, i32 } %s0, i32 0, 1
  ret { ptr, i32 } %s1
}

; Custom reverse injects 10 * v * dout so we know the custom rule ran (not body AD).
define void @rev_my_square_status(ptr nocapture readnone %x, ptr nocapture %dx, ptr nocapture readnone %out, ptr nocapture %dout, ptr %tape) {
entry:
  %v = load double, ptr %tape, align 8
  call void @free(ptr %tape)
  %do = load double, ptr %dout, align 8
  store double 0.000000e+00, ptr %dout, align 8
  %ten_v = fmul double %v, 1.000000e+01
  %contrib = fmul double %ten_v, %do
  %old = load double, ptr %dx, align 8
  %upd = fadd double %old, %contrib
  store double %upd, ptr %dx, align 8
  ret void
}

define void @caller_unused_ret(ptr %x, ptr %out) {
entry:
  %status = call i32 @my_square_status(ptr %x, ptr %out)
  ret void
}

define double @caller_used_ret(ptr %x) {
entry:
  %tmp = alloca double, align 8
  store double 0.000000e+00, ptr %tmp, align 8
  %status = call i32 @my_square_status(ptr %x, ptr %tmp)
  %ok = icmp eq i32 %status, 0
  %val = load double, ptr %tmp, align 8
  %res = select i1 %ok, double %val, double 0.000000e+00
  ret double %res
}

; CHECK-LABEL: define internal void @diffecaller_unused_ret(
; CHECK: call { ptr, i32 } @aug_my_square_status(
; CHECK: call void @rev_my_square_status(

; CHECK-LABEL: define internal void @diffecaller_used_ret(
; CHECK: call { ptr, i32 } @aug_my_square_status(
; CHECK: call void @rev_my_square_status(

define i32 @main() {
entry:
  %x = alloca double, align 8
  %dx1 = alloca double, align 8
  %dx2 = alloca double, align 8
  %dx3 = alloca double, align 8
  %out = alloca double, align 8
  %dout = alloca double, align 8
  store double 3.000000e+00, ptr %x, align 8
  store double 0.000000e+00, ptr %dx1, align 8
  store double 0.000000e+00, ptr %dx2, align 8
  store double 0.000000e+00, ptr %dx3, align 8
  store double 0.000000e+00, ptr %out, align 8
  store double 2.000000e+00, ptr %dout, align 8

  %dup = load i32, ptr @enzyme_dup, align 4
  %cst = load i32, ptr @enzyme_const, align 4

  ; 1. Unused status return inside caller: dx1 = 10 * 3 * 2 = 60
  call void (...) @__enzyme_autodiff(ptr @caller_unused_ret, i32 %dup, ptr %x, ptr %dx1, i32 %dup, ptr %out, ptr %dout)

  ; 2. Used status return inside caller: dx2 = 10 * 3 * 1 = 30
  call void (...) @__enzyme_autodiff(ptr @caller_used_ret, i32 %dup, ptr %x, ptr %dx2)

  ; 3. Direct top-level ReverseModeCombined on my_square_status with enzyme_const ret: dx3 = 10 * 3 * 5 = 150
  store double 5.000000e+00, ptr %dout, align 8
  call void (...) @__enzyme_autodiff(ptr @my_square_status, i32 %cst, i32 %dup, ptr %x, ptr %dx3, i32 %dup, ptr %out, ptr %dout)

  %r1 = load double, ptr %dx1, align 8
  %r2 = load double, ptr %dx2, align 8
  %r3 = load double, ptr %dx3, align 8
  call i32 (ptr, ...) @printf(ptr @.str, double %r1, double %r2, double %r3)
  ret i32 0
}

; EVAL: 60.0000 30.0000 150.0000

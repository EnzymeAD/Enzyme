; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -preserve-nvvm -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="preserve-nvvm,enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; Verify that __enzyme_register_derivative works on functions returning a
; constant non-void integer status code, both when the caller checks the
; status return and when the caller ignores it (and with a constant scalar arg).

@.str = private unnamed_addr constant [16 x i8] c"%.4f %.4f %.4f\0A\00", align 1
@enzyme_dup = global i32 0, align 4
@enzyme_const = global i32 0, align 4

@__enzyme_register_derivative_sq = global [2 x ptr] [ptr @my_square_status, ptr @fwd_my_square_status], align 16

declare double @__enzyme_fwddiff(...)
declare void @__enzyme_fwddiff_void(...)
declare i32 @printf(ptr, ...)

define i32 @my_square_status(ptr nocapture readonly %x, ptr nocapture writeonly %out, i32 %flag) noinline {
entry:
  %v = load double, ptr %x, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %out, align 8
  ret i32 %flag
}

; Custom forward rule sets dout = 10 * v * dx and returns status = flag.
define i32 @fwd_my_square_status(ptr nocapture readonly %x, ptr nocapture readonly %dx, ptr nocapture writeonly %out, ptr nocapture writeonly %dout, i32 %flag, i32 %dflag) {
entry:
  %v = load double, ptr %x, align 8
  %dv = load double, ptr %dx, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %out, align 8
  %ten_v = fmul double %v, 1.000000e+01
  %dres = fmul double %ten_v, %dv
  store double %dres, ptr %dout, align 8
  ret i32 %flag
}

define double @caller_used_ret(ptr %x) {
entry:
  %tmp = alloca double, align 8
  %status = call i32 @my_square_status(ptr %x, ptr %tmp, i32 0)
  %ok = icmp eq i32 %status, 0
  %val = load double, ptr %tmp, align 8
  %res = select i1 %ok, double %val, double 0.000000e+00
  ret double %res
}

define void @caller_unused_ret(ptr %x, ptr %out) {
entry:
  %status = call i32 @my_square_status(ptr %x, ptr %out, i32 0)
  ret void
}

; Also test a no-constant-arg function returning i32 directly.
@__enzyme_register_derivative_sq2 = global [2 x ptr] [ptr @my_square_nocst, ptr @fwd_my_square_nocst], align 16

define i32 @my_square_nocst(ptr nocapture readonly %x, ptr nocapture writeonly %out) noinline {
entry:
  %v = load double, ptr %x, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %out, align 8
  ret i32 0
}

define i32 @fwd_my_square_nocst(ptr nocapture readonly %x, ptr nocapture readonly %dx, ptr nocapture writeonly %out, ptr nocapture writeonly %dout) {
entry:
  %v = load double, ptr %x, align 8
  %dv = load double, ptr %dx, align 8
  %sq = fmul double %v, %v
  store double %sq, ptr %out, align 8
  %ten_v = fmul double %v, 1.000000e+01
  %dres = fmul double %ten_v, %dv
  store double %dres, ptr %dout, align 8
  ret i32 0
}

define double @caller_nocst_both(ptr %x) {
entry:
  %tmp1 = alloca double, align 8
  %tmp2 = alloca double, align 8
  %s_unused = call i32 @my_square_nocst(ptr %x, ptr %tmp1)
  %s_used = call i32 @my_square_nocst(ptr %x, ptr %tmp2)
  %ok = icmp eq i32 %s_used, 0
  %v1 = load double, ptr %tmp1, align 8
  %v2 = load double, ptr %tmp2, align 8
  %sum = fadd double %v1, %v2
  %res = select i1 %ok, double %sum, double 0.000000e+00
  ret double %res
}

; CHECK-LABEL: define internal double @fwddiffecaller_used_ret(
; CHECK: call {{(i32 )?}}@fixderivative_my_square_status(

; CHECK-LABEL: define internal double @fwddiffecaller_nocst_both(
; CHECK: call void @fixderivative_my_square_nocst(
; CHECK: call i32 @fwd_my_square_nocst(

define i32 @main() {
entry:
  %x = alloca double, align 8
  %dx = alloca double, align 8
  %out = alloca double, align 8
  %dout = alloca double, align 8
  store double 3.000000e+00, ptr %x, align 8
  store double 2.000000e+00, ptr %dx, align 8
  store double 0.000000e+00, ptr %out, align 8
  store double 0.000000e+00, ptr %dout, align 8

  %dup = load i32, ptr @enzyme_dup, align 4

  ; 1. Used status with constant i32 arg: 10 * 3 * 2 = 60
  %r1 = call double (...) @__enzyme_fwddiff(ptr @caller_used_ret, i32 %dup, ptr %x, ptr %dx)

  ; 2. Unused status with constant i32 arg: dout = 10 * 3 * 2 = 60
  %r2_unused = call double (...) @__enzyme_fwddiff(ptr @caller_unused_ret, i32 %dup, ptr %x, ptr %dx, i32 %dup, ptr %out, ptr %dout)
  %r2 = load double, ptr %dout, align 8

  ; 3. Both unused + used status without constant arg: 60 + 60 = 120
  %r3 = call double (...) @__enzyme_fwddiff(ptr @caller_nocst_both, i32 %dup, ptr %x, ptr %dx)

  call i32 (ptr, ...) @printf(ptr @.str, double %r1, double %r2, double %r3)
  ret i32 0
}

; EVAL: 60.0000 60.0000 120.0000

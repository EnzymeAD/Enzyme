; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -enzyme-strict-aliasing=0 -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-strict-aliasing=0 -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-strict-aliasing=0 -enzyme-loose-types=1 -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -enzyme-strict-aliasing=0 -S | %lli - | FileCheck %s --check-prefix=EVAL

; Verify that reverse mode differentiates:
; 1. A loop-carried double spilled across an exception frame via double <-> i64
;    bitcasts and volatile stack stores/loads under -enzyme-strict-aliasing=0.
; 2. A volatile i64 spill store of a bitcast floating-point BinaryOperator whose
;    catch-block reload was eliminated, without mistyping the store as Integer
;    or asserting in GradientUtils::invertPointerM on the feeding fsub.

@.str = private unnamed_addr constant [16 x i8] c"%.4f %.4f %.4f\0A\00", align 1
@enzyme_dup = global i32 0, align 4
@enzyme_const = global i32 0, align 4

declare void @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)

define void @spilled_loop_carried(ptr nocapture %out, ptr nocapture readonly %m, i64 %n) {
entry:
  %phic = alloca i64, align 8
  %dead_spill = alloca i64, align 8
  %m0 = load double, ptr %m, align 8
  %m1_ptr = getelementptr inbounds double, ptr %m, i64 1
  %m1 = load double, ptr %m1_ptr, align 8
  %c_low_global = fsub double %m0, %m1
  %dead_bits = bitcast double %c_low_global to i64
  store volatile i64 %dead_bits, ptr %dead_spill, align 8
  br label %outer_header

outer_header:
  %i = phi i64 [ 0, %entry ], [ %i_next, %outer_latch ]
  %carried = phi double [ %c_low_global, %entry ], [ %reloaded_d, %outer_latch ]
  %scaled = fmul double %carried, 8.000000e-01
  %acc_ptr = getelementptr inbounds double, ptr %out, i64 %i
  store double %scaled, ptr %acc_ptr, align 8
  br label %inner_header

inner_header:
  %j = phi i64 [ 0, %outer_header ], [ %j_next, %inner_latch ]
  %inner_phi = phi double [ %carried, %outer_header ], [ %reloaded_d, %inner_latch ]
  %bits = bitcast double %inner_phi to i64
  store volatile i64 %bits, ptr %phic, align 8
  br label %inner_latch

inner_latch:
  %reloaded_bits = load i64, ptr %phic, align 8
  %reloaded_d = bitcast i64 %reloaded_bits to double
  %j_next = add nuw nsw i64 %j, 1
  %j_cmp = icmp slt i64 %j_next, 2
  br i1 %j_cmp, label %inner_header, label %outer_latch

outer_latch:
  %i_next = add nuw nsw i64 %i, 1
  %i_cmp = icmp slt i64 %i_next, %n
  br i1 %i_cmp, label %outer_header, label %exit

exit:
  ret void
}

; CHECK-LABEL: define internal void @diffespilled_loop_carried(
; CHECK: fadd fast double

define i32 @main() {
entry:
  %out = alloca [2 x double], align 8
  %dout = alloca [2 x double], align 8
  %m = alloca [2 x double], align 8
  %dm = alloca [2 x double], align 8
  store double 0.000000e+00, ptr %out, align 8
  %out1 = getelementptr inbounds double, ptr %out, i64 1
  store double 0.000000e+00, ptr %out1, align 8
  store double 1.000000e+00, ptr %dout, align 8
  %dout1 = getelementptr inbounds double, ptr %dout, i64 1
  store double 1.000000e+00, ptr %dout1, align 8

  store double 9.000000e+00, ptr %m, align 8
  %m1 = getelementptr inbounds double, ptr %m, i64 1
  store double 4.000000e+00, ptr %m1, align 8
  store double 0.000000e+00, ptr %dm, align 8
  %dm1 = getelementptr inbounds double, ptr %dm, i64 1
  store double 0.000000e+00, ptr %dm1, align 8

  %dup = load i32, ptr @enzyme_dup, align 4
  %cst = load i32, ptr @enzyme_const, align 4
  call void (...) @__enzyme_autodiff(ptr @spilled_loop_carried, i32 %dup, ptr %out, ptr %dout, i32 %dup, ptr %m, ptr %dm, i32 %cst, i64 2)

  ; out[0] = 4.0, out[1] = 4.0; dm0 = 1.6000, dm1 = -1.6000
  %o0 = load double, ptr %out, align 8
  %res0 = load double, ptr %dm, align 8
  %res1 = load double, ptr %dm1, align 8
  call i32 (ptr, ...) @printf(ptr @.str, double %o0, double %res0, double %res1)
  ret i32 0
}

; EVAL: 4.0000 1.6000 -1.6000

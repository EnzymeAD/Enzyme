; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL
; EVAL: 23.00 10.00 -11.00 -9.00 0.60 0.80

; Verify that C99 Annex G complex float builtins (__mulsc3, __divsc3) lowered
; with an x86-64 <2 x float> vector return type differentiate cleanly in
; scalar/batched forward mode and reverse mode, and that subroutine/value
; extractions do not fold past cached aggregate insertions in split reverse
; mode.

@.fmt = private unnamed_addr constant [31 x i8] c"%.2f %.2f %.2f %.2f %.2f %.2f\0A\00", align 1
@enzyme_width = global i32 0, align 4

declare <2 x float> @__mulsc3(float, float, float, float)
declare <2 x float> @__divsc3(float, float, float, float)
declare double @cabs([2 x double]) nounwind willreturn
declare <2 x float> @__enzyme_fwddiff(...)
declare [2 x <2 x float>] @__enzyme_fwddiff2(...)
declare { float, float, float, float } @__enzyme_autodiff(...)
declare { ptr, double } @__enzyme_augmentfwd(...)
declare void @__enzyme_reverse(...)
declare i32 @printf(ptr, ...)

define <2 x float> @mulsc3_vec(float %xre, float %xim, float %yre, float %yim) {
entry:
  %res = call <2 x float> @__mulsc3(float %xre, float %xim, float %yre, float %yim)
  ret <2 x float> %res
}

define <2 x float> @divsc3_vec(float %xre, float %xim, float %yre, float %yim) {
entry:
  %res = call <2 x float> @__divsc3(float %xre, float %xim, float %yre, float %yim)
  ret <2 x float> %res
}

define float @mulsc3_loss(float %xre, float %xim, float %yre, float %yim) {
entry:
  %res = call <2 x float> @__mulsc3(float %xre, float %xim, float %yre, float %yim)
  %re = extractelement <2 x float> %res, i64 0
  %im = extractelement <2 x float> %res, i64 1
  %sum = fadd float %re, %im
  ret float %sum
}

define float @divsc3_loss(float %xre, float %xim, float %yre, float %yim) {
entry:
  %res = call <2 x float> @__divsc3(float %xre, float %xim, float %yre, float %yim)
  %re = extractelement <2 x float> %res, i64 0
  %im = extractelement <2 x float> %res, i64 1
  %sum = fadd float %re, %im
  ret float %sum
}

define double @cabs_split_insert(ptr %p) {
entry:
  %re = load double, ptr %p, align 8
  %im_ptr = getelementptr inbounds double, ptr %p, i64 1
  %im = load double, ptr %im_ptr, align 8
  %agg0 = insertvalue [2 x double] poison, double %re, 0
  %agg1 = insertvalue [2 x double] %agg0, double %im, 1
  store double 0.000000e+00, ptr %p, align 8
  store double 0.000000e+00, ptr %im_ptr, align 8
  %res = call double @cabs([2 x double] %agg1)
  ret double %res
}

; CHECK-LABEL: define i32 @main(
define i32 @main() {
entry:
  %buf = alloca [2 x double], align 8
  %dbuf = alloca [2 x double], align 8
  store double 3.000000e+00, ptr %buf, align 8
  %buf_im = getelementptr inbounds double, ptr %buf, i64 1
  store double 4.000000e+00, ptr %buf_im, align 8
  store [2 x double] zeroinitializer, ptr %dbuf, align 8
  ; CHECK: call fast <2 x float> @fwddiffemulsc3_vec(
  %f1 = call <2 x float> (...) @__enzyme_fwddiff(ptr @mulsc3_vec, float 2.000000e+00, float 1.000000e+00, float 3.000000e+00, float 2.000000e+00, float 4.000000e+00, float 3.000000e+00, float -5.000000e+00, float -1.000000e+00)
  ; CHECK: call fast [2 x <2 x float>] @fwddiffe2mulsc3_vec(
  %f2 = call [2 x <2 x float>] (...) @__enzyme_fwddiff2(ptr @mulsc3_vec, ptr @enzyme_width, i32 2, float 2.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 2.000000e+00, float -1.000000e+00, float 4.000000e+00, float 3.000000e+00, float -1.000000e+00, float -5.000000e+00, float -1.000000e+00, float 4.000000e+00)
  ; CHECK: call fast <2 x float> @fwddiffedivsc3_vec(
  %fd = call <2 x float> (...) @__enzyme_fwddiff(ptr @divsc3_vec, float 2.000000e+00, float 1.000000e+00, float 3.000000e+00, float 2.000000e+00, float 4.000000e+00, float 3.000000e+00, float -5.000000e+00, float -1.000000e+00)
  ; CHECK: call { float, float, float, float } @diffemulsc3_loss(
  %rm = call { float, float, float, float } (...) @__enzyme_autodiff(ptr @mulsc3_loss, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float -5.000000e+00)
  ; CHECK: call { float, float, float, float } @diffedivsc3_loss(
  %rd = call { float, float, float, float } (...) @__enzyme_autodiff(ptr @divsc3_loss, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float -5.000000e+00)
  ; CHECK: call { ptr, double } @augmented_cabs_split_insert(
  %aug = call { ptr, double } (...) @__enzyme_augmentfwd(ptr @cabs_split_insert, ptr %buf, ptr %dbuf)
  %tape = extractvalue { ptr, double } %aug, 0
  ; CHECK: call void @diffecabs_split_insert(
  call void (...) @__enzyme_reverse(ptr @cabs_split_insert, ptr %buf, ptr %dbuf, double 1.000000e+00, ptr %tape)
  %dre = load double, ptr %dbuf, align 8
  %dbuf_im = getelementptr inbounds double, ptr %dbuf, i64 1
  %dim = load double, ptr %dbuf_im, align 8
  %r0 = extractelement <2 x float> %f1, i64 0
  %r1 = extractelement <2 x float> %f1, i64 1
  %b1 = extractvalue [2 x <2 x float>] %f2, 1
  %r2 = extractelement <2 x float> %b1, i64 0
  %r3 = extractelement <2 x float> %b1, i64 1
  %d0 = fpext float %r0 to double
  %d1 = fpext float %r1 to double
  %d2 = fpext float %r2 to double
  %d3 = fpext float %r3 to double
  call i32 (ptr, ...) @printf(ptr @.fmt, double %d0, double %d1, double %d2, double %d3, double %dre, double %dim)
  ret i32 0
}

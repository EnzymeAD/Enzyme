; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; Verify that simplifyLoad / getAllLoadedValuesFrom rejects loads that straddle
; two adjacent stores ([0..8) and [8..16) read at [4..12)) or extract a nonzero
; byte slice of a wider non-load store ([0..8) read at [4..8)), rather than
; asserting on offset >= suboff or returning the unshifted aggregate store.

@.str = private unnamed_addr constant [11 x i8] c"%.2f %.2f\0A\00", align 1
@enzyme_const = global i32 0, align 4

declare float @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)

define float @dot2_scaled(float %scale, <2 x float> %vec) {
entry:
  %v0 = extractelement <2 x float> %vec, i64 0
  %v1 = extractelement <2 x float> %vec, i64 1
  %sum = fadd float %v0, %v1
  %prod = fmul float %scale, %sum
  ret float %prod
}

; CHECK-LABEL: define float @caller_straddle(
define float @caller_straddle(float %scale, i64 %lo, i64 %hi) {
entry:
  %buf = alloca [16 x i8], align 8
  store i64 %lo, ptr %buf, align 8
  %hi_ptr = getelementptr inbounds i8, ptr %buf, i64 8
  store i64 %hi, ptr %hi_ptr, align 8
  %mid_ptr = getelementptr inbounds i8, ptr %buf, i64 4
  %vec = load <2 x float>, ptr %mid_ptr, align 4
  ; CHECK: %[[AGG:.+]] = call { float } @diffedot2_scaled(float %scale, <2 x float> %vec, float 1.000000e+00)
  ; CHECK-NEXT: %[[GRAD:.+]] = extractvalue { float } %[[AGG]], 0
  ; CHECK-NEXT: ret float %[[GRAD]]
  %grad = call float (...) @__enzyme_autodiff(ptr @dot2_scaled, float %scale, ptr @enzyme_const, <2 x float> %vec)
  ret float %grad
}

define float @mul_const(float %x, float %c) {
entry:
  %r = fmul float %x, %c
  ret float %r
}

; CHECK-LABEL: define float @caller_subslice(
define float @caller_subslice(float %x, i64 %packed) {
entry:
  %buf = alloca i64, align 8
  store i64 %packed, ptr %buf, align 8
  %upper_ptr = getelementptr inbounds i8, ptr %buf, i64 4
  %c = load float, ptr %upper_ptr, align 4
  ; CHECK: %[[AGG2:.+]] = call { float } @diffemul_const(float %x, float %c, float 1.000000e+00)
  ; CHECK-NEXT: %[[GRAD2:.+]] = extractvalue { float } %[[AGG2]], 0
  ; CHECK-NEXT: ret float %[[GRAD2]]
  %grad = call float (...) @__enzyme_autodiff(ptr @mul_const, float %x, ptr @enzyme_const, float %c)
  ret float %grad
}

define i32 @main() {
entry:
  ; Pack float 2.0 (0x40000000) in upper 32 bits of %lo and float 5.0 (0x40a00000) in lower 32 bits of %hi.
  %lo = shl i64 1073741824, 32
  %hi = add i64 1084227584, 0
  ; EVAL: 7.00 3.00
  %g1 = call float @caller_straddle(float 4.000000e+00, i64 %lo, i64 %hi)
  ; Pack float 3.0 (0x40400000) in upper 32 bits of %packed.
  %packed = shl i64 1077936128, 32
  %g2 = call float @caller_subslice(float 9.000000e+00, i64 %packed)
  %d1 = fpext float %g1 to double
  %d2 = fpext float %g2 to double
  call i32 (ptr, ...) @printf(ptr @.str, double %d1, double %d2)
  ret i32 0
}

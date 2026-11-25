; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL

; EVAL: flags=6 dx=12.000000

@.fmt = private unnamed_addr constant [16 x i8] c"flags=%d dx=%f\0A\00", align 1
@enzyme_const = external global i32, align 4

declare double @__enzyme_autodiff(...)
declare i32 @printf(ptr, ...)

define double @mark_and_square(double %x, ptr %flags) {
entry:
  %prev_or = atomicrmw or ptr %flags, i32 7 seq_cst, align 4
  %prev_xor = atomicrmw xor ptr %flags, i32 1 seq_cst, align 4
  %combined = add i32 %prev_or, %prev_xor
  %nonneg = icmp sge i32 %combined, 0
  %sq = fmul double %x, %x
  %scaled = fmul double %sq, 2.000000e+00
  %res = select i1 %nonneg, double %scaled, double 0.000000e+00
  ret double %res
}

define i32 @main() {
entry:
  %flags = alloca i32, align 4
  store i32 0, ptr %flags, align 4
  %const_code = load i32, ptr @enzyme_const, align 4
  %dx = call double (...) @__enzyme_autodiff(ptr @mark_and_square, double 3.000000e+00, i32 %const_code, ptr %flags)
  %final_flags = load i32, ptr %flags, align 4
  call i32 (ptr, ...) @printf(ptr @.fmt, i32 %final_flags, double %dx)
  ret i32 0
}

; CHECK-LABEL: define internal { double } @diffemark_and_square(
; CHECK: atomicrmw or ptr %flags, i32 7 seq_cst, align 4
; CHECK: atomicrmw xor ptr %flags, i32 1 seq_cst, align 4

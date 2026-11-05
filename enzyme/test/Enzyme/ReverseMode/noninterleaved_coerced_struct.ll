; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; EVAL: 26.0 12.0 18.0

@enzyme_const = global i32 0, align 4
@enzyme_dup = global i32 0, align 4
@.fmt = private unnamed_addr constant [16 x i8] c"%.1f %.1f %.1f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare double @__enzyme_autodiff(...)

define double @dot_coerced_view(ptr %x_ptr, i32 %x_len, ptr %w_ptr, i32 %w_len, double %alpha) {
entry:
  %x0 = load double, ptr %x_ptr, align 8
  %x1_ptr = getelementptr inbounds double, ptr %x_ptr, i64 1
  %x1 = load double, ptr %x1_ptr, align 8
  %w0 = load double, ptr %w_ptr, align 8
  %w1_ptr = getelementptr inbounds double, ptr %w_ptr, i64 1
  %w1 = load double, ptr %w1_ptr, align 8
  %p0 = fmul double %x0, %w0
  %p1 = fmul double %x1, %w1
  %sum = fadd double %p0, %p1
  %n_sum = add nsw i32 %x_len, %w_len
  %scale = sitofp i32 %n_sum to double
  %scaled = fmul double %sum, %scale
  %res = fmul double %scaled, %alpha
  ret double %res
}

; CHECK-LABEL: define double @grad_coerced_both(
define double @grad_coerced_both(ptr %x_ptr, i32 %x_len, ptr %dx_ptr, i32 %dx_len, ptr %w_ptr, i32 %w_len, double %alpha) {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
  %cst = load i32, ptr @enzyme_const, align 4
; CHECK: call {{(fast )?}}{ double } @diffedot_coerced_view(ptr %x_ptr, ptr %dx_ptr, i32 %x_len, i32 %dx_len, ptr %w_ptr, i32 %w_len, double %alpha, double 1.000000e+00)
  %da = call double (...) @__enzyme_autodiff(ptr @dot_coerced_view, i32 %dup, ptr %x_ptr, i32 %x_len, ptr %dx_ptr, i32 %dx_len, i32 %cst, ptr %w_ptr, i32 %w_len, double %alpha)
  ret double %da
}

define i32 @main() {
entry:
  %x_buf = alloca [2 x double], align 16
  %dx_buf = alloca [2 x double], align 16
  %w_buf = alloca [2 x double], align 16

  %x0 = getelementptr inbounds [2 x double], ptr %x_buf, i64 0, i64 0
  %x1 = getelementptr inbounds [2 x double], ptr %x_buf, i64 0, i64 1
  store double 2.000000e+00, ptr %x0, align 8
  store double 3.000000e+00, ptr %x1, align 8

  %dx0 = getelementptr inbounds [2 x double], ptr %dx_buf, i64 0, i64 0
  %dx1 = getelementptr inbounds [2 x double], ptr %dx_buf, i64 0, i64 1
  store double 0.000000e+00, ptr %dx0, align 8
  store double 0.000000e+00, ptr %dx1, align 8

  %w0 = getelementptr inbounds [2 x double], ptr %w_buf, i64 0, i64 0
  %w1 = getelementptr inbounds [2 x double], ptr %w_buf, i64 0, i64 1
  store double 1.000000e+00, ptr %w0, align 8
  store double 1.500000e+00, ptr %w1, align 8

  %da = call double @grad_coerced_both(ptr %x0, i32 2, ptr %dx0, i32 2, ptr %w0, i32 2, double 3.000000e+00)
  %dx0_v = load double, ptr %dx0, align 8
  %dx1_v = load double, ptr %dx1, align 8
  call i32 (ptr, ...) @printf(ptr @.fmt, double %da, double %dx0_v, double %dx1_v)
  ret i32 0
}

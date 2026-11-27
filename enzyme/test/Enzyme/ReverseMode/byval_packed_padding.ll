; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi
; EVAL: 26.0 12.0

; Packed C++ ABI struct layout `<{ ptr, i32, [4 x i8] }>` with explicit
; synthetic `[4 x i8]` alignment tail padding on `byval` call sites while
; register-coerced callee parameters omit the tail padding.

%struct.PackedView = type <{ ptr, i32, [4 x i8] }>

@enzyme_const = global i32 0, align 4
@enzyme_dup = global i32 0, align 4
@.fmt = private unnamed_addr constant [11 x i8] c"%.1f %.1f\0A\00", align 1

declare i32 @printf(ptr, ...)
declare double @__enzyme_autodiff(...)

define double @dot_packed_view(ptr %x_ptr, i32 %x_len, ptr %w_ptr, i32 %w_len, double %alpha) {
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

; CHECK-LABEL: define double @grad_byval_packed(
define double @grad_byval_packed(ptr %x, ptr %dx, ptr %w, double %alpha) {
entry:
  %dup = load i32, ptr @enzyme_dup, align 4
  %cst = load i32, ptr @enzyme_const, align 4
; CHECK: call {{(fast )?}}{ double } @diffedot_packed_view(ptr {{.+}}, ptr {{.+}}, i32 {{.+}}, ptr {{.+}}, i32 {{.+}}, double %alpha, double 1.000000e+00)
  %da = call double (...) @__enzyme_autodiff(ptr @dot_packed_view, i32 %dup, ptr byval(%struct.PackedView) align 8 %x, ptr byval(%struct.PackedView) align 8 %dx, i32 %cst, ptr byval(%struct.PackedView) align 8 %w, double %alpha)
  ret double %da
}

define i32 @main() {
entry:
  %x_buf = alloca [2 x double], align 16
  %dx_buf = alloca [2 x double], align 16
  %w_buf = alloca [2 x double], align 16
  %x_view = alloca %struct.PackedView, align 8
  %dx_view = alloca %struct.PackedView, align 8
  %w_view = alloca %struct.PackedView, align 8

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

  store ptr %x0, ptr %x_view, align 8
  %xl = getelementptr inbounds %struct.PackedView, ptr %x_view, i64 0, i32 1
  store i32 2, ptr %xl, align 4
  %xpad = getelementptr inbounds i8, ptr %x_view, i64 12
  store i32 777, ptr %xpad, align 4

  store ptr %dx0, ptr %dx_view, align 8
  %dxl = getelementptr inbounds %struct.PackedView, ptr %dx_view, i64 0, i32 1
  store i32 2, ptr %dxl, align 4
  %dxpad = getelementptr inbounds i8, ptr %dx_view, i64 12
  store i32 888, ptr %dxpad, align 4

  store ptr %w0, ptr %w_view, align 8
  %wl = getelementptr inbounds %struct.PackedView, ptr %w_view, i64 0, i32 1
  store i32 2, ptr %wl, align 4
  %wpad = getelementptr inbounds i8, ptr %w_view, i64 12
  store i32 999, ptr %wpad, align 4

  %g1 = call double @grad_byval_packed(ptr %x_view, ptr %dx_view, ptr %w_view, double 3.000000e+00)
  %dx0_after = load double, ptr %dx0, align 8
  call i32 (ptr, ...) @printf(ptr @.fmt, double %g1, double %dx0_after)
  ret i32 0
}

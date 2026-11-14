; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

declare void @__enzyme_fwddiff(ptr, ...)
declare i32 @printf(ptr, ...)

@fmt = private unnamed_addr constant [11 x i8] c"%.1f %.1f\0A\00", align 1

define void @cblas_daxpby(i32 %n, double %alpha, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy) noinline {
entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %next, %loop ]
  %x_off = mul i32 %idx, %incx
  %y_off = mul i32 %idx, %incy
  %x_ptr = getelementptr inbounds double, ptr %x, i32 %x_off
  %y_ptr = getelementptr inbounds double, ptr %y, i32 %y_off
  %x_val = load double, ptr %x_ptr, align 8
  %y_val = load double, ptr %y_ptr, align 8
  %ax = fmul double %alpha, %x_val
  %by = fmul double %beta, %y_val
  %sum = fadd double %ax, %by
  store double %sum, ptr %y_ptr, align 8
  %next = add i32 %idx, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @cblas_daxpy(i32 %n, double %alpha, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %next, %loop ]
  %x_off = mul i32 %idx, %incx
  %y_off = mul i32 %idx, %incy
  %x_ptr = getelementptr inbounds double, ptr %x, i32 %x_off
  %y_ptr = getelementptr inbounds double, ptr %y, i32 %y_off
  %x_val = load double, ptr %x_ptr, align 8
  %y_val = load double, ptr %y_ptr, align 8
  %scaled = fmul double %alpha, %x_val
  %sum = fadd double %y_val, %scaled
  store double %sum, ptr %y_ptr, align 8
  %next = add i32 %idx, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @cblas_dscal(i32 %n, double %alpha, ptr %x, i32 %incx) noinline {
entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %next, %loop ]
  %x_off = mul i32 %idx, %incx
  %x_ptr = getelementptr inbounds double, ptr %x, i32 %x_off
  %x_val = load double, ptr %x_ptr, align 8
  %scaled = fmul double %alpha, %x_val
  store double %scaled, ptr %x_ptr, align 8
  %next = add i32 %idx, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @test_axpby(i32 %n, double %alpha, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy) {
entry:
  call void @cblas_daxpby(i32 %n, double %alpha, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy)
  ret void
}

define i32 @main() {
entry:
  %x = alloca [2 x double], align 16
  %dx = alloca [2 x double], align 16
  %y = alloca [2 x double], align 16
  %dy = alloca [2 x double], align 16

  %x0 = getelementptr inbounds [2 x double], ptr %x, i64 0, i64 0
  %x1 = getelementptr inbounds [2 x double], ptr %x, i64 0, i64 1
  store double 2.0, ptr %x0, align 8
  store double 5.0, ptr %x1, align 8

  %dx0 = getelementptr inbounds [2 x double], ptr %dx, i64 0, i64 0
  %dx1 = getelementptr inbounds [2 x double], ptr %dx, i64 0, i64 1
  store double 1.0, ptr %dx0, align 8
  store double 3.0, ptr %dx1, align 8

  %y0 = getelementptr inbounds [2 x double], ptr %y, i64 0, i64 0
  %y1 = getelementptr inbounds [2 x double], ptr %y, i64 0, i64 1
  store double 3.0, ptr %y0, align 8
  store double 7.0, ptr %y1, align 8

  %dy0 = getelementptr inbounds [2 x double], ptr %dy, i64 0, i64 0
  %dy1 = getelementptr inbounds [2 x double], ptr %dy, i64 0, i64 1
  store double 2.0, ptr %dy0, align 8
  store double 4.0, ptr %dy1, align 8

  call void (ptr, ...) @__enzyme_fwddiff(ptr @test_axpby, i32 2, double 4.0, double 0.5, ptr %x0, ptr %dx0, i32 1, double 6.0, double 1.5, ptr %y0, ptr %dy0, i32 1)
  %vdy0 = load double, ptr %dy0, align 8
  %vdy1 = load double, ptr %dy1, align 8
  call i32 (ptr, ...) @printf(ptr @fmt, double %vdy0, double %vdy1)
  ret i32 0
}

; CHECK-LABEL: define internal void @fwddiffetest_axpby(
; CHECK: call void @cblas_dscal(i32 %n, double %beta, ptr %[[DY:.+]], i32 %incy)
; CHECK: call void @cblas_daxpy(i32 %n, double %[[DBETA:.+]], ptr %y, i32 %incy, ptr %[[DY]], i32 %incy)
; CHECK: call void @cblas_daxpy(i32 %n, double %alpha, ptr %[[DX:.+]], i32 %incx, ptr %[[DY]], i32 %incy)
; CHECK: call void @cblas_daxpy(i32 %n, double %[[DALPHA:.+]], ptr %x, i32 %incx, ptr %[[DY]], i32 %incy)
; CHECK: call void @cblas_daxpby(i32 %n, double %alpha, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy)

; EVAL: 21.5 49.0

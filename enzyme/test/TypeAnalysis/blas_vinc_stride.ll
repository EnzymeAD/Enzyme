; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -print-type-analysis -type-analysis-func=callee -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="print-type-analysis" -type-analysis-func=callee -disable-output | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

declare double @__enzyme_autodiff(ptr, ...)
declare i32 @printf(ptr, ...)

@fmt = private unnamed_addr constant [11 x i8] c"%.6f %.6f\0A\00", align 1

define void @cblas_sscal(i32 %n, float %alpha, ptr %x, i32 %incx) noinline {
  ret void
}

define void @cblas_zaxpy(i32 %n, ptr %alpha, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
  ret void
}

define double @cblas_ddot(i32 %n, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %next, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %acc_next, %loop ]
  %x_off = mul i32 %idx, %incx
  %y_off = mul i32 %idx, %incy
  %x_ptr = getelementptr inbounds double, ptr %x, i32 %x_off
  %y_ptr = getelementptr inbounds double, ptr %y, i32 %y_off
  %x_val = load double, ptr %x_ptr, align 8
  %y_val = load double, ptr %y_ptr, align 8
  %prod = fmul double %x_val, %y_val
  %acc_next = fadd double %acc, %prod
  %next = add i32 %idx, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %acc_next
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

define void @callee(ptr %x, ptr %y, ptr %neg_f, ptr %alpha_z, ptr %zx, ptr %zy) {
entry:
  %dot = call double @cblas_ddot(i32 3, ptr %x, i32 1, ptr %y, i32 2)
  call void @cblas_sscal(i32 3, float 2.000000e+00, ptr %neg_f, i32 -1)
  call void @cblas_zaxpy(i32 2, ptr %alpha_z, ptr %zx, i32 1, ptr %zy, i32 1)
  ret void
}

define double @dot_const_stride(ptr %x, ptr %y) {
entry:
  %res = call double @cblas_ddot(i32 3, ptr %x, i32 1, ptr %y, i32 2)
  ret double %res
}

define i32 @main() {
entry:
  %x = alloca [3 x double], align 16
  %dx = alloca [3 x double], align 16
  %y = alloca [5 x double], align 16
  %dy = alloca [5 x double], align 16

  %x0 = getelementptr inbounds [3 x double], ptr %x, i64 0, i64 0
  %x1 = getelementptr inbounds [3 x double], ptr %x, i64 0, i64 1
  %x2 = getelementptr inbounds [3 x double], ptr %x, i64 0, i64 2
  store double 2.0, ptr %x0, align 8
  store double 3.0, ptr %x1, align 8
  store double 5.0, ptr %x2, align 8

  %dx0 = getelementptr inbounds [3 x double], ptr %dx, i64 0, i64 0
  %dx1 = getelementptr inbounds [3 x double], ptr %dx, i64 0, i64 1
  %dx2 = getelementptr inbounds [3 x double], ptr %dx, i64 0, i64 2
  store double 0.0, ptr %dx0, align 8
  store double 0.0, ptr %dx1, align 8
  store double 0.0, ptr %dx2, align 8

  %y0 = getelementptr inbounds [5 x double], ptr %y, i64 0, i64 0
  %y1 = getelementptr inbounds [5 x double], ptr %y, i64 0, i64 1
  %y2 = getelementptr inbounds [5 x double], ptr %y, i64 0, i64 2
  %y3 = getelementptr inbounds [5 x double], ptr %y, i64 0, i64 3
  %y4 = getelementptr inbounds [5 x double], ptr %y, i64 0, i64 4
  store double 7.0, ptr %y0, align 8
  store double 100.0, ptr %y1, align 8
  store double 11.0, ptr %y2, align 8
  store double 200.0, ptr %y3, align 8
  store double 13.0, ptr %y4, align 8

  %dy0 = getelementptr inbounds [5 x double], ptr %dy, i64 0, i64 0
  %dy1 = getelementptr inbounds [5 x double], ptr %dy, i64 0, i64 1
  %dy2 = getelementptr inbounds [5 x double], ptr %dy, i64 0, i64 2
  %dy3 = getelementptr inbounds [5 x double], ptr %dy, i64 0, i64 3
  %dy4 = getelementptr inbounds [5 x double], ptr %dy, i64 0, i64 4
  store double 0.0, ptr %dy0, align 8
  store double 0.0, ptr %dy1, align 8
  store double 0.0, ptr %dy2, align 8
  store double 0.0, ptr %dy3, align 8
  store double 0.0, ptr %dy4, align 8

  call double (ptr, ...) @__enzyme_autodiff(ptr @dot_const_stride, ptr %x0, ptr %dx0, ptr %y0, ptr %dy0)

  %res_dx1 = load double, ptr %dx1, align 8
  %res_dy2 = load double, ptr %dy2, align 8
  call i32 (ptr, ...) @printf(ptr @fmt, double %res_dx1, double %res_dy2)
  ret i32 0
}

; CHECK-LABEL: callee - {}
; CHECK-NEXT: ptr %x: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double}
; CHECK-NEXT: ptr %y: {[-1]:Pointer, [-1,0]:Float@double, [-1,16]:Float@double, [-1,32]:Float@double}
; CHECK-NEXT: ptr %neg_f: {[-1]:Pointer, [-1,0]:Float@float, [-1,4]:Float@float, [-1,8]:Float@float}
; CHECK-NEXT: ptr %alpha_z: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double}
; CHECK-NEXT: ptr %zx: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Float@double}
; CHECK-NEXT: ptr %zy: {[-1]:Pointer, [-1,0]:Float@double, [-1,8]:Float@double, [-1,16]:Float@double, [-1,24]:Float@double}

; EVAL: 11.000000 3.000000

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %OPloadEnzyme -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | FileCheck %s
; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

; Verify forward-mode BLAS derivatives with active beta and non-unit incy:
; 1. axpy(dbeta, y, dy) must be emitted AFTER beta-scaling dy (gemv/symv/scal),
;    not before it (which wrongly computed beta * (dy + dbeta * y)).
; 2. Forward mode must not mark y as cached, so axpy preserves incy instead of
;    passing stride 1.

declare void @__enzyme_fwddiff(ptr, ...)
declare i32 @printf(ptr, ...)

@fmt = private unnamed_addr constant [25 x i8] c"%.1f %.1f %.1f %.1f\0A\00\00\00\00\00", align 1

define void @daxpy_(ptr %np, ptr %ap, ptr %x, ptr %incxp, ptr %y, ptr %incyp) noinline {
entry:
  %n = load i32, ptr %np, align 4
  %a = load double, ptr %ap, align 8
  %incx = load i32, ptr %incxp, align 4
  %incy = load i32, ptr %incyp, align 4
  %cmp0 = icmp sle i32 %n, 0
  br i1 %cmp0, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %xo = mul i32 %i, %incx
  %yo = mul i32 %i, %incy
  %xp = getelementptr double, ptr %x, i32 %xo
  %yp = getelementptr double, ptr %y, i32 %yo
  %xv = load double, ptr %xp, align 8
  %yv = load double, ptr %yp, align 8
  %p = fmul double %a, %xv
  %s = fadd double %yv, %p
  store double %s, ptr %yp, align 8
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @dscal_(ptr %np, ptr %ap, ptr %x, ptr %incxp) noinline {
entry:
  %n = load i32, ptr %np, align 4
  %a = load double, ptr %ap, align 8
  %incx = load i32, ptr %incxp, align 4
  %cmp0 = icmp sle i32 %n, 0
  br i1 %cmp0, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %xo = mul i32 %i, %incx
  %xp = getelementptr double, ptr %x, i32 %xo
  %xv = load double, ptr %xp, align 8
  %s = fmul double %a, %xv
  store double %s, ptr %xp, align 8
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; Minimal column-major N-only dgemv_ (m x n, lda)
define void @dgemv_(ptr %tap, ptr %mp, ptr %np, ptr %ap, ptr %A, ptr %ldap, ptr %x, ptr %incxp, ptr %bp, ptr %y, ptr %incyp, i64 %l1) noinline {
entry:
  %m = load i32, ptr %mp, align 4
  %n = load i32, ptr %np, align 4
  %a = load double, ptr %ap, align 8
  %lda = load i32, ptr %ldap, align 4
  %incx = load i32, ptr %incxp, align 4
  %b = load double, ptr %bp, align 8
  %incy = load i32, ptr %incyp, align 4
  br label %outer
outer:
  %i = phi i32 [ 0, %entry ], [ %inext, %outer_end ]
  br label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %jnext, %inner ]
  %acc = phi double [ 0.0, %outer ], [ %accn, %inner ]
  %jlda = mul i32 %j, %lda
  %aoff = add i32 %i, %jlda
  %aptr = getelementptr double, ptr %A, i32 %aoff
  %av = load double, ptr %aptr, align 8
  %xoff = mul i32 %j, %incx
  %xptr = getelementptr double, ptr %x, i32 %xoff
  %xv = load double, ptr %xptr, align 8
  %prod = fmul double %av, %xv
  %accn = fadd double %acc, %prod
  %jnext = add i32 %j, 1
  %jdone = icmp eq i32 %jnext, %n
  br i1 %jdone, label %outer_end, label %inner
outer_end:
  %yoff = mul i32 %i, %incy
  %yptr = getelementptr double, ptr %y, i32 %yoff
  %yv = load double, ptr %yptr, align 8
  %t1 = fmul double %a, %accn
  %t2 = fmul double %b, %yv
  %res = fadd double %t1, %t2
  store double %res, ptr %yptr, align 8
  %inext = add i32 %i, 1
  %idone = icmp eq i32 %inext, %m
  br i1 %idone, label %exit, label %outer
exit:
  ret void
}

; Minimal symmetric 2x2 upper column-major dsymv_
define void @dsymv_(ptr %uplop, ptr %np, ptr %ap, ptr %A, ptr %ldap, ptr %x, ptr %incxp, ptr %bp, ptr %y, ptr %incyp, i64 %l1) noinline {
entry:
  %n = load i32, ptr %np, align 4
  %a = load double, ptr %ap, align 8
  %lda = load i32, ptr %ldap, align 4
  %incx = load i32, ptr %incxp, align 4
  %b = load double, ptr %bp, align 8
  %incy = load i32, ptr %incyp, align 4
  br label %outer
outer:
  %i = phi i32 [ 0, %entry ], [ %inext, %outer_end ]
  br label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %jnext, %inner ]
  %acc = phi double [ 0.0, %outer ], [ %accn, %inner ]
  %le = icmp sle i32 %i, %j
  %r = select i1 %le, i32 %i, i32 %j
  %c = select i1 %le, i32 %j, i32 %i
  %clda = mul i32 %c, %lda
  %aoff = add i32 %r, %clda
  %aptr = getelementptr double, ptr %A, i32 %aoff
  %av = load double, ptr %aptr, align 8
  %xoff = mul i32 %j, %incx
  %xptr = getelementptr double, ptr %x, i32 %xoff
  %xv = load double, ptr %xptr, align 8
  %prod = fmul double %av, %xv
  %accn = fadd double %acc, %prod
  %jnext = add i32 %j, 1
  %jdone = icmp eq i32 %jnext, %n
  br i1 %jdone, label %outer_end, label %inner
outer_end:
  %yoff = mul i32 %i, %incy
  %yptr = getelementptr double, ptr %y, i32 %yoff
  %yv = load double, ptr %yptr, align 8
  %t1 = fmul double %a, %accn
  %t2 = fmul double %b, %yv
  %res = fadd double %t1, %t2
  store double %res, ptr %yptr, align 8
  %inext = add i32 %i, 1
  %idone = icmp eq i32 %inext, %n
  br i1 %idone, label %exit, label %outer
exit:
  ret void
}

define void @cblas_daxpy(i32 %n, double %a, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  %cmp0 = icmp sle i32 %n, 0
  br i1 %cmp0, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %xo = mul i32 %i, %incx
  %yo = mul i32 %i, %incy
  %xp = getelementptr double, ptr %x, i32 %xo
  %yp = getelementptr double, ptr %y, i32 %yo
  %xv = load double, ptr %xp, align 8
  %yv = load double, ptr %yp, align 8
  %p = fmul double %a, %xv
  %s = fadd double %yv, %p
  store double %s, ptr %yp, align 8
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @cblas_dscal(i32 %n, double %a, ptr %x, i32 %incx) noinline {
entry:
  %cmp0 = icmp sle i32 %n, 0
  br i1 %cmp0, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %xo = mul i32 %i, %incx
  %xp = getelementptr double, ptr %x, i32 %xo
  %xv = load double, ptr %xp, align 8
  %s = fmul double %a, %xv
  store double %s, ptr %xp, align 8
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @cblas_daxpby(i32 %n, double %a, ptr %x, i32 %incx, double %b, ptr %y, i32 %incy) noinline {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %xo = mul i32 %i, %incx
  %yo = mul i32 %i, %incy
  %xp = getelementptr double, ptr %x, i32 %xo
  %yp = getelementptr double, ptr %y, i32 %yo
  %xv = load double, ptr %xp, align 8
  %yv = load double, ptr %yp, align 8
  %t1 = fmul double %a, %xv
  %t2 = fmul double %b, %yv
  %s = fadd double %t1, %t2
  store double %s, ptr %yp, align 8
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @cblas_dgemv(i32 %layout, i32 %ta, i32 %m, i32 %n, double %a, ptr %A, i32 %lda, ptr %x, i32 %incx, double %b, ptr %y, i32 %incy) noinline {
entry:
  %ta_c = alloca i8, align 1
  %mp = alloca i32, align 4
  %np = alloca i32, align 4
  %ap = alloca double, align 8
  %ldap = alloca i32, align 4
  %incxp = alloca i32, align 4
  %bp = alloca double, align 8
  %incyp = alloca i32, align 4
  store i8 78, ptr %ta_c, align 1
  store i32 %m, ptr %mp, align 4
  store i32 %n, ptr %np, align 4
  store double %a, ptr %ap, align 8
  store i32 %lda, ptr %ldap, align 4
  store i32 %incx, ptr %incxp, align 4
  store double %b, ptr %bp, align 8
  store i32 %incy, ptr %incyp, align 4
  call void @dgemv_(ptr %ta_c, ptr %mp, ptr %np, ptr %ap, ptr %A, ptr %ldap, ptr %x, ptr %incxp, ptr %bp, ptr %y, ptr %incyp, i64 1)
  ret void
}

define void @f_gemv(ptr %ta, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %x, ptr %incx, ptr %beta, ptr %y, ptr %incy) {
  call void @dgemv_(ptr %ta, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %x, ptr %incx, ptr %beta, ptr %y, ptr %incy, i64 1)
  ret void
}

define void @f_symv(ptr %uplo, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %x, ptr %incx, ptr %beta, ptr %y, ptr %incy) {
  call void @dsymv_(ptr %uplo, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %x, ptr %incx, ptr %beta, ptr %y, ptr %incy, i64 1)
  ret void
}

define void @f_cgemv(i32 %l, i32 %ta, i32 %m, i32 %n, double %alpha, ptr %A, i32 %lda, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy) {
  call void @cblas_dgemv(i32 %l, i32 %ta, i32 %m, i32 %n, double %alpha, ptr %A, i32 %lda, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy)
  ret void
}

define void @f_caxpby(i32 %n, double %alpha, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy) {
  call void @cblas_daxpby(i32 %n, double %alpha, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy)
  ret void
}

define i32 @main() {
entry:
  %ta = alloca i8, align 1
  %uplo = alloca i8, align 1
  %n = alloca i32, align 4
  %inc2 = alloca i32, align 4
  %alpha = alloca double, align 8
  %beta = alloca double, align 8
  %dbeta = alloca double, align 8
  %A = alloca [4 x double], align 16
  %x = alloca [4 x double], align 16
  %dx = alloca [4 x double], align 16
  %y = alloca [4 x double], align 16
  %dy = alloca [4 x double], align 16

  store i8 78, ptr %ta, align 1
  store i8 85, ptr %uplo, align 1
  store i32 2, ptr %n, align 4
  store i32 2, ptr %inc2, align 4
  store double 1.0, ptr %alpha, align 8
  store double 3.0, ptr %beta, align 8
  store double 2.0, ptr %dbeta, align 8

  ; A = [[2, 3], [3, 5]] (symmetric, col-major: 2, 3, 3, 5)
  store [4 x double] [double 2.0, double 3.0, double 3.0, double 5.0], ptr %A, align 16
  ; x (stride 2): [1.0, -88.0, 2.0, -88.0], dx: [0.5, -88.0, 1.0, -88.0]
  store [4 x double] [double 1.0, double -88.0, double 2.0, double -88.0], ptr %x, align 16
  store [4 x double] [double 0.5, double -88.0, double 1.0, double -88.0], ptr %dx, align 16

  ; Test 1: dgemv_ (alpha=1, beta=3, dbeta=2, y=[10,-99,20,-99], dy=[1,-99,4,-99])
  ; A*dx = [2*0.5+3*1, 3*0.5+5*1] = [4.0, 6.5]
  ; dy_new = 4.0 + 3*1 + 2*10 = 27.0, 6.5 + 3*4 + 2*20 = 58.5; sentinels stay -99.0
  store [4 x double] [double 10.0, double -99.0, double 20.0, double -99.0], ptr %y, align 16
  store [4 x double] [double 1.0, double -99.0, double 4.0, double -99.0], ptr %dy, align 16
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_gemv,
    metadata !"enzyme_const", ptr %ta, metadata !"enzyme_const", ptr %n, metadata !"enzyme_const", ptr %n,
    metadata !"enzyme_const", ptr %alpha, metadata !"enzyme_const", ptr %A, metadata !"enzyme_const", ptr %n,
    metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", ptr %inc2,
    metadata !"enzyme_dup", ptr %beta, ptr %dbeta,
    metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_const", ptr %inc2)
  %g0 = load double, ptr %dy, align 8
  %gp1 = getelementptr [4 x double], ptr %dy, i64 0, i64 1
  %g1 = load double, ptr %gp1, align 8
  %gp2 = getelementptr [4 x double], ptr %dy, i64 0, i64 2
  %g2 = load double, ptr %gp2, align 8
  %gp3 = getelementptr [4 x double], ptr %dy, i64 0, i64 3
  %g3 = load double, ptr %gp3, align 8
  call i32 (ptr, ...) @printf(ptr @fmt, double %g0, double %g1, double %g2, double %g3)

  ; Test 2: dsymv_ (same inputs -> same exact expected result: 27.0 -99.0 58.5 -99.0)
  store [4 x double] [double 10.0, double -99.0, double 20.0, double -99.0], ptr %y, align 16
  store [4 x double] [double 1.0, double -99.0, double 4.0, double -99.0], ptr %dy, align 16
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_symv,
    metadata !"enzyme_const", ptr %uplo, metadata !"enzyme_const", ptr %n,
    metadata !"enzyme_const", ptr %alpha, metadata !"enzyme_const", ptr %A, metadata !"enzyme_const", ptr %n,
    metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", ptr %inc2,
    metadata !"enzyme_dup", ptr %beta, ptr %dbeta,
    metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_const", ptr %inc2)
  %s0 = load double, ptr %dy, align 8
  %s1 = load double, ptr %gp1, align 8
  %s2 = load double, ptr %gp2, align 8
  %s3 = load double, ptr %gp3, align 8
  call i32 (ptr, ...) @printf(ptr @fmt, double %s0, double %s1, double %s2, double %s3)

  ; Test 3: cblas_dgemv (same inputs -> 27.0 -99.0 58.5 -99.0)
  store [4 x double] [double 10.0, double -99.0, double 20.0, double -99.0], ptr %y, align 16
  store [4 x double] [double 1.0, double -99.0, double 4.0, double -99.0], ptr %dy, align 16
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_cgemv,
    metadata !"enzyme_const", i32 102, metadata !"enzyme_const", i32 111,
    metadata !"enzyme_const", i32 2, metadata !"enzyme_const", i32 2,
    metadata !"enzyme_const", double 1.0, metadata !"enzyme_const", ptr %A, metadata !"enzyme_const", i32 2,
    metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", i32 2,
    metadata !"enzyme_dup", double 3.0, double 2.0,
    metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_const", i32 2)
  %cg0 = load double, ptr %dy, align 8
  %cg1 = load double, ptr %gp1, align 8
  %cg2 = load double, ptr %gp2, align 8
  %cg3 = load double, ptr %gp3, align 8
  call i32 (ptr, ...) @printf(ptr @fmt, double %cg0, double %cg1, double %cg2, double %cg3)

  ; Test 4: cblas_daxpby (n=2, alpha=2, dalpha=0, x=[1,-88,2,-88], dx=[0.5,-88,1,-88],
  ;                       beta=3, dbeta=2, y=[10,-99,20,-99], dy=[1,-99,4,-99], incx=2, incy=2)
  ; dy_new = 2*[0.5, 1.0] + 3*[1, 4] + 2*[10, 20] = [24.0, 54.0]; sentinels stay -99.0
  store [4 x double] [double 10.0, double -99.0, double 20.0, double -99.0], ptr %y, align 16
  store [4 x double] [double 1.0, double -99.0, double 4.0, double -99.0], ptr %dy, align 16
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_caxpby,
    metadata !"enzyme_const", i32 2, metadata !"enzyme_const", double 2.0,
    metadata !"enzyme_dup", ptr %x, ptr %dx, metadata !"enzyme_const", i32 2,
    metadata !"enzyme_dup", double 3.0, double 2.0,
    metadata !"enzyme_dup", ptr %y, ptr %dy, metadata !"enzyme_const", i32 2)
  %a0 = load double, ptr %dy, align 8
  %a1 = load double, ptr %gp1, align 8
  %a2 = load double, ptr %gp2, align 8
  %a3 = load double, ptr %gp3, align 8
  call i32 (ptr, ...) @printf(ptr @fmt, double %a0, double %a1, double %a2, double %a3)

  ret i32 0
}

; CHECK-LABEL: define internal void @fwddiffef_gemv(
; CHECK: call void @dgemv_(ptr %ta, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %"x'", ptr %incx, ptr %beta, ptr %"y'", ptr %incy, i64 1)
; CHECK: call void @daxpy_(ptr %{{.+}}, ptr %"beta'", ptr %y, ptr %incy, ptr %"y'", ptr %incy)
; CHECK: call void @dgemv_(ptr %ta, ptr %m, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %x, ptr %incx, ptr %beta, ptr %y, ptr %incy, i64 1)

; CHECK-LABEL: define internal void @fwddiffef_symv(
; CHECK: call void @dsymv_(ptr %uplo, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %"x'", ptr %incx, ptr %beta, ptr %"y'", ptr %incy, i64 1)
; CHECK: call void @daxpy_(ptr %n, ptr %"beta'", ptr %y, ptr %incy, ptr %"y'", ptr %incy)
; CHECK: call void @dsymv_(ptr %uplo, ptr %n, ptr %alpha, ptr %A, ptr %lda, ptr %x, ptr %incx, ptr %beta, ptr %y, ptr %incy, i64 1)

; CHECK-LABEL: define internal void @fwddiffef_cgemv(
; CHECK: call void @cblas_dgemv(i32 %l, i32 %ta, i32 %m, i32 %n, double %alpha, ptr %A, i32 %lda, ptr %"x'", i32 %incx, double %beta, ptr %"y'", i32 %incy)
; CHECK: call void @cblas_daxpy(i32 %{{.+}}, double %"beta'", ptr %y, i32 %incy, ptr %"y'", i32 %incy)
; CHECK: call void @cblas_dgemv(i32 %l, i32 %ta, i32 %m, i32 %n, double %alpha, ptr %A, i32 %lda, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy)

; CHECK-LABEL: define internal void @fwddiffef_caxpby(
; CHECK: call void @cblas_dscal(i32 %n, double %beta, ptr %"y'", i32 %incy)
; CHECK: call void @cblas_daxpy(i32 %n, double %"beta'", ptr %y, i32 %incy, ptr %"y'", i32 %incy)
; CHECK: call void @cblas_daxpy(i32 %n, double %alpha, ptr %"x'", i32 %incx, ptr %"y'", i32 %incy)
; CHECK: call void @cblas_daxpby(i32 %n, double %alpha, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy)

; EVAL: 27.0 -99.0 58.5 -99.0
; EVAL-NEXT: 27.0 -99.0 58.5 -99.0
; EVAL-NEXT: 27.0 -99.0 58.5 -99.0
; EVAL-NEXT: 24.0 -99.0 54.0 -99.0

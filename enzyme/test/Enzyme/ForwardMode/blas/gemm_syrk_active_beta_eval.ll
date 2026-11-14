; RUN: %opt < %s %OPnewLoadEnzyme -passes="enzyme" -S | %lli - | FileCheck %s --check-prefix=EVAL

; Numerical evaluation of forward-mode BLAS matrix derivatives with active beta
; and C: padded ldc, ColMajor and RowMajor layouts, beta-only active (lascl +
; AxpyMat), and triangular Upper/Lower syrk leaving the opposite triangle and
; padding untouched.

declare void @__enzyme_fwddiff(ptr, ...)
declare i32 @printf(ptr, ...)

@fmt6 = private unnamed_addr constant [31 x i8] c"%.1f %.1f %.1f %.1f %.1f %.1f\0A\00", align 1

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

define void @dgemm_(ptr %tap, ptr %tbp, ptr %mp, ptr %np, ptr %kp, ptr %ap, ptr %A, ptr %ldap, ptr %B, ptr %ldbp, ptr %bp, ptr %C, ptr %ldcp, i64 %l1, i64 %l2) noinline {
entry:
  %m = load i32, ptr %mp, align 4
  %n = load i32, ptr %np, align 4
  %k = load i32, ptr %kp, align 4
  %alpha = load double, ptr %ap, align 8
  %lda = load i32, ptr %ldap, align 4
  %ldb = load i32, ptr %ldbp, align 4
  %beta = load double, ptr %bp, align 8
  %ldc = load i32, ptr %ldcp, align 4
  br label %col
col:
  %j = phi i32 [ 0, %entry ], [ %jnext, %col_end ]
  br label %row
row:
  %i = phi i32 [ 0, %col ], [ %inext, %row_end ]
  br label %kloop
kloop:
  %p = phi i32 [ 0, %row ], [ %pnext, %kloop ]
  %acc = phi double [ 0.0, %row ], [ %accn, %kloop ]
  %plda = mul i32 %p, %lda
  %aoff = add i32 %i, %plda
  %aptr = getelementptr double, ptr %A, i32 %aoff
  %av = load double, ptr %aptr, align 8
  %jldb = mul i32 %j, %ldb
  %boff = add i32 %p, %jldb
  %bptr = getelementptr double, ptr %B, i32 %boff
  %bv = load double, ptr %bptr, align 8
  %prod = fmul double %av, %bv
  %accn = fadd double %acc, %prod
  %pnext = add i32 %p, 1
  %pdone = icmp eq i32 %pnext, %k
  br i1 %pdone, label %row_end, label %kloop
row_end:
  %jldc = mul i32 %j, %ldc
  %coff = add i32 %i, %jldc
  %cptr = getelementptr double, ptr %C, i32 %coff
  %cv = load double, ptr %cptr, align 8
  %t1 = fmul double %alpha, %accn
  %t2 = fmul double %beta, %cv
  %res = fadd double %t1, %t2
  store double %res, ptr %cptr, align 8
  %inext = add i32 %i, 1
  %rdone = icmp eq i32 %inext, %m
  br i1 %rdone, label %col_end, label %row
col_end:
  %jnext = add i32 %j, 1
  %cdone = icmp eq i32 %jnext, %n
  br i1 %cdone, label %exit, label %col
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

define void @cblas_dgemm(i32 %layout, i32 %ta, i32 %tb, i32 %m, i32 %n, i32 %k, double %alpha, ptr %A, i32 %lda, ptr %B, i32 %ldb, double %beta, ptr %C, i32 %ldc) noinline {
entry:
  %isrm = icmp eq i32 %layout, 101
  br label %iloop
iloop:
  %i = phi i32 [ 0, %entry ], [ %inext, %iend ]
  br label %jloop
jloop:
  %j = phi i32 [ 0, %iloop ], [ %jnext, %jend ]
  br label %ploop
ploop:
  %p = phi i32 [ 0, %jloop ], [ %pnext, %ploop ]
  %acc = phi double [ 0.0, %jloop ], [ %accn, %ploop ]
  %rm_a1 = mul i32 %i, %lda
  %rm_a = add i32 %rm_a1, %p
  %cm_a1 = mul i32 %p, %lda
  %cm_a = add i32 %i, %cm_a1
  %aoff = select i1 %isrm, i32 %rm_a, i32 %cm_a
  %rm_b1 = mul i32 %p, %ldb
  %rm_b = add i32 %rm_b1, %j
  %cm_b1 = mul i32 %j, %ldb
  %cm_b = add i32 %p, %cm_b1
  %boff = select i1 %isrm, i32 %rm_b, i32 %cm_b
  %ap = getelementptr double, ptr %A, i32 %aoff
  %bp = getelementptr double, ptr %B, i32 %boff
  %av = load double, ptr %ap, align 8
  %bv = load double, ptr %bp, align 8
  %pr = fmul double %av, %bv
  %accn = fadd double %acc, %pr
  %pnext = add i32 %p, 1
  %pdone = icmp eq i32 %pnext, %k
  br i1 %pdone, label %jend, label %ploop
jend:
  %rm_c1 = mul i32 %i, %ldc
  %rm_c = add i32 %rm_c1, %j
  %cm_c1 = mul i32 %j, %ldc
  %cm_c = add i32 %i, %cm_c1
  %coff = select i1 %isrm, i32 %rm_c, i32 %cm_c
  %cp = getelementptr double, ptr %C, i32 %coff
  %cv = load double, ptr %cp, align 8
  %t1 = fmul double %alpha, %accn
  %t2 = fmul double %beta, %cv
  %res = fadd double %t1, %t2
  store double %res, ptr %cp, align 8
  %jnext = add i32 %j, 1
  %jdone = icmp eq i32 %jnext, %n
  br i1 %jdone, label %iend, label %jloop
iend:
  %inext = add i32 %i, 1
  %idone = icmp eq i32 %inext, %m
  br i1 %idone, label %exit, label %iloop
exit:
  ret void
}

define void @cblas_dsyrk(i32 %layout, i32 %uplo, i32 %trans, i32 %n, i32 %k, double %alpha, ptr %A, i32 %lda, double %beta, ptr %C, i32 %ldc) noinline {
entry:
  %isrm = icmp eq i32 %layout, 101
  %isup = icmp eq i32 %uplo, 121
  br label %iloop
iloop:
  %i = phi i32 [ 0, %entry ], [ %inext, %iend ]
  br label %jloop
jloop:
  %j = phi i32 [ 0, %iloop ], [ %jnext, %jcont ]
  %skip_up = icmp sgt i32 %i, %j
  %skip_lo = icmp slt i32 %i, %j
  %skip = select i1 %isup, i1 %skip_up, i1 %skip_lo
  br i1 %skip, label %jcont, label %ploop
ploop:
  %p = phi i32 [ 0, %jloop ], [ %pnext, %ploop ]
  %acc = phi double [ 0.0, %jloop ], [ %accn, %ploop ]
  %rm_i1 = mul i32 %i, %lda
  %rm_i = add i32 %rm_i1, %p
  %cm_i1 = mul i32 %p, %lda
  %cm_i = add i32 %i, %cm_i1
  %ioff = select i1 %isrm, i32 %rm_i, i32 %cm_i
  %rm_j1 = mul i32 %j, %lda
  %rm_j = add i32 %rm_j1, %p
  %cm_j1 = mul i32 %p, %lda
  %cm_j = add i32 %j, %cm_j1
  %joff = select i1 %isrm, i32 %rm_j, i32 %cm_j
  %aip = getelementptr double, ptr %A, i32 %ioff
  %ajp = getelementptr double, ptr %A, i32 %joff
  %aiv = load double, ptr %aip, align 8
  %ajv = load double, ptr %ajp, align 8
  %pr = fmul double %aiv, %ajv
  %accn = fadd double %acc, %pr
  %pnext = add i32 %p, 1
  %pdone = icmp eq i32 %pnext, %k
  br i1 %pdone, label %store_c, label %ploop
store_c:
  %rm_c1 = mul i32 %i, %ldc
  %rm_c = add i32 %rm_c1, %j
  %cm_c1 = mul i32 %j, %ldc
  %cm_c = add i32 %i, %cm_c1
  %coff = select i1 %isrm, i32 %rm_c, i32 %cm_c
  %cp = getelementptr double, ptr %C, i32 %coff
  %cv = load double, ptr %cp, align 8
  %t1 = fmul double %alpha, %accn
  %t2 = fmul double %beta, %cv
  %res = fadd double %t1, %t2
  store double %res, ptr %cp, align 8
  br label %jcont
jcont:
  %jnext = add i32 %j, 1
  %jdone = icmp eq i32 %jnext, %n
  br i1 %jdone, label %iend, label %jloop
iend:
  %inext = add i32 %i, 1
  %idone = icmp eq i32 %inext, %n
  br i1 %idone, label %exit, label %iloop
exit:
  ret void
}

define void @f_dgemm(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc) {
  call void @dgemm_(ptr %ta, ptr %tb, ptr %m, ptr %n, ptr %k, ptr %alpha, ptr %A, ptr %lda, ptr %B, ptr %ldb, ptr %beta, ptr %C, ptr %ldc, i64 1, i64 1)
  ret void
}

define void @f_cgemm(i32 %l, i32 %ta, i32 %tb, i32 %m, i32 %n, i32 %k, double %alpha, ptr %A, i32 %lda, ptr %B, i32 %ldb, double %beta, ptr %C, i32 %ldc) {
  call void @cblas_dgemm(i32 %l, i32 %ta, i32 %tb, i32 %m, i32 %n, i32 %k, double %alpha, ptr %A, i32 %lda, ptr %B, i32 %ldb, double %beta, ptr %C, i32 %ldc)
  ret void
}

define void @f_csyrk(i32 %l, i32 %uplo, i32 %trans, i32 %n, i32 %k, double %alpha, ptr %A, i32 %lda, double %beta, ptr %C, i32 %ldc) {
  call void @cblas_dsyrk(i32 %l, i32 %uplo, i32 %trans, i32 %n, i32 %k, double %alpha, ptr %A, i32 %lda, double %beta, ptr %C, i32 %ldc)
  ret void
}

define i32 @main() {
entry:
  %ta = alloca i8, align 1
  %m = alloca i32, align 4
  %n = alloca i32, align 4
  %k = alloca i32, align 4
  %ldc3 = alloca i32, align 4
  %alpha = alloca double, align 8
  %beta = alloca double, align 8
  %dbeta = alloca double, align 8
  %A = alloca [2 x double], align 16
  %dA = alloca [2 x double], align 16
  %B = alloca [2 x double], align 16
  %dB = alloca [2 x double], align 16
  %C = alloca [6 x double], align 16
  %dC = alloca [6 x double], align 16

  store i8 78, ptr %ta, align 1
  store i32 2, ptr %m, align 4
  store i32 2, ptr %n, align 4
  store i32 1, ptr %k, align 4
  store i32 3, ptr %ldc3, align 4
  store double 1.0, ptr %alpha, align 8
  store double 2.0, ptr %beta, align 8
  store double 3.0, ptr %dbeta, align 8
  store [2 x double] [double 1.0, double 2.0], ptr %A, align 16
  store [2 x double] [double 0.5, double 1.0], ptr %dA, align 16
  store [2 x double] [double 2.0, double 4.0], ptr %B, align 16
  store [2 x double] [double 0.0, double 0.0], ptr %dB, align 16

  %p1 = getelementptr [6 x double], ptr %dC, i64 0, i64 1
  %p2 = getelementptr [6 x double], ptr %dC, i64 0, i64 2
  %p3 = getelementptr [6 x double], ptr %dC, i64 0, i64 3
  %p4 = getelementptr [6 x double], ptr %dC, i64 0, i64 4
  %p5 = getelementptr [6 x double], ptr %dC, i64 0, i64 5

  ; Test 1: dgemm_ padded ldc=3
  store [6 x double] [double 1.0, double 2.0, double -99.0, double 3.0, double 4.0, double -99.0], ptr %C, align 16
  store [6 x double] [double 10.0, double 20.0, double -99.0, double 30.0, double 40.0, double -99.0], ptr %dC, align 16
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_dgemm,
    metadata !"enzyme_const", ptr %ta, metadata !"enzyme_const", ptr %ta,
    metadata !"enzyme_const", ptr %m, metadata !"enzyme_const", ptr %n, metadata !"enzyme_const", ptr %k,
    metadata !"enzyme_const", ptr %alpha,
    metadata !"enzyme_dup", ptr %A, ptr %dA, metadata !"enzyme_const", ptr %m,
    metadata !"enzyme_dup", ptr %B, ptr %dB, metadata !"enzyme_const", ptr %k,
    metadata !"enzyme_dup", ptr %beta, ptr %dbeta,
    metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_const", ptr %ldc3)
  %v10 = load double, ptr %dC, align 8
  %v11 = load double, ptr %p1, align 8
  %v12 = load double, ptr %p2, align 8
  %v13 = load double, ptr %p3, align 8
  %v14 = load double, ptr %p4, align 8
  %v15 = load double, ptr %p5, align 8
  call i32 (ptr, ...) @printf(ptr @fmt6, double %v10, double %v11, double %v12, double %v13, double %v14, double %v15)

  ; Test 2: cblas_dgemm RowMajor (101) padded ldc=3, beta-only active (lascl + AxpyMat)
  store [6 x double] [double 1.0, double 2.0, double -99.0, double 3.0, double 4.0, double -99.0], ptr %C, align 16
  store [6 x double] [double 10.0, double 20.0, double -99.0, double 30.0, double 40.0, double -99.0], ptr %dC, align 16
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_cgemm,
    metadata !"enzyme_const", i32 101, metadata !"enzyme_const", i32 111, metadata !"enzyme_const", i32 111,
    metadata !"enzyme_const", i32 2, metadata !"enzyme_const", i32 2, metadata !"enzyme_const", i32 1,
    metadata !"enzyme_const", double 1.0,
    metadata !"enzyme_const", ptr %A, metadata !"enzyme_const", i32 1,
    metadata !"enzyme_const", ptr %B, metadata !"enzyme_const", i32 2,
    metadata !"enzyme_dup", double 2.0, double 3.0,
    metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_const", i32 3)
  %v20 = load double, ptr %dC, align 8
  %v21 = load double, ptr %p1, align 8
  %v22 = load double, ptr %p2, align 8
  %v23 = load double, ptr %p3, align 8
  %v24 = load double, ptr %p4, align 8
  %v25 = load double, ptr %p5, align 8
  call i32 (ptr, ...) @printf(ptr @fmt6, double %v20, double %v21, double %v22, double %v23, double %v24, double %v25)

  ; Test 3: cblas_dsyrk RowMajor (101) Upper (121) padded ldc=3, beta-only active
  store [6 x double] [double 1.0, double 2.0, double -99.0, double -99.0, double 4.0, double -99.0], ptr %C, align 16
  store [6 x double] [double 10.0, double 20.0, double -99.0, double -99.0, double 40.0, double -99.0], ptr %dC, align 16
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_csyrk,
    metadata !"enzyme_const", i32 101, metadata !"enzyme_const", i32 121, metadata !"enzyme_const", i32 111,
    metadata !"enzyme_const", i32 2, metadata !"enzyme_const", i32 1,
    metadata !"enzyme_const", double 1.0,
    metadata !"enzyme_const", ptr %A, metadata !"enzyme_const", i32 1,
    metadata !"enzyme_dup", double 2.0, double 3.0,
    metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_const", i32 3)
  %v30 = load double, ptr %dC, align 8
  %v31 = load double, ptr %p1, align 8
  %v32 = load double, ptr %p2, align 8
  %v33 = load double, ptr %p3, align 8
  %v34 = load double, ptr %p4, align 8
  %v35 = load double, ptr %p5, align 8
  call i32 (ptr, ...) @printf(ptr @fmt6, double %v30, double %v31, double %v32, double %v33, double %v34, double %v35)

  ; Test 4: cblas_dsyrk ColMajor (102) Lower (122) padded ldc=3, beta-only active
  store [6 x double] [double 1.0, double 2.0, double -99.0, double -99.0, double 4.0, double -99.0], ptr %C, align 16
  store [6 x double] [double 10.0, double 20.0, double -99.0, double -99.0, double 40.0, double -99.0], ptr %dC, align 16
  call void (ptr, ...) @__enzyme_fwddiff(ptr @f_csyrk,
    metadata !"enzyme_const", i32 102, metadata !"enzyme_const", i32 122, metadata !"enzyme_const", i32 111,
    metadata !"enzyme_const", i32 2, metadata !"enzyme_const", i32 1,
    metadata !"enzyme_const", double 1.0,
    metadata !"enzyme_const", ptr %A, metadata !"enzyme_const", i32 2,
    metadata !"enzyme_dup", double 2.0, double 3.0,
    metadata !"enzyme_dup", ptr %C, ptr %dC, metadata !"enzyme_const", i32 3)
  %v40 = load double, ptr %dC, align 8
  %v41 = load double, ptr %p1, align 8
  %v42 = load double, ptr %p2, align 8
  %v43 = load double, ptr %p3, align 8
  %v44 = load double, ptr %p4, align 8
  %v45 = load double, ptr %p5, align 8
  call i32 (ptr, ...) @printf(ptr @fmt6, double %v40, double %v41, double %v42, double %v43, double %v44, double %v45)

  ret i32 0
}

; EVAL: 24.0 48.0 -99.0 71.0 96.0 -99.0
; EVAL-NEXT: 23.0 46.0 -99.0 69.0 92.0 -99.0
; EVAL-NEXT: 23.0 46.0 -99.0 -99.0 92.0 -99.0
; EVAL-NEXT: 23.0 46.0 -99.0 -99.0 92.0 -99.0

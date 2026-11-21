; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -passes="enzyme,verify" -S | %lli - | FileCheck %s --check-prefix=EVAL; fi

declare void @__enzyme_autodiff(ptr, ...)
declare i32 @printf(ptr, ...)

@fmt_f = private unnamed_addr constant [52 x i8] c"F %.1f %.1f | %.1f %.1f %.1f | %.1f %.1f %.1f %.1f\0A\00", align 1
@fmt_c = private unnamed_addr constant [41 x i8] c"C %d %d | %.1f %.1f %.1f %.1f %.1f %.1f\0A\00", align 1
@fmt_l = private unnamed_addr constant [35 x i8] c"L | %.1f %.1f %.1f %.1f %.1f %.1f\0A\00", align 1

; Minimal reference BLAS implementations (noinline) for E2E verification.
define internal void @spmv_impl(i1 %row_major, i1 %is_lower, i32 %n, double %alpha, ptr %ap, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy) noinline {
entry:
  %le0 = icmp sle i32 %n, 0
  br i1 %le0, label %exit, label %init

init:
  %n_m1 = sub i32 1, %n
  %x_neg = icmp slt i32 %incx, 0
  %kx_neg = mul i32 %n_m1, %incx
  %kx = select i1 %x_neg, i32 %kx_neg, i32 0
  %y_neg = icmp slt i32 %incy, 0
  %ky_neg = mul i32 %n_m1, %incy
  %ky = select i1 %y_neg, i32 %ky_neg, i32 0
  %col_lower = xor i1 %is_lower, %row_major
  %beta_zero = fcmp oeq double %beta, 0.0
  br label %outer

outer:
  %i = phi i32 [ 0, %init ], [ %i_next, %outer_latch ]
  %yi_off = mul i32 %i, %incy
  %yi_idx = add i32 %ky, %yi_off
  %y_ptr = getelementptr inbounds double, ptr %y, i32 %yi_idx
  %y_old = load double, ptr %y_ptr, align 8
  %y_scaled = fmul double %beta, %y_old
  %y_init = select i1 %beta_zero, double 0.0, double %y_scaled
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j_next, %inner ]
  %acc = phi double [ 0.0, %outer ], [ %acc_next, %inner ]
  %ilej = icmp sle i32 %i, %j
  %r = select i1 %ilej, i32 %i, i32 %j
  %c = select i1 %ilej, i32 %j, i32 %i
  ; col-major upper: r + c*(c+1)/2
  %cp1 = add i32 %c, 1
  %c_prod = mul i32 %c, %cp1
  %c_tri = sdiv i32 %c_prod, 2
  %u_idx = add i32 %r, %c_tri
  ; col-major lower: (c - r) + r*(2*n - r + 1)/2
  %twon = mul i32 2, %n
  %twon_mr = sub i32 %twon, %r
  %twon_p1 = add i32 %twon_mr, 1
  %r_prod = mul i32 %r, %twon_p1
  %r_tri = sdiv i32 %r_prod, 2
  %cmr = sub i32 %c, %r
  %l_idx = add i32 %cmr, %r_tri
  %ap_idx = select i1 %col_lower, i32 %l_idx, i32 %u_idx
  %ap_ptr = getelementptr inbounds double, ptr %ap, i32 %ap_idx
  %aij = load double, ptr %ap_ptr, align 8
  %xj_off = mul i32 %j, %incx
  %xj_idx = add i32 %kx, %xj_off
  %x_ptr = getelementptr inbounds double, ptr %x, i32 %xj_idx
  %xj = load double, ptr %x_ptr, align 8
  %prod = fmul double %aij, %xj
  %acc_next = fadd double %acc, %prod
  %j_next = add i32 %j, 1
  %j_done = icmp eq i32 %j_next, %n
  br i1 %j_done, label %outer_latch, label %inner

outer_latch:
  %asum = fmul double %alpha, %acc_next
  %y_new = fadd double %y_init, %asum
  store double %y_new, ptr %y_ptr, align 8
  %i_next = add i32 %i, 1
  %i_done = icmp eq i32 %i_next, %n
  br i1 %i_done, label %exit, label %outer

exit:
  ret void
}

define internal void @spr2_impl(i1 %row_major, i1 %is_lower, i32 %n, double %alpha, ptr %x, i32 %incx, ptr %y, i32 %incy, ptr %ap) noinline {
entry:
  %le0 = icmp sle i32 %n, 0
  br i1 %le0, label %exit, label %init

init:
  %n_m1 = sub i32 1, %n
  %x_neg = icmp slt i32 %incx, 0
  %kx_neg = mul i32 %n_m1, %incx
  %kx = select i1 %x_neg, i32 %kx_neg, i32 0
  %y_neg = icmp slt i32 %incy, 0
  %ky_neg = mul i32 %n_m1, %incy
  %ky = select i1 %y_neg, i32 %ky_neg, i32 0
  %col_lower = xor i1 %is_lower, %row_major
  br label %outer

outer:
  %i = phi i32 [ 0, %init ], [ %i_next, %outer_latch ]
  br label %inner

inner:
  %j = phi i32 [ %i, %outer ], [ %j_next, %inner ]
  %xi_off = mul i32 %i, %incx
  %xi_idx = add i32 %kx, %xi_off
  %xi_ptr = getelementptr inbounds double, ptr %x, i32 %xi_idx
  %xi = load double, ptr %xi_ptr, align 8
  %xj_off = mul i32 %j, %incx
  %xj_idx = add i32 %kx, %xj_off
  %xj_ptr = getelementptr inbounds double, ptr %x, i32 %xj_idx
  %xj = load double, ptr %xj_ptr, align 8
  %yi_off = mul i32 %i, %incy
  %yi_idx = add i32 %ky, %yi_off
  %yi_ptr = getelementptr inbounds double, ptr %y, i32 %yi_idx
  %yi = load double, ptr %yi_ptr, align 8
  %yj_off = mul i32 %j, %incy
  %yj_idx = add i32 %ky, %yj_off
  %yj_ptr = getelementptr inbounds double, ptr %y, i32 %yj_idx
  %yj = load double, ptr %yj_ptr, align 8
  %p1 = fmul double %xi, %yj
  %p2 = fmul double %xj, %yi
  %sum = fadd double %p1, %p2
  %delta = fmul double %alpha, %sum
  %jp1 = add i32 %j, 1
  %j_prod = mul i32 %j, %jp1
  %j_tri = sdiv i32 %j_prod, 2
  %u_idx = add i32 %i, %j_tri
  %twon = mul i32 2, %n
  %twon_mi = sub i32 %twon, %i
  %twon_p1 = add i32 %twon_mi, 1
  %i_prod = mul i32 %i, %twon_p1
  %i_tri = sdiv i32 %i_prod, 2
  %jmi = sub i32 %j, %i
  %l_idx = add i32 %jmi, %i_tri
  %ap_idx = select i1 %col_lower, i32 %l_idx, i32 %u_idx
  %ap_ptr = getelementptr inbounds double, ptr %ap, i32 %ap_idx
  %ap_old = load double, ptr %ap_ptr, align 8
  %ap_new = fadd double %ap_old, %delta
  store double %ap_new, ptr %ap_ptr, align 8
  %j_next = add i32 %j, 1
  %j_done = icmp eq i32 %j_next, %n
  br i1 %j_done, label %outer_latch, label %inner

outer_latch:
  %i_next = add i32 %i, 1
  %i_done = icmp eq i32 %i_next, %n
  br i1 %i_done, label %exit, label %outer

exit:
  ret void
}

define internal double @dot_impl(i32 %n, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  %le0 = icmp sle i32 %n, 0
  br i1 %le0, label %ret0, label %init

ret0:
  ret double 0.0

init:
  %n_m1 = sub i32 1, %n
  %x_neg = icmp slt i32 %incx, 0
  %kx_neg = mul i32 %n_m1, %incx
  %kx = select i1 %x_neg, i32 %kx_neg, i32 0
  %y_neg = icmp slt i32 %incy, 0
  %ky_neg = mul i32 %n_m1, %incy
  %ky = select i1 %y_neg, i32 %ky_neg, i32 0
  br label %loop

loop:
  %i = phi i32 [ 0, %init ], [ %inext, %loop ]
  %acc = phi double [ 0.0, %init ], [ %acc_next, %loop ]
  %xoff = mul i32 %i, %incx
  %xidx = add i32 %kx, %xoff
  %xp = getelementptr inbounds double, ptr %x, i32 %xidx
  %xv = load double, ptr %xp, align 8
  %yoff = mul i32 %i, %incy
  %yidx = add i32 %ky, %yoff
  %yp = getelementptr inbounds double, ptr %y, i32 %yidx
  %yv = load double, ptr %yp, align 8
  %p = fmul double %xv, %yv
  %acc_next = fadd double %acc, %p
  %inext = add i32 %i, 1
  %done = icmp eq i32 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret double %acc_next
}

define internal void @scal_impl(i32 %n, double %a, ptr %x, i32 %incx) noinline {
entry:
  %le0 = icmp sle i32 %n, 0
  br i1 %le0, label %exit, label %init

init:
  %n_m1 = sub i32 1, %n
  %x_neg = icmp slt i32 %incx, 0
  %kx_neg = mul i32 %n_m1, %incx
  %kx = select i1 %x_neg, i32 %kx_neg, i32 0
  br label %loop

loop:
  %i = phi i32 [ 0, %init ], [ %inext, %loop ]
  %xoff = mul i32 %i, %incx
  %xidx = add i32 %kx, %xoff
  %xp = getelementptr inbounds double, ptr %x, i32 %xidx
  %xv = load double, ptr %xp, align 8
  %nv = fmul double %a, %xv
  store double %nv, ptr %xp, align 8
  %inext = add i32 %i, 1
  %done = icmp eq i32 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define internal void @copy_impl(i32 %n, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  %le0 = icmp sle i32 %n, 0
  br i1 %le0, label %exit, label %init

init:
  %n_m1 = sub i32 1, %n
  %x_neg = icmp slt i32 %incx, 0
  %kx_neg = mul i32 %n_m1, %incx
  %kx = select i1 %x_neg, i32 %kx_neg, i32 0
  %y_neg = icmp slt i32 %incy, 0
  %ky_neg = mul i32 %n_m1, %incy
  %ky = select i1 %y_neg, i32 %ky_neg, i32 0
  br label %loop

loop:
  %i = phi i32 [ 0, %init ], [ %inext, %loop ]
  %xoff = mul i32 %i, %incx
  %xidx = add i32 %kx, %xoff
  %xp = getelementptr inbounds double, ptr %x, i32 %xidx
  %xv = load double, ptr %xp, align 8
  %yoff = mul i32 %i, %incy
  %yidx = add i32 %ky, %yoff
  %yp = getelementptr inbounds double, ptr %y, i32 %yidx
  store double %xv, ptr %yp, align 8
  %inext = add i32 %i, 1
  %done = icmp eq i32 %inext, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define void @dspmv_(ptr %uplo, ptr %n, ptr %alpha, ptr %ap, ptr %x, ptr %incx, ptr %beta, ptr %y, ptr %incy, i64 %len) noinline {
entry:
  %uc = load i8, ptr %uplo, align 1
  %isL = icmp eq i8 %uc, 76
  %isl = icmp eq i8 %uc, 108
  %is_lower = or i1 %isL, %isl
  %nv = load i32, ptr %n, align 4
  %av = load double, ptr %alpha, align 8
  %ix = load i32, ptr %incx, align 4
  %bv = load double, ptr %beta, align 8
  %iy = load i32, ptr %incy, align 4
  call void @spmv_impl(i1 false, i1 %is_lower, i32 %nv, double %av, ptr %ap, ptr %x, i32 %ix, double %bv, ptr %y, i32 %iy)
  ret void
}

define void @dspr2_(ptr %uplo, ptr %n, ptr %alpha, ptr %x, ptr %incx, ptr %y, ptr %incy, ptr %ap, i64 %len) noinline {
entry:
  %uc = load i8, ptr %uplo, align 1
  %isL = icmp eq i8 %uc, 76
  %isl = icmp eq i8 %uc, 108
  %is_lower = or i1 %isL, %isl
  %nv = load i32, ptr %n, align 4
  %av = load double, ptr %alpha, align 8
  %ix = load i32, ptr %incx, align 4
  %iy = load i32, ptr %incy, align 4
  call void @spr2_impl(i1 false, i1 %is_lower, i32 %nv, double %av, ptr %x, i32 %ix, ptr %y, i32 %iy, ptr %ap)
  ret void
}

define double @ddot_(ptr %n, ptr %x, ptr %incx, ptr %y, ptr %incy) noinline {
entry:
  %nv = load i32, ptr %n, align 4
  %ix = load i32, ptr %incx, align 4
  %iy = load i32, ptr %incy, align 4
  %r = call double @dot_impl(i32 %nv, ptr %x, i32 %ix, ptr %y, i32 %iy)
  ret double %r
}

define void @dscal_(ptr %n, ptr %a, ptr %x, ptr %incx) noinline {
entry:
  %nv = load i32, ptr %n, align 4
  %av = load double, ptr %a, align 8
  %ix = load i32, ptr %incx, align 4
  call void @scal_impl(i32 %nv, double %av, ptr %x, i32 %ix)
  ret void
}

define void @dcopy_(ptr %n, ptr %x, ptr %incx, ptr %y, ptr %incy) noinline {
entry:
  %nv = load i32, ptr %n, align 4
  %ix = load i32, ptr %incx, align 4
  %iy = load i32, ptr %incy, align 4
  call void @copy_impl(i32 %nv, ptr %x, i32 %ix, ptr %y, i32 %iy)
  ret void
}

define void @cblas_dspmv(i32 %layout, i32 %uplo, i32 %n, double %alpha, ptr %ap, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy) noinline {
entry:
  %rm = icmp eq i32 %layout, 101
  %il = icmp eq i32 %uplo, 122
  call void @spmv_impl(i1 %rm, i1 %il, i32 %n, double %alpha, ptr %ap, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy)
  ret void
}

define void @cblas_dspr2(i32 %layout, i32 %uplo, i32 %n, double %alpha, ptr %x, i32 %incx, ptr %y, i32 %incy, ptr %ap) noinline {
entry:
  %rm = icmp eq i32 %layout, 101
  %il = icmp eq i32 %uplo, 122
  call void @spr2_impl(i1 %rm, i1 %il, i32 %n, double %alpha, ptr %x, i32 %incx, ptr %y, i32 %incy, ptr %ap)
  ret void
}

define double @cblas_ddot(i32 %n, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  %r = call double @dot_impl(i32 %n, ptr %x, i32 %incx, ptr %y, i32 %incy)
  ret double %r
}

define void @cblas_dscal(i32 %n, double %a, ptr %x, i32 %incx) noinline {
entry:
  call void @scal_impl(i32 %n, double %a, ptr %x, i32 %incx)
  ret void
}

define void @cblas_dcopy(i32 %n, ptr %x, i32 %incx, ptr %y, i32 %incy) noinline {
entry:
  call void @copy_impl(i32 %n, ptr %x, i32 %incx, ptr %y, i32 %incy)
  ret void
}

; 1) Fortran dspmv_ with AP overwritten (tests ArgType::ap n*(n+1)/2 caching,
;    vector-sized y0 temporary, and fixed DiagUpdateSPMV arg offsets & upper step)
define void @test_fortran(ptr noalias %uplo, i32 %n, i32 %incx, i32 %incy, ptr noalias %alpha, ptr noalias %ap, ptr noalias %x, ptr noalias %beta, ptr noalias %y) {
entry:
  %na = alloca i32, align 4
  %ixa = alloca i32, align 4
  %iya = alloca i32, align 4
  store i32 %n, ptr %na, align 4
  store i32 %incx, ptr %ixa, align 4
  store i32 %incy, ptr %iya, align 4
  call void @dspmv_(ptr %uplo, ptr %na, ptr %alpha, ptr %ap, ptr %x, ptr %ixa, ptr %beta, ptr %y, ptr %iya, i64 1)
  ; Overwrite AP so cache.ap must copy all n*(n+1)/2 doubles
  %np1 = add i32 %n, 1
  %prod = mul i32 %n, %np1
  %len = sdiv i32 %prod, 2
  %has_elems = icmp sgt i32 %len, 0
  br i1 %has_elems, label %clobber, label %done

clobber:
  %i = phi i32 [ 0, %entry ], [ %inext, %clobber ]
  %p = getelementptr inbounds double, ptr %ap, i32 %i
  store double 999.0, ptr %p, align 8
  %inext = add i32 %i, 1
  %cdone = icmp eq i32 %inext, %len
  br i1 %cdone, label %done, label %clobber

done:
  ret void
}

; 2) CBLAS cblas_dspmv with layout + negative stride + AP overwritten
define void @test_cblas(i32 %layout, i32 %uplo, i32 %n, i32 %incx, i32 %incy, double %alpha, ptr noalias %ap, ptr noalias %x, double %beta, ptr noalias %y) {
entry:
  call void @cblas_dspmv(i32 %layout, i32 %uplo, i32 %n, double %alpha, ptr %ap, ptr %x, i32 %incx, double %beta, ptr %y, i32 %incy)
  %np1 = add i32 %n, 1
  %prod = mul i32 %n, %np1
  %len = sdiv i32 %prod, 2
  %has_elems = icmp sgt i32 %len, 0
  br i1 %has_elems, label %clobber, label %done

clobber:
  %i = phi i32 [ 0, %entry ], [ %inext, %clobber ]
  %p = getelementptr inbounds double, ptr %ap, i32 %i
  store double 999.0, ptr %p, align 8
  %inext = add i32 %i, 1
  %cdone = icmp eq i32 %inext, %len
  br i1 %cdone, label %done, label %clobber

done:
  ret void
}

; 3) Loop-varying CBLAS layout + loop-loaded AP pointer whose slot is overwritten
;    (tests lookup(arg_layout, Builder2) and blasDiffUseUpdater ArgType::ap shadow)
define void @test_loop_layout_and_ap(ptr noalias %layouts, ptr noalias %ap_ptrs, ptr noalias %x, ptr noalias %y) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inext, %loop ]
  %lslot = getelementptr inbounds i32, ptr %layouts, i64 %i
  %lay = load i32, ptr %lslot, align 4
  store i32 0, ptr %lslot, align 4
  %aslot = getelementptr inbounds ptr, ptr %ap_ptrs, i64 %i
  %ap = load ptr, ptr %aslot, align 8
  store ptr null, ptr %aslot, align 8
  call void @cblas_dspmv(i32 %lay, i32 121, i32 3, double 1.0, ptr %ap, ptr %x, i32 1, double 1.0, ptr %y, i32 1)
  %inext = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %inext, 2
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

define i32 @main() {
entry:
  ; --- Test 1: Fortran dspmv_ 'U', n=3, ap overwritten ---
  ; Matrix A (upper col-major):
  ; [1 2 4]
  ; [2 3 5]
  ; [4 5 6]
  ; ap = [1, 2, 3, 4, 5, 6], x = [1, 2, 3], y = [1, 1, 1], dy = [2, 3, 5], alpha = 2, beta = 3
  ; A*x = [17, 23, 32], dalpha = dy.(A*x) = 34 + 69 + 160 = 263.0
  ; dbeta = dy.y_in = 10.0
  ; dx = 2 * A*dy = 2 * [28, 38, 53] = [56, 76, 106]
  ; dap (upper): [2*1*2, 2*(1*3+2*2), 2*2*3, 2*(1*5+3*2), 2*(2*5+3*3), 2*3*5]
  ;            = [4, 14, 12, 22, 38, 30]
  %u = alloca i8, align 1
  store i8 85, ptr %u, align 1
  %alpha = alloca double, align 8
  %dalpha = alloca double, align 8
  store double 2.0, ptr %alpha, align 8
  store double 0.0, ptr %dalpha, align 8
  %beta = alloca double, align 8
  %dbeta = alloca double, align 8
  store double 3.0, ptr %beta, align 8
  store double 0.0, ptr %dbeta, align 8
  %ap = alloca [6 x double], align 8
  %dap = alloca [6 x double], align 8
  %x = alloca [5 x double], align 8
  %dx = alloca [5 x double], align 8
  %y = alloca [3 x double], align 8
  %dy = alloca [3 x double], align 8
  br label %init_f

init_f:
  store double 1.0, ptr %ap, align 8
  %ap1 = getelementptr inbounds double, ptr %ap, i64 1
  store double 2.0, ptr %ap1, align 8
  %ap2 = getelementptr inbounds double, ptr %ap, i64 2
  store double 3.0, ptr %ap2, align 8
  %ap3 = getelementptr inbounds double, ptr %ap, i64 3
  store double 4.0, ptr %ap3, align 8
  %ap4 = getelementptr inbounds double, ptr %ap, i64 4
  store double 5.0, ptr %ap4, align 8
  %ap5 = getelementptr inbounds double, ptr %ap, i64 5
  store double 6.0, ptr %ap5, align 8
  store [6 x double] zeroinitializer, ptr %dap, align 8
  store double 1.0, ptr %x, align 8
  %x1 = getelementptr inbounds double, ptr %x, i64 1
  store double 2.0, ptr %x1, align 8
  %x2 = getelementptr inbounds double, ptr %x, i64 2
  store double 3.0, ptr %x2, align 8
  store [5 x double] zeroinitializer, ptr %dx, align 8
  store double 1.0, ptr %y, align 8
  %y1 = getelementptr inbounds double, ptr %y, i64 1
  store double 1.0, ptr %y1, align 8
  %y2 = getelementptr inbounds double, ptr %y, i64 2
  store double 1.0, ptr %y2, align 8
  store double 2.0, ptr %dy, align 8
  %dy1 = getelementptr inbounds double, ptr %dy, i64 1
  store double 3.0, ptr %dy1, align 8
  %dy2 = getelementptr inbounds double, ptr %dy, i64 2
  store double 5.0, ptr %dy2, align 8

  call void (ptr, ...) @__enzyme_autodiff(ptr @test_fortran,
    metadata !"enzyme_const", ptr %u,
    metadata !"enzyme_const", i32 3,
    metadata !"enzyme_const", i32 1,
    metadata !"enzyme_const", i32 1,
    metadata !"enzyme_dup", ptr %alpha, ptr %dalpha,
    metadata !"enzyme_dup", ptr %ap, ptr %dap,
    metadata !"enzyme_dup", ptr %x, ptr %dx,
    metadata !"enzyme_dup", ptr %beta, ptr %dbeta,
    metadata !"enzyme_dup", ptr %y, ptr %dy)

  %da = load double, ptr %dalpha, align 8
  %db = load double, ptr %dbeta, align 8
  %dx0 = load double, ptr %dx, align 8
  %dx1p = getelementptr inbounds double, ptr %dx, i64 1
  %dx1 = load double, ptr %dx1p, align 8
  %dx2p = getelementptr inbounds double, ptr %dx, i64 2
  %dx2 = load double, ptr %dx2p, align 8
  %dap0 = load double, ptr %dap, align 8
  %dap2p = getelementptr inbounds double, ptr %dap, i64 2
  %dap2 = load double, ptr %dap2p, align 8
  %dap4p = getelementptr inbounds double, ptr %dap, i64 4
  %dap4 = load double, ptr %dap4p, align 8
  %dap5p = getelementptr inbounds double, ptr %dap, i64 5
  %dap5 = load double, ptr %dap5p, align 8
  %p1 = call i32 (ptr, ...) @printf(ptr @fmt_f, double %da, double %db, double %dx0, double %dx1, double %dx2, double %dap0, double %dap2, double %dap4, double %dap5)

  ; --- Test 2: CBLAS RowMajor (101) Upper (121), incx=-2, incy=-1 ---
  ; x_buf = [3, 99, 2, 99, 1], incx=-2 => logical x = [1, 2, 3]
  ; dy_buf = [13, 11, 7], incy=-1 => logical dy = [7, 11, 13]
  ; RowMajor Upper stores [A00, A01, A02, A11, A12, A22]:
  ; dap = [1*7, 1*11+2*7, 1*13+3*7, 2*11, 2*13+3*11, 3*13]
  ;     = [7.0, 25.0, 34.0, 22.0, 59.0, 39.0]
  store double 1.0, ptr %ap, align 8
  store double 2.0, ptr %ap1, align 8
  store double 3.0, ptr %ap2, align 8
  store double 4.0, ptr %ap3, align 8
  store double 5.0, ptr %ap4, align 8
  store double 6.0, ptr %ap5, align 8
  store [6 x double] zeroinitializer, ptr %dap, align 8
  store double 3.0, ptr %x, align 8
  store double 99.0, ptr %x1, align 8
  store double 2.0, ptr %x2, align 8
  %x3 = getelementptr inbounds double, ptr %x, i64 3
  store double 99.0, ptr %x3, align 8
  %x4 = getelementptr inbounds double, ptr %x, i64 4
  store double 1.0, ptr %x4, align 8
  store [5 x double] zeroinitializer, ptr %dx, align 8
  store [3 x double] zeroinitializer, ptr %y, align 8
  store double 13.0, ptr %dy, align 8
  store double 11.0, ptr %dy1, align 8
  store double 7.0, ptr %dy2, align 8

  call void (ptr, ...) @__enzyme_autodiff(ptr @test_cblas,
    metadata !"enzyme_const", i32 101,
    metadata !"enzyme_const", i32 121,
    metadata !"enzyme_const", i32 3,
    metadata !"enzyme_const", i32 -2,
    metadata !"enzyme_const", i32 -1,
    metadata !"enzyme_const", double 1.0,
    metadata !"enzyme_dup", ptr %ap, ptr %dap,
    metadata !"enzyme_dup", ptr %x, ptr %dx,
    metadata !"enzyme_const", double 0.0,
    metadata !"enzyme_dup", ptr %y, ptr %dy)

  %c0 = load double, ptr %dap, align 8
  %c1p = getelementptr inbounds double, ptr %dap, i64 1
  %c1 = load double, ptr %c1p, align 8
  %c2 = load double, ptr %dap2p, align 8
  %c3p = getelementptr inbounds double, ptr %dap, i64 3
  %c3 = load double, ptr %c3p, align 8
  %c4 = load double, ptr %dap4p, align 8
  %c5 = load double, ptr %dap5p, align 8
  %p2 = call i32 (ptr, ...) @printf(ptr @fmt_c, i32 101, i32 121, double %c0, double %c1, double %c2, double %c3, double %c4, double %c5)

  ; --- Test 3: CBLAS RowMajor (101) Lower (122), incx=1, incy=1 ---
  ; RowMajor Lower stores [A00, A10, A11, A20, A21, A22]:
  ; logical x = [1, 2, 3], logical dy = [7, 11, 13]
  ; dap = [1*7, 2*7+1*11, 2*11, 3*7+1*13, 3*11+2*13, 3*13]
  ;     = [7.0, 25.0, 22.0, 34.0, 59.0, 39.0]
  store [6 x double] zeroinitializer, ptr %dap, align 8
  store double 1.0, ptr %x, align 8
  store double 2.0, ptr %x1, align 8
  store double 3.0, ptr %x2, align 8
  store [5 x double] zeroinitializer, ptr %dx, align 8
  store [3 x double] zeroinitializer, ptr %y, align 8
  store double 7.0, ptr %dy, align 8
  store double 11.0, ptr %dy1, align 8
  store double 13.0, ptr %dy2, align 8

  call void (ptr, ...) @__enzyme_autodiff(ptr @test_cblas,
    metadata !"enzyme_const", i32 101,
    metadata !"enzyme_const", i32 122,
    metadata !"enzyme_const", i32 3,
    metadata !"enzyme_const", i32 1,
    metadata !"enzyme_const", i32 1,
    metadata !"enzyme_const", double 1.0,
    metadata !"enzyme_dup", ptr %ap, ptr %dap,
    metadata !"enzyme_dup", ptr %x, ptr %dx,
    metadata !"enzyme_const", double 0.0,
    metadata !"enzyme_dup", ptr %y, ptr %dy)

  %r0 = load double, ptr %dap, align 8
  %r1 = load double, ptr %c1p, align 8
  %r2 = load double, ptr %dap2p, align 8
  %r3 = load double, ptr %c3p, align 8
  %r4 = load double, ptr %dap4p, align 8
  %r5 = load double, ptr %dap5p, align 8
  %p3 = call i32 (ptr, ...) @printf(ptr @fmt_c, i32 101, i32 122, double %r0, double %r1, double %r2, double %r3, double %r4, double %r5)

  ; --- Test 4: Loop-varying layout (101 then 102) + loop-loaded ap pointer ---
  ; Both iterations share the same %dap shadow (one RowMajor Upper, one ColMajor Upper),
  ; x = [1, 2, 3], dy = [7, 11, 13], beta = 1 => dy unchanged across iterations:
  ; RowMajor Upper: [7, 25, 34, 22, 59, 39]
  ; ColMajor Upper: [7, 25, 22, 34, 59, 39]
  ; Sum:            [14, 50, 56, 56, 118, 78]
  %lays = alloca [2 x i32], align 4
  store i32 101, ptr %lays, align 4
  %lays1 = getelementptr inbounds i32, ptr %lays, i64 1
  store i32 102, ptr %lays1, align 4
  %ap_ptrs = alloca [2 x ptr], align 8
  %dap_ptrs = alloca [2 x ptr], align 8
  store ptr %ap, ptr %ap_ptrs, align 8
  %app1 = getelementptr inbounds ptr, ptr %ap_ptrs, i64 1
  store ptr %ap, ptr %app1, align 8
  store ptr %dap, ptr %dap_ptrs, align 8
  %dapp1 = getelementptr inbounds ptr, ptr %dap_ptrs, i64 1
  store ptr %dap, ptr %dapp1, align 8
  store [6 x double] zeroinitializer, ptr %dap, align 8
  store [3 x double] zeroinitializer, ptr %y, align 8
  store double 7.0, ptr %dy, align 8
  store double 11.0, ptr %dy1, align 8
  store double 13.0, ptr %dy2, align 8

  call void (ptr, ...) @__enzyme_autodiff(ptr @test_loop_layout_and_ap,
    metadata !"enzyme_const", ptr %lays,
    metadata !"enzyme_dup", ptr %ap_ptrs, ptr %dap_ptrs,
    metadata !"enzyme_const", ptr %x,
    metadata !"enzyme_dup", ptr %y, ptr %dy)

  %l0 = load double, ptr %dap, align 8
  %l1 = load double, ptr %c1p, align 8
  %l2 = load double, ptr %dap2p, align 8
  %l3 = load double, ptr %c3p, align 8
  %l4 = load double, ptr %dap4p, align 8
  %l5 = load double, ptr %dap5p, align 8
  %p4 = call i32 (ptr, ...) @printf(ptr @fmt_l, double %l0, double %l1, double %l2, double %l3, double %l4, double %l5)

  ret i32 0
}

; Verify distinct Fortran vs CBLAS helper names and fixed signatures/math:
; CHECK-LABEL: define internal void @__enzyme_spmv_diagd_(ptr %blasuplo, ptr %blasn, ptr noalias {{.*}}readonly %blasalpha, ptr noalias {{.*}}readonly %blasx, ptr %blasincx, ptr noalias {{.*}}readonly %blasdy, ptr %blasincy, ptr noalias {{.*}}%blasdAP)
; CHECK: uper:
; CHECK: %iter.next = add i64 %iteration, 1
; CHECK-NEXT: %k.step = add i64 %iter.next, 1
; CHECK-NEXT: %k.next = add i64 %k, %k.step

; CHECK-LABEL: define internal void @__enzyme_spmv_diagcblas_d(i32 %blaslayout, i32 %blasuplo, i32 %blasn, double %blasalpha, ptr noalias {{.*}}readonly %blasx, i32 %blasincx, ptr noalias {{.*}}readonly %blasdy, i32 %blasincy, ptr noalias {{.*}}%blasdAP)
; CHECK: icmp eq i32 %blaslayout, 101
; CHECK: xor i1
; CHECK: icmp sle i32 %blasn, 0

; Verify E2E numerical derivatives:
; EVAL: F 263.0 10.0 | 56.0 76.0 106.0 | 4.0 12.0 38.0 30.0
; EVAL-NEXT: C 101 121 | 7.0 25.0 34.0 22.0 59.0 39.0
; EVAL-NEXT: C 101 122 | 7.0 25.0 22.0 34.0 59.0 39.0
; EVAL-NEXT: L | 14.0 50.0 56.0 56.0 118.0 78.0

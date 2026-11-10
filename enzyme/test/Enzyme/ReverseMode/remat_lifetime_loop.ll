; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

%struct.Data = type { i32, i32, i32*, double* }

@enzyme_const = external global i32

declare double @exp(double) nounwind readnone
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
declare void @__enzyme_autodiff(i8*, ...)

define double @energy(double* nocapture readonly %D, %struct.Data* nocapture readonly %d) {
entry:
  %phi = alloca [3 x double], align 16
  %phi_i8 = bitcast [3 x double]* %phi to i8*
  %np_ptr = getelementptr inbounds %struct.Data, %struct.Data* %d, i64 0, i32 1
  %np0 = load i32, i32* %np_ptr, align 4
  %cmp_np0 = icmp sgt i32 %np0, 0
  br i1 %cmp_np0, label %p_preheader, label %exit

p_preheader:
  %nk_field = getelementptr inbounds %struct.Data, %struct.Data* %d, i64 0, i32 2
  %x_field = getelementptr inbounds %struct.Data, %struct.Data* %d, i64 0, i32 3
  %n_field = getelementptr inbounds %struct.Data, %struct.Data* %d, i64 0, i32 0
  %n0 = load i32, i32* %n_field, align 8
  br label %p_header

exit:
  %res = phi double [ 0.0, %entry ], [ %s_next, %p_latch ]
  ret double %res

p_header:
  %n_p = phi i32 [ %n0, %p_preheader ], [ %n_latch, %p_latch ]
  %p = phi i64 [ 0, %p_preheader ], [ %p_next, %p_latch ]
  %s = phi double [ 0.0, %p_preheader ], [ %s_next, %p_latch ]
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %phi_i8)
  %cmp_n = icmp sgt i32 %n_p, 0
  br i1 %cmp_n, label %fill_preheader, label %p_latch

fill_preheader:
  %nk_base = load i32*, i32** %nk_field, align 8
  br label %fill_header

fill_header:
  %n_fill1 = phi i32 [ %n_p, %fill_preheader ], [ %n_after_k, %fill_latch ]
  %n_fill2 = phi i32 [ %n_p, %fill_preheader ], [ %n_after_k, %fill_latch ]
  %nk_cur = phi i32* [ %nk_base, %fill_preheader ], [ %nk_after_k, %fill_latch ]
  %i = phi i64 [ 0, %fill_preheader ], [ %i_next, %fill_latch ]
  %nk_gep = getelementptr inbounds i32, i32* %nk_cur, i64 %i
  %nk_val = load i32, i32* %nk_gep, align 4
  %cmp_nk = icmp sgt i32 %nk_val, 0
  br i1 %cmp_nk, label %k_preheader, label %fill_latch

k_preheader:
  %i_tr = trunc i64 %i to i32
  %i_fp = sitofp i32 %i_tr to double
  %i_off = fadd double %i_fp, 5.000000e-01
  br label %k_body

k_body:
  %k = phi i32 [ 0, %k_preheader ], [ %k_next, %k_body ]
  %v_acc = phi double [ 0.0, %k_preheader ], [ %v_next, %k_body ]
  %k_fp = sitofp i32 %k to double
  %ik = fadd double %i_off, %k_fp
  %neg_ik = fneg double %ik
  %x_base = load double*, double** %x_field, align 8
  %x_gep = getelementptr inbounds double, double* %x_base, i64 %p
  %xp = load double, double* %x_gep, align 8
  %arg = fmul double %xp, %neg_ik
  %e = tail call double @exp(double %arg)
  %v_next = fadd double %v_acc, %e
  %k_next = add nuw nsw i32 %k, 1
  %nk_reload = load i32*, i32** %nk_field, align 8
  %nk_gep2 = getelementptr inbounds i32, i32* %nk_reload, i64 %i
  %nk_lim = load i32, i32* %nk_gep2, align 4
  %cmp_k = icmp slt i32 %k_next, %nk_lim
  br i1 %cmp_k, label %k_body, label %k_exit

k_exit:
  %n_reload = load i32, i32* %n_field, align 8
  br label %fill_latch

fill_latch:
  %n_after_k = phi i32 [ %n_fill1, %fill_header ], [ %n_reload, %k_exit ]
  %n_after_k2 = phi i32 [ %n_fill2, %fill_header ], [ %n_reload, %k_exit ]
  %nk_after_k = phi i32* [ %nk_cur, %fill_header ], [ %nk_reload, %k_exit ]
  %v_final = phi double [ 0.0, %fill_header ], [ %v_next, %k_exit ]
  %phi_gep = getelementptr inbounds [3 x double], [3 x double]* %phi, i64 0, i64 %i
  store double %v_final, double* %phi_gep, align 8
  %i_next = add nuw nsw i64 %i, 1
  %n_ext = sext i32 %n_after_k2 to i64
  %cmp_i = icmp slt i64 %i_next, %n_ext
  br i1 %cmp_i, label %fill_header, label %rho_check

rho_check:
  %cmp_rho = icmp sgt i32 %n_after_k, 0
  br i1 %cmp_rho, label %rho_preheader, label %p_latch

rho_preheader:
  %n_zext = zext i32 %n_after_k to i64
  br label %rho_i_loop

rho_i_loop:
  %ri = phi i64 [ 0, %rho_preheader ], [ %ri_next, %rho_i_latch ]
  %rho_acc = phi double [ 0.0, %rho_preheader ], [ %rho_j_next, %rho_i_latch ]
  %phi_ri_ptr = getelementptr inbounds [3 x double], [3 x double]* %phi, i64 0, i64 %ri
  %phi_ri = load double, double* %phi_ri_ptr, align 8
  %ri_tr = trunc i64 %ri to i32
  %row_off = mul i32 %n_after_k, %ri_tr
  %row_ext = zext i32 %row_off to i64
  %D_row = getelementptr inbounds double, double* %D, i64 %row_ext
  br label %rho_j_loop

rho_j_loop:
  %rj = phi i64 [ 0, %rho_i_loop ], [ %rj_next, %rho_j_loop ]
  %rho_j_acc = phi double [ %rho_acc, %rho_i_loop ], [ %rho_j_next, %rho_j_loop ]
  %D_ptr = getelementptr inbounds double, double* %D_row, i64 %rj
  %D_val = load double, double* %D_ptr, align 8
  %prod1 = fmul double %phi_ri, %D_val
  %phi_rj_ptr = getelementptr inbounds [3 x double], [3 x double]* %phi, i64 0, i64 %rj
  %phi_rj = load double, double* %phi_rj_ptr, align 8
  %prod2 = fmul double %prod1, %phi_rj
  %rho_j_next = fadd double %rho_j_acc, %prod2
  %rj_next = add nuw nsw i64 %rj, 1
  %cmp_rj = icmp eq i64 %rj_next, %n_zext
  br i1 %cmp_rj, label %rho_i_latch, label %rho_j_loop

rho_i_latch:
  %ri_next = add nuw nsw i64 %ri, 1
  %cmp_ri = icmp eq i64 %ri_next, %n_zext
  br i1 %cmp_ri, label %p_latch, label %rho_i_loop

p_latch:
  %n_latch = phi i32 [ %n_p, %p_header ], [ %n_after_k, %rho_check ], [ %n_after_k, %rho_i_latch ]
  %rho_final = phi double [ 0.0, %p_header ], [ 0.0, %rho_check ], [ %rho_j_next, %rho_i_latch ]
  %rho_sq = fmul double %rho_final, %rho_final
  %s_next = fadd double %s, %rho_sq
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %phi_i8)
  %p_next = add nuw nsw i64 %p, 1
  %np_reload = load i32, i32* %np_ptr, align 4
  %np_ext = sext i32 %np_reload to i64
  %cmp_p = icmp slt i64 %p_next, %np_ext
  br i1 %cmp_p, label %p_header, label %exit
}

define void @caller(double* %D, double* %dD, %struct.Data* %d) {
entry:
  %c = load i32, i32* @enzyme_const
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, %struct.Data*)* @energy to i8*), double* %D, double* %dD, i32 %c, %struct.Data* %d)
  ret void
}

; CHECK: define internal void @diffeenergy(
; CHECK-LABEL: remat_p_header_fill_latch:
; CHECK: %phi_gep_unwrap = getelementptr inbounds [3 x double], {{.*}} %phi, i64 0, i64 %fiv
; CHECK: store double {{.*}}, {{.*}} %phi_gep_unwrap
; CHECK-LABEL: remat_p_header_rho_j_loop:
; CHECK-NOT: load i64, {{.*}} %"iv{{[0-9]+}}'ac"
; CHECK-NOT: i64 %fiv{{$}}
; CHECK: br i1 %cmp_nk_unwrap

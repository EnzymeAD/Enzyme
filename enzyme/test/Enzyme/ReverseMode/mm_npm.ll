; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,mem2reg,early-cse-memssa,correlated-propagation,simplifycfg,instcombine,adce"  -enzyme-preopt=false -S | FileCheck %s

declare dso_local void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #0

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #1

declare dso_local void @free(i8* nocapture)

; Function Attrs: nounwind uwtable
define dso_local void @mat_mult(i64 %rows, i64 %cols, float** noalias nocapture readonly %lhs, float** noalias nocapture readonly %rhs, double** noalias nocapture %out) local_unnamed_addr #2 {
entry:
  %lhs_data = load float*, float** %lhs, align 8, !tbaa !2
  %rhs_data = load float*, float** %rhs, align 8, !tbaa !2
  %out_data = load double*, double** %out, align 8, !tbaa !2
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.inc47, %entry
  %i = phi i64 [ 0, %entry ], [ %nexti, %for.inc47 ]
  %lhs_i = getelementptr inbounds float, float* %lhs_data, i64 %i
  br label %for.body5

for.body5:                                        ; preds = %for.inc44, %for.cond2.preheader
  %j = phi i64 [ 0, %for.cond2.preheader ], [ %nextj, %for.inc44 ]
  %L_lhs_i = load float, float* %lhs_i, align 8, !tbaa !8
  %0 = load float, float* %lhs_i, align 8, !tbaa !8
  %1 = mul nsw i64 %j, %rows
  %rhs_ij = getelementptr inbounds float, float* %rhs_data, i64 %1
  %L_rhs_ij = load float, float* %rhs_ij, align 8, !tbaa !8
  %mul = fmul fast float %L_rhs_ij, %L_lhs_i
  %mul_ext = fpext float %mul to double
  %2 = add nsw i64 %1, %i
  %out_iji = getelementptr inbounds double, double* %out_data, i64 %2
  store double %mul_ext, double* %out_iji, align 8, !tbaa !6
  br label %for.body23

for.body23:                                       ; preds = %for.body23, %for.body5
  %3 = phi float [ %add, %for.body23 ], [ %mul, %for.body5 ]
  %k = phi i64 [ %nextk, %for.body23 ], [ 1, %for.body5 ]
  %4 = mul nsw i64 %k, %rows
  %5 = add nsw i64 %4, %i
  %lhs_ki = getelementptr inbounds float, float* %lhs_data, i64 %5
  %L_lhs_ki = load float, float* %lhs_ki, align 8, !tbaa !8
  %6 = add nsw i64 %k, %1
  %rhs_kk = getelementptr inbounds float, float* %rhs_data, i64 %6
  %L_rhs_kk = load float, float* %rhs_kk, align 8, !tbaa !8
  %mul2 = fmul fast float %L_rhs_kk, %L_lhs_ki
  %add = fadd fast float %3, %mul2
  %add_ext = fpext float %add to double
  store double %add_ext, double* %out_iji, align 8, !tbaa !6
  %nextk = add nuw nsw i64 %k, 1
  %cmp22 = icmp slt i64 %nextk, %cols
  br i1 %cmp22, label %for.body23, label %for.inc44

for.inc44:                                        ; preds = %for.body23
  %nextj = add nuw nsw i64 %j, 1
  %exitcond = icmp eq i64 %nextj, %cols
  br i1 %exitcond, label %for.inc47, label %for.body5

for.inc47:                                        ; preds = %for.inc44
  %nexti = add nuw nsw i64 %i, 1
  %exitcond99 = icmp eq i64 %nexti, %rows
  br i1 %exitcond99, label %for.end49, label %for.cond2.preheader

for.end49:                                        ; preds = %for.inc47
  ret void
}

define void @caller(i64 %rows, i64 %cols, float** %A, float** %dA, float** %B, float** %dB, double** %C, double** %dC) {
entry:
  tail call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (i64, i64, float**, float**, double**)* @mat_mult to i8*), i64 %rows, i64 %cols, float** %A, float** %dA, float** %B, float** %dB, double** %C, double** %dC)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #3

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.1.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !4, i64 0}

; CHECK: define internal void @diffemat_mult(i64 %rows, i64 %cols, float** noalias nocapture readonly %lhs, float** nocapture %"lhs'", float** noalias nocapture readonly %rhs, float** nocapture %"rhs'", double** noalias nocapture %out, double** nocapture %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[dlhs_data:.+]] = load float*, float** %"lhs'", align 8
; CHECK-NEXT:   %lhs_data = load float*, float** %lhs, align 8, !tbaa !2
; CHECK-NEXT:   %[[drhs_data:.+]] = load float*, float** %"rhs'", align 8
; CHECK-NEXT:   %rhs_data = load float*, float** %rhs, align 8, !tbaa !2
; CHECK-NEXT:   %[[dout_data:.+]] = load double*, double** %"out'", align 8
; CHECK-NEXT:   %out_data = load double*, double** %out, align 8, !tbaa !2
; TODO-MAXCHECK-NEXT:   %0 = icmp sgt i64 %cols, 2
; TODO-MAXCHECK-NEXT:   %smax = select i1 %0, i64 %cols, i64 2
; CHECK:   %[[smaxm2:.+]] = add nsw i64 %smax, -2
; CHECK-NEXT:   br label %for.cond2.preheader

; CHECK: for.cond2.preheader:                              ; preds = %for.inc47, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %for.inc47 ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %lhs_i = getelementptr inbounds float, float* %lhs_data, i64 %iv
; CHECK-NEXT:   br label %for.body5

; CHECK: for.body5:                                        ; preds = %for.inc44, %for.cond2.preheader
; CHECK-NEXT:   %iv1 = phi i64 [ %iv.next2, %for.inc44 ], [ 0, %for.cond2.preheader ]
; CHECK-NEXT:   %iv.next2 = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:   %L_lhs_i = load float, float* %lhs_i, align 8, !tbaa !6
; CHECK-NEXT:   %[[ivrows:.+]] = mul nsw i64 %iv1, %rows
; CHECK-NEXT:   %rhs_ij = getelementptr inbounds float, float* %rhs_data, i64 %[[ivrows]]
; CHECK-NEXT:   %L_rhs_ij = load float, float* %rhs_ij, align 8, !tbaa !6
; CHECK-NEXT:   %mul = fmul fast float %L_rhs_ij, %L_lhs_i
; CHECK-NEXT:   %mul_ext = fpext float %mul to double
; CHECK-NEXT:   %[[ivrowsiv:.+]] = add nsw i64 %[[ivrows]], %iv
; CHECK-NEXT:   %out_iji = getelementptr inbounds double, double* %out_data, i64 %[[ivrowsiv]]
; CHECK-NEXT:   store double %mul_ext, double* %out_iji, align 8, !tbaa !8
; CHECK-NEXT:   br label %for.body23

; CHECK: for.body23:                                       ; preds = %for.body23, %for.body5
; CHECK-NEXT:   %iv3 = phi i64 [ %iv.next4, %for.body23 ], [ 0, %for.body5 ]
; CHECK-NEXT:   %[[fphi:.+]] = phi float [ %add, %for.body23 ], [ %mul, %for.body5 ]
; CHECK-NEXT:   %iv.next4 = add nuw nsw i64 %iv3, 1
; CHECK-NEXT:   %[[n4rows:.+]] = mul nsw i64 %iv.next4, %rows
; CHECK-NEXT:   %[[riv:.+]] = add nsw i64 %[[n4rows]], %iv
; CHECK-NEXT:   %lhs_ki = getelementptr inbounds float, float* %lhs_data, i64 %[[riv]]
; CHECK-NEXT:   %L_lhs_ki = load float, float* %lhs_ki, align 8, !tbaa !6
; CHECK-NEXT:   %[[iv41:.+]] = add nsw i64 %iv.next4, %[[ivrows]]
; CHECK-NEXT:   %rhs_kk = getelementptr inbounds float, float* %rhs_data, i64 %[[iv41]]
; CHECK-NEXT:   %L_rhs_kk = load float, float* %rhs_kk, align 8, !tbaa !6
; CHECK-NEXT:   %mul2 = fmul fast float %L_rhs_kk, %L_lhs_ki
; CHECK-NEXT:   %add = fadd fast float %[[fphi]], %mul2
; CHECK-NEXT:   %add_ext = fpext float %add to double
; CHECK-NEXT:   store double %add_ext, double* %out_iji, align 8, !tbaa !8
; CHECK-NEXT:   %nextk = add nuw nsw i64 %iv3, 2
; CHECK-NEXT:   %cmp22 = icmp slt i64 %nextk, %cols
; CHECK-NEXT:   br i1 %cmp22, label %for.body23, label %for.inc44

; CHECK: for.inc44:                                        ; preds = %for.body23
; CHECK-NEXT:   %exitcond = icmp eq i64 %iv.next2, %cols
; CHECK-NEXT:   br i1 %exitcond, label %for.inc47, label %for.body5

; CHECK: for.inc47:                                        ; preds = %for.inc44
; CHECK-NEXT:   %exitcond99 = icmp eq i64 %iv.next, %rows
; CHECK-NEXT:   br i1 %exitcond99, label %invertfor.inc47, label %for.cond2.preheader

; CHECK: invertentry:                                      ; preds = %invertfor.cond2.preheader
; CHECK-NEXT:   ret void

; CHECK: invertfor.cond2.preheader:                        ; preds = %invertfor.body5
; CHECK-NEXT:   %[[ncmp:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[ncmp]], label %invertentry, label %incinvertfor.cond2.preheader

; CHECK: incinvertfor.cond2.preheader:                     ; preds = %invertfor.cond2.preheader
; CHECK-NEXT:   br label %invertfor.inc47

; CHECK: invertfor.body5:                                  ; preds = %invertfor.body23
; CHECK-NEXT:   store double 0.000000e+00, double* %"out_iji'ipg_unwrap4", align 8, !tbaa !8, !alias.scope !13, !noalias !10
; CHECK-NEXT:   %lhs_i_unwrap = getelementptr inbounds float, float* %lhs_data, i64 %"iv'ac.0"
; CHECK-NEXT:   %L_lhs_i_unwrap = load float, float* %lhs_i_unwrap, align 8, !tbaa !6
; CHECK-NEXT:   %m0diffeL_rhs_ij = fmul fast float %22, %L_lhs_i_unwrap
; CHECK-NEXT:   %rhs_ij_unwrap = getelementptr inbounds float, float* %rhs_data, i64 %_unwrap2
; CHECK-NEXT:   %L_rhs_ij_unwrap = load float, float* %rhs_ij_unwrap, align 8, !tbaa !6
; CHECK-NEXT:   %m1diffeL_lhs_i = fmul fast float %22, %L_rhs_ij_unwrap
; CHECK-NEXT:   %"rhs_ij'ipg_unwrap" = getelementptr inbounds float, float* %"rhs_data'ipl", i64 %_unwrap2
; CHECK-NEXT:   %9 = load float, float* %"rhs_ij'ipg_unwrap", align 8, !tbaa !6, !alias.scope !15, !noalias !18
; CHECK-NEXT:   %10 = fadd fast float %9, %m0diffeL_rhs_ij
; CHECK-NEXT:   store float %10, float* %"rhs_ij'ipg_unwrap", align 8, !tbaa !6, !alias.scope !15, !noalias !18
; CHECK-NEXT:   %"lhs_i'ipg_unwrap" = getelementptr inbounds float, float* %"lhs_data'ipl", i64 %"iv'ac.0"
; CHECK-NEXT:   %11 = load float, float* %"lhs_i'ipg_unwrap", align 8, !tbaa !6, !alias.scope !20, !noalias !23
; CHECK-NEXT:   %12 = fadd fast float %11, %m1diffeL_lhs_i
; CHECK-NEXT:   store float %12, float* %"lhs_i'ipg_unwrap", align 8, !tbaa !6, !alias.scope !20, !noalias !23
; CHECK-NEXT:   %13 = icmp eq i64 %"iv1'ac.0", 0
; CHECK-NEXT:   br i1 %13, label %invertfor.cond2.preheader, label %incinvertfor.body5

; CHECK: incinvertfor.body5:                               ; preds = %invertfor.body5
; CHECK-NEXT:   br label %invertfor.inc44

; CHECK: invertfor.body23:                                 ; preds = %invertfor.inc44, %incinvertfor.body23
; CHECK-NEXT:   %"add'de.0" = phi float [ 0.000000e+00, %invertfor.inc44 ], [ %16, %incinvertfor.body23 ]
; CHECK-NEXT:   %"iv3'ac.0" = phi i64 [ %[[smaxm2]], %invertfor.inc44 ], [ %23, %incinvertfor.body23 ]
; CHECK-NEXT:   %_unwrap2 = mul nsw i64 %"iv1'ac.0", %rows
; CHECK-NEXT:   %_unwrap3 = add nsw i64 %_unwrap2, %"iv'ac.0"
; CHECK-NEXT:   %"out_iji'ipg_unwrap4" = getelementptr inbounds double, double* %"out_data'ipl", i64 %_unwrap3
; CHECK-NEXT:   %14 = load double, double* %"out_iji'ipg_unwrap4", align 8, !tbaa !8, !noalias !25
; CHECK-NEXT:   store double 0.000000e+00, double* %"out_iji'ipg_unwrap4", align 8, !tbaa !8, !alias.scope !13, !noalias !10
; CHECK-NEXT:   %15 = fptrunc double %14 to float
; CHECK-NEXT:   %16 = fadd fast float %"add'de.0", %15
; CHECK-NEXT:   %iv.next4_unwrap = add nuw nsw i64 %"iv3'ac.0", 1
; CHECK-NEXT:   %_unwrap5 = mul nsw i64 %iv.next4_unwrap, %rows
; CHECK-NEXT:   %_unwrap6 = add nsw i64 %_unwrap5, %"iv'ac.0"
; CHECK-NEXT:   %lhs_ki_unwrap = getelementptr inbounds float, float* %lhs_data, i64 %_unwrap6
; CHECK-NEXT:   %L_lhs_ki_unwrap = load float, float* %lhs_ki_unwrap, align 8, !tbaa !6
; CHECK-NEXT:   %m0diffeL_rhs_kk = fmul fast float %16, %L_lhs_ki_unwrap
; CHECK-NEXT:   %_unwrap7 = add nsw i64 %iv.next4_unwrap, %_unwrap2
; CHECK-NEXT:   %rhs_kk_unwrap = getelementptr inbounds float, float* %rhs_data, i64 %_unwrap7
; CHECK-NEXT:   %L_rhs_kk_unwrap = load float, float* %rhs_kk_unwrap, align 8, !tbaa !6
; CHECK-NEXT:   %m1diffeL_lhs_ki = fmul fast float %16, %L_rhs_kk_unwrap
; CHECK-NEXT:   %"rhs_kk'ipg_unwrap" = getelementptr inbounds float, float* %"rhs_data'ipl", i64 %_unwrap7
; CHECK-NEXT:   %17 = load float, float* %"rhs_kk'ipg_unwrap", align 8, !tbaa !6, !alias.scope !26, !noalias !29
; CHECK-NEXT:   %18 = fadd fast float %17, %m0diffeL_rhs_kk
; CHECK-NEXT:   store float %18, float* %"rhs_kk'ipg_unwrap", align 8, !tbaa !6, !alias.scope !26, !noalias !29
; CHECK-NEXT:   %"lhs_ki'ipg_unwrap" = getelementptr inbounds float, float* %"lhs_data'ipl", i64 %_unwrap6
; CHECK-NEXT:   %19 = load float, float* %"lhs_ki'ipg_unwrap", align 8, !tbaa !6, !alias.scope !31, !noalias !34
; CHECK-NEXT:   %20 = fadd fast float %19, %m1diffeL_lhs_ki
; CHECK-NEXT:   store float %20, float* %"lhs_ki'ipg_unwrap", align 8, !tbaa !6, !alias.scope !31, !noalias !34
; CHECK-NEXT:   %21 = icmp eq i64 %"iv3'ac.0", 0
; CHECK-NEXT:   %22 = select fast i1 %21, float %16, float 0.000000e+00
; CHECK-NEXT:   br i1 %21, label %invertfor.body5, label %incinvertfor.body23

; CHECK: incinvertfor.body23:                              ; preds = %invertfor.body23
; CHECK-NEXT:   %23 = add nsw i64 %"iv3'ac.0", -1
; CHECK-NEXT:   br label %invertfor.body23

; CHECK: invertfor.inc44:                                  ; preds = %invertfor.inc47, %incinvertfor.body5
; CHECK-NEXT:   %"iv1'ac.0.in" = phi i64 [ %cols, %invertfor.inc47 ], [ %"iv1'ac.0", %incinvertfor.body5 ]
; CHECK-NEXT:   %"iv1'ac.0" = add i64 %"iv1'ac.0.in", -1
; CHECK-NEXT:   br label %invertfor.body23

; CHECK: invertfor.inc47:                                  ; preds = %for.inc47, %incinvertfor.cond2.preheader
; CHECK-NEXT:   %"iv'ac.0.in" = phi i64 [ %"iv'ac.0", %incinvertfor.cond2.preheader ], [ %rows, %for.inc47 ]
; CHECK-NEXT:   %"iv'ac.0" = add i64 %"iv'ac.0.in", -1
; CHECK-NEXT:   br label %invertfor.inc44
; CHECK-NEXT: }
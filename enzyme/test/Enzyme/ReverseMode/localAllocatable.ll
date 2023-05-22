; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s; fi

define float @square(float* noalias nocapture readonly dereferenceable(4) %X) {
alloca_1:
  %"X_PTR$_4" = alloca float*, align 8
  store float* null, float** %"X_PTR$_4", align 8, !tbaa !0
  %"(i8**)X_PTR$_4$" = bitcast float** %"X_PTR$_4" to i8**
  %func_result = call i32 @for_alloc_allocatable(i64 4, i8** nonnull %"(i8**)X_PTR$_4$", i32 262144) #0
  %X_fetch.1 = load float, float* %X, align 1, !tbaa !5
  %"X_PTR$_4_fetch.2" = load float*, float** %"X_PTR$_4", align 8, !tbaa !0
  %rel.not = icmp eq float* %"X_PTR$_4_fetch.2", null
  br i1 %rel.not, label %LHS_not_allocated_lab29, label %LHS_allocated_lab28

LHS_not_allocated_lab29:                          ; preds = %alloca_1
  %func_result4 = call i32 @for_alloc_allocatable_handle(i64 4, i8** nonnull %"(i8**)X_PTR$_4$", i32 262144, i8* null) #0
  %"X_PTR$_4_fetch.3.pre" = load float*, float** %"X_PTR$_4", align 8, !tbaa !0
  br label %LHS_allocated_lab28

LHS_allocated_lab28:                              ; preds = %LHS_not_allocated_lab29, %alloca_1
  %"X_PTR$_4_fetch.3" = phi float* [ %"X_PTR$_4_fetch.3.pre", %LHS_not_allocated_lab29 ], [ %"X_PTR$_4_fetch.2", %alloca_1 ]
  store float %X_fetch.1, float* %"X_PTR$_4_fetch.3", align 8, !tbaa !7
  %"X_PTR$_4_fetch.4" = load float, float* %"X_PTR$_4_fetch.3", align 1, !tbaa !5
  %mul = fmul reassoc ninf nsz arcp contract afn float %"X_PTR$_4_fetch.4", %"X_PTR$_4_fetch.4"
  %"(i8*)X_PTR$_4_fetch.8$" = bitcast float* %"X_PTR$_4_fetch.3" to i8*
  %func_result6 = tail call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)X_PTR$_4_fetch.8$", i32 262144) #0
  %rel8 = icmp eq i32 %func_result6, 0
  br i1 %rel8, label %dealloc.list.end37, label %dealloc.list.then36

dealloc.list.then36:                              ; preds = %LHS_allocated_lab28
  %func_result12 = tail call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)X_PTR$_4_fetch.8$", i32 262144) #0
  br label %dealloc.list.end37

dealloc.list.end37:                               ; preds = %dealloc.list.then36, %LHS_allocated_lab28
  ret float %mul
}

declare i32 @for_alloc_allocatable(i64, i8** nocapture, i32) local_unnamed_addr

declare i32 @for_alloc_allocatable_handle(i64, i8** nocapture, i32, i8*) local_unnamed_addr

declare i32 @for_dealloc_allocatable(i8* nocapture readonly, i32) local_unnamed_addr

define float @grad_square(float* noalias readonly dereferenceable(4) %X) local_unnamed_addr {
alloca_2:
  %"grad_square$DX$_5" = alloca float, align 8
  store float 0.000000e+00, float* %"grad_square$DX$_5", align 8, !tbaa !9
  call void @square__enzyme_autodiff_(float (...)* bitcast (float (float*)* @square to float (...)*), float* nonnull %X, float* nonnull %"grad_square$DX$_5") #0
  %"grad_square$DX$_5_fetch.12" = load float, float* %"grad_square$DX$_5", align 8, !tbaa !9
  ret float %"grad_square$DX$_5_fetch.12"
}

declare void @square__enzyme_autodiff_(float (...)* noalias, float* noalias readonly dereferenceable(4), float* noalias dereferenceable(4)) local_unnamed_addr

!0 = !{!1, !1, i64 0}
!1 = !{!"ifx$unique_sym$3", !2, i64 0}
!2 = !{!"Fortran Data Symbol", !3, i64 0}
!3 = !{!"Generic Fortran Symbol", !4, i64 0}
!4 = !{!"ifx$root$2$square"}
!5 = !{!6, !6, i64 0}
!6 = !{!"ifx$unique_sym$4", !2, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"ifx$unique_sym$5", !2, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"Fortran Data Symbol", !11, i64 0}
!11 = !{!"Generic Fortran Symbol", !12, i64 0}
!12 = !{!"ifx$root$3$grad_square"}

attributes #0 = { nounwind }

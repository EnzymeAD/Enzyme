; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=ad_mp_square_ -o /dev/null | FileCheck %s; fi

define float @ad_mp_square_(float* noalias nocapture readonly dereferenceable(4) %X) {
alloca_1:
  %"ad_mp_square_$X_PTR" = alloca float*, align 8
  store float* null, float** %"ad_mp_square_$X_PTR", align 8, !tbaa !0
  %"(i8**)ad_mp_square_$X_PTR$" = bitcast float** %"ad_mp_square_$X_PTR" to i8**
  %func_result = call i32 @for_alloc_allocatable(i64 4, i8** nonnull %"(i8**)ad_mp_square_$X_PTR$", i32 262144)
  %X_fetch.1 = load float, float* %X, align 1, !tbaa !5
  %"ad_mp_square_$X_PTR_fetch.2" = load float*, float** %"ad_mp_square_$X_PTR", align 8, !tbaa !0
  %rel.not = icmp eq float* %"ad_mp_square_$X_PTR_fetch.2", null
  br i1 %rel.not, label %LHS_not_allocated_lab8, label %LHS_allocated_lab7

LHS_not_allocated_lab8:                           ; preds = %alloca_1
  %func_result4 = call i32 @for_alloc_allocatable_handle(i64 4, i8** nonnull %"(i8**)ad_mp_square_$X_PTR$", i32 262144, i8* null)
  %"ad_mp_square_$X_PTR_fetch.3.pre" = load float*, float** %"ad_mp_square_$X_PTR", align 8, !tbaa !0
  br label %LHS_allocated_lab7

LHS_allocated_lab7:                               ; preds = %LHS_not_allocated_lab8, %alloca_1
  %"ad_mp_square_$X_PTR_fetch.3" = phi float* [ %"ad_mp_square_$X_PTR_fetch.3.pre", %LHS_not_allocated_lab8 ], [ %"ad_mp_square_$X_PTR_fetch.2", %alloca_1 ]
  store float %X_fetch.1, float* %"ad_mp_square_$X_PTR_fetch.3", align 8, !tbaa !7
  %"ad_mp_square_$X_PTR_fetch.3_fetch.1" = load float, float* %"ad_mp_square_$X_PTR_fetch.3", align 1, !tbaa !5
  %mul = fmul reassoc ninf nsz arcp contract afn float %"ad_mp_square_$X_PTR_fetch.3_fetch.1", %"ad_mp_square_$X_PTR_fetch.3_fetch.1"
  %"(i8*)ad_mp_square_$X_PTR_fetch.8$" = bitcast float* %"ad_mp_square_$X_PTR_fetch.3" to i8*
  %func_result6 = tail call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)ad_mp_square_$X_PTR_fetch.8$", i32 262144)
  %rel8 = icmp eq i32 %func_result6, 0
  br i1 %rel8, label %dealloc.list.end16, label %dealloc.list.then15

dealloc.list.then15:                              ; preds = %LHS_allocated_lab7
  %func_result12 = tail call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)ad_mp_square_$X_PTR_fetch.8$", i32 262144)
  br label %dealloc.list.end16

dealloc.list.end16:                               ; preds = %dealloc.list.then15, %LHS_allocated_lab7
  ret float %mul
}

declare i32 @for_alloc_allocatable(i64, i8** nocapture, i32) local_unnamed_addr
declare i32 @for_alloc_allocatable_handle(i64, i8** nocapture, i32, i8*) local_unnamed_addr
declare i32 @for_dealloc_allocatable(i8* nocapture readonly, i32) local_unnamed_addr

!0 = !{!1, !1, i64 0}
!1 = !{!"ifx$unique_sym$1", !2, i64 0}
!2 = !{!"Fortran Data Symbol", !3, i64 0}
!3 = !{!"Generic Fortran Symbol", !4, i64 0}
!4 = !{!"ifx$root$1$ad_mp_square_"}
!5 = !{!6, !6, i64 0}
!6 = !{!"ifx$unique_sym$2", !2, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"ifx$unique_sym$3", !2, i64 0}

; CHECK: ad_mp_square_ - {[-1]:Float@float} |{[-1]:Pointer, [-1,-1]:Float@float}:{}
; CHECK-NEXT: float* %X: {[-1]:Pointer, [-1,-1]:Float@float}
; CHECK-NEXT: alloca_1
; CHECK-NEXT:   %"ad_mp_square_$X_PTR" = alloca float*, align 8: {[-1]:Pointer, [-1,-1]:Pointer, [-1,-1,0]:Float@float}
; CHECK-NEXT:   store float* null, float** %"ad_mp_square_$X_PTR", align 8, !tbaa !0: {}
; CHECK-NEXT:   %"(i8**)ad_mp_square_$X_PTR$" = bitcast float** %"ad_mp_square_$X_PTR" to i8**: {[-1]:Pointer, [-1,-1]:Pointer, [-1,-1,0]:Float@float}
; CHECK-NEXT:   %func_result = call i32 @for_alloc_allocatable(i64 4, i8** nonnull %"(i8**)ad_mp_square_$X_PTR$", i32 262144): {[-1]:Integer}
; CHECK-NEXT:   %X_fetch.1 = load float, float* %X, align 1, !tbaa !5: {[-1]:Float@float}
; CHECK-NEXT:   %"ad_mp_square_$X_PTR_fetch.2" = load float*, float** %"ad_mp_square_$X_PTR", align 8, !tbaa !0: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %rel.not = icmp eq float* %"ad_mp_square_$X_PTR_fetch.2", null: {[-1]:Integer}
; CHECK-NEXT:   br i1 %rel.not, label %LHS_not_allocated_lab8, label %LHS_allocated_lab7: {}
; CHECK-NEXT: LHS_not_allocated_lab8
; CHECK-NEXT:   %func_result4 = call i32 @for_alloc_allocatable_handle(i64 4, i8** nonnull %"(i8**)ad_mp_square_$X_PTR$", i32 262144, i8* null): {[-1]:Integer}
; CHECK-NEXT:   %"ad_mp_square_$X_PTR_fetch.3.pre" = load float*, float** %"ad_mp_square_$X_PTR", align 8, !tbaa !0: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   br label %LHS_allocated_lab7: {}
; CHECK-NEXT: LHS_allocated_lab7
; CHECK-NEXT:   %"ad_mp_square_$X_PTR_fetch.3" = phi float* [ %"ad_mp_square_$X_PTR_fetch.3.pre", %LHS_not_allocated_lab8 ], [ %"ad_mp_square_$X_PTR_fetch.2", %alloca_1 ]: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   store float %X_fetch.1, float* %"ad_mp_square_$X_PTR_fetch.3", align 8, !tbaa !7: {}
; CHECK-NEXT:   %"ad_mp_square_$X_PTR_fetch.3_fetch.1" = load float, float* %"ad_mp_square_$X_PTR_fetch.3", align 1, !tbaa !5: {[-1]:Float@float}
; CHECK-NEXT:   %mul = fmul reassoc ninf nsz arcp contract afn float %"ad_mp_square_$X_PTR_fetch.3_fetch.1", %"ad_mp_square_$X_PTR_fetch.3_fetch.1": {[-1]:Float@float}
; CHECK-NEXT:   %"(i8*)ad_mp_square_$X_PTR_fetch.8$" = bitcast float* %"ad_mp_square_$X_PTR_fetch.3" to i8*: {[-1]:Pointer, [-1,0]:Float@float}
; CHECK-NEXT:   %func_result6 = tail call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)ad_mp_square_$X_PTR_fetch.8$", i32 262144): {[-1]:Integer}
; CHECK-NEXT:   %rel8 = icmp eq i32 %func_result6, 0: {[-1]:Integer}
; CHECK-NEXT:   br i1 %rel8, label %dealloc.list.end16, label %dealloc.list.then15: {}
; CHECK-NEXT: dealloc.list.then15
; CHECK-NEXT:   %func_result12 = tail call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)ad_mp_square_$X_PTR_fetch.8$", i32 262144): {[-1]:Integer}
; CHECK-NEXT:   br label %dealloc.list.end16: {}
; CHECK-NEXT: dealloc.list.end16
; CHECK-NEXT:   ret float %mul: {}
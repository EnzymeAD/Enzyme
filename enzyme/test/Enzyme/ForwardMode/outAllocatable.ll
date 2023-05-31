; RUN: if [ %llvmver -ge 13 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -sroa -simplifycfg -instcombine -adce -S | FileCheck %s; fi

define void @square(float* noalias nocapture readonly dereferenceable(4) %X, float** noalias nocapture dereferenceable(4) %Y) {
alloca_1:
  %Y_fetch.11 = load float*, float** %Y, align 1, !tbaa !0
  %rel1.not = icmp eq float* %Y_fetch.11, null
  br i1 %rel1.not, label %bb8_endif, label %bb_new51_uif_true

bb8_endif:                                        ; preds = %bb_new51_uif_true, %alloca_1
  %"(i8**)Y$" = bitcast float** %Y to i8**
  %func_result3 = tail call i32 @for_alloc_allocatable(i64 4, i8** nonnull %"(i8**)Y$", i32 262144)
  %X_fetch.12 = load float, float* %X, align 1, !tbaa !6
  %Y_fetch.14 = load float*, float** %Y, align 1, !tbaa !0
  %rel8.not = icmp eq float* %Y_fetch.14, null
  br i1 %rel8.not, label %LHS_not_allocated_lab60, label %LHS_allocated_lab59

bb_new51_uif_true:                                ; preds = %alloca_1
  %"(i8*)Y_fetch.11$" = bitcast float* %Y_fetch.11 to i8*
  %func_result = tail call i32 @for_deallocate(i8* nonnull %"(i8*)Y_fetch.11$", i32 262144)
  store float* null, float** %Y, align 1, !tbaa !0
  br label %bb8_endif

LHS_not_allocated_lab60:                          ; preds = %bb8_endif
  %func_result7 = tail call i32 @for_alloc_allocatable_handle(i64 4, i8** nonnull %"(i8**)Y$", i32 262144, i8* null)
  %Y_fetch.15.pre = load float*, float** %Y, align 1, !tbaa !0
  br label %LHS_allocated_lab59

LHS_allocated_lab59:                              ; preds = %LHS_not_allocated_lab60, %bb8_endif
  %Y_fetch.15 = phi float* [ %Y_fetch.15.pre, %LHS_not_allocated_lab60 ], [ %Y_fetch.14, %bb8_endif ]
  %mul = fmul reassoc ninf nsz arcp contract afn float %X_fetch.12, %X_fetch.12
  store float %mul, float* %Y_fetch.15, align 1, !tbaa !8
  ret void
}

declare i32 @for_alloc_allocatable(i64, i8** nocapture, i32) local_unnamed_addr
declare i32 @for_alloc_allocatable_handle(i64, i8** nocapture, i32, i8*) local_unnamed_addr
declare i32 @for_deallocate(i8* nocapture readonly, i32) local_unnamed_addr

define void @grad_square(float* noalias readonly dereferenceable(4) %X, float* noalias dereferenceable(4) %DX, float** noalias dereferenceable(4) %Y, float** noalias dereferenceable(4) %DY) local_unnamed_addr {
alloca_3:
  tail call void @fort__enzyme_fwddiff_(void (...)* bitcast (void (float*, float**)* @square to void (...)*), float* nonnull %X, float* nonnull %DX, float** nonnull %Y, float** nonnull %DY)
  ret void
}

declare void @fort__enzyme_fwddiff_(void (...)* noalias, float* noalias readonly dereferenceable(4), float* noalias dereferenceable(4), float** noalias dereferenceable(4), float** noalias dereferenceable(4)) local_unnamed_addr

!0 = !{!1, !1, i64 0}
!1 = !{!"ifx$unique_sym$11", !2, i64 0}
!2 = !{!"Fortran Data Symbol", !3, i64 0}
!3 = !{!"Generic Fortran Symbol", !4, i64 0}
!4 = !{!"ifx$root$2$square"}
!5 = !{i64 144}
!6 = !{!7, !7, i64 0}
!7 = !{!"ifx$unique_sym$12", !2, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"ifx$unique_sym$13", !2, i64 0}


; CHECK: define internal void @fwddiffesquare(float* noalias nocapture readonly dereferenceable(4) %X, float* nocapture %"X'", float** noalias nocapture dereferenceable(4) %Y, float** nocapture %"Y'") #0 {
; CHECK-NEXT: alloca_1:
; CHECK-NEXT:   %Y_fetch.11 = load float*, float** %Y, align 1, !tbaa !0, !alias.scope !9, !noalias !12
; CHECK-NEXT:   %rel1.not = icmp eq float* %Y_fetch.11, null
; CHECK-NEXT:   br i1 %rel1.not, label %bb8_endif, label %bb_new51_uif_true

; CHECK: bb8_endif:                                        ; preds = %__enzyme_checked_free_1.exit, %alloca_1
; CHECK-NEXT:   %"(i8**)Y$'ipc" = bitcast float** %"Y'" to i8**
; CHECK-NEXT:   %"(i8**)Y$" = bitcast float** %Y to i8**
; CHECK-NEXT:   %0 = call i32 @for_alloc_allocatable(i64 4, i8** %"(i8**)Y$'ipc", i32 262144)
; CHECK-NEXT:   %1 = bitcast float** %"Y'" to i32**
; CHECK-NEXT:   %2 = load i32*, i32** %1, align 8
; CHECK-NEXT:   store i32 0, i32* %2, align 1
; CHECK-NEXT:   %func_result3 = tail call i32 @for_alloc_allocatable(i64 4, i8** nonnull %"(i8**)Y$", i32 262144) #0
; CHECK-NEXT:   %"X_fetch.12'ipl" = load float, float* %"X'", align 1, !tbaa !5, !alias.scope !14, !noalias !17
; CHECK-NEXT:   %X_fetch.12 = load float, float* %X, align 1, !tbaa !5, !alias.scope !17, !noalias !14
; CHECK-NEXT:   %Y_fetch.14 = load float*, float** %Y, align 1, !tbaa !0, !alias.scope !9, !noalias !12
; CHECK-NEXT:   %rel8.not = icmp eq float* %Y_fetch.14, null
; CHECK-NEXT:   br i1 %rel8.not, label %LHS_not_allocated_lab60, label %LHS_allocated_lab59

; CHECK: bb_new51_uif_true:                                ; preds = %alloca_1
; CHECK-NEXT:   %3 = bitcast float** %"Y'" to i8**
; CHECK-NEXT:   %"Y_fetch.11'ipl1" = load i8*, i8** %3, align 1, !tbaa !0, !alias.scope !12, !noalias !9
; CHECK-NEXT:   %"(i8*)Y_fetch.11$" = bitcast float* %Y_fetch.11 to i8*
; CHECK-NEXT:   %func_result = tail call i32 @for_deallocate(i8* nonnull %"(i8*)Y_fetch.11$", i32 262144) #0
; CHECK-NEXT:   %.not = icmp eq i8* %"Y_fetch.11'ipl1", %"(i8*)Y_fetch.11$"
; CHECK-NEXT:   br i1 %.not, label %__enzyme_checked_free_1.exit, label %free0.i

; CHECK: free0.i:                                          ; preds = %bb_new51_uif_true
; CHECK-NEXT:   %4 = call i32 @for_deallocate(i8* nonnull %"Y_fetch.11'ipl1", i32 262144) #3
; CHECK-NEXT:   br label %__enzyme_checked_free_1.exit

; CHECK: __enzyme_checked_free_1.exit:                     ; preds = %bb_new51_uif_true, %free0.i
; CHECK-NEXT:   store float* null, float** %"Y'", align 1, !tbaa !0, !alias.scope !12, !noalias !9
; CHECK-NEXT:   store float* null, float** %Y, align 1, !tbaa !0, !alias.scope !9, !noalias !12
; CHECK-NEXT:   br label %bb8_endif

; CHECK: LHS_not_allocated_lab60:                          ; preds = %bb8_endif
; CHECK-NEXT:   %5 = call i32 @for_alloc_allocatable_handle(i64 4, i8** %"(i8**)Y$'ipc", i32 262144, i8* null)
; CHECK-NEXT:   %6 = bitcast float** %"Y'" to i32**
; CHECK-NEXT:   %7 = load i32*, i32** %6, align 8
; CHECK-NEXT:   store i32 0, i32* %7, align 1
; CHECK-NEXT:   %func_result7 = tail call i32 @for_alloc_allocatable_handle(i64 4, i8** nonnull %"(i8**)Y$", i32 262144, i8* null) #0
; CHECK-NEXT:   %Y_fetch.15.pre = load float*, float** %Y, align 1, !tbaa !0, !alias.scope !9, !noalias !12
; CHECK-NEXT:   br label %LHS_allocated_lab59

; CHECK: LHS_allocated_lab59:                              ; preds = %LHS_not_allocated_lab60, %bb8_endif
; CHECK-NEXT:   %Y_fetch.15 = phi float* [ %Y_fetch.15.pre, %LHS_not_allocated_lab60 ], [ %Y_fetch.14, %bb8_endif ]
; CHECK-NEXT:   %8 = load float*, float** %"Y'", align 1, !tbaa !0, !alias.scope !12, !noalias !9
; CHECK-NEXT:   %mul = fmul reassoc ninf nsz arcp contract afn float %X_fetch.12, %X_fetch.12
; CHECK-NEXT:   %9 = fadd fast float %"X_fetch.12'ipl", %"X_fetch.12'ipl"
; CHECK-NEXT:   %10 = fmul fast float %9, %X_fetch.12
; CHECK-NEXT:   store float %10, float* %8, align 1, !tbaa !7, !alias.scope !19, !noalias !22
; CHECK-NEXT:   store float %mul, float* %Y_fetch.15, align 1, !tbaa !7, !alias.scope !22, !noalias !19
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: ; Function Attrs: argmemonly nocallback nofree nounwind willreturn writeonly
; CHECK-NEXT: declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1

; CHECK: ; Function Attrs: alwaysinline argmemonly nounwind
; CHECK-NEXT: define internal void @__enzyme_checked_free_1(i8* nocapture %0, i8* nocapture %1, i32 %2) #2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.not = icmp eq i8* %0, %1
; CHECK-NEXT:   br i1 %.not, label %end, label %free0

; CHECK: free0:                                            ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @for_deallocate(i8* nonnull %1, i32 %2) #3
; CHECK-NEXT:   br label %end

; CHECK: end:                                              ; preds = %free0, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
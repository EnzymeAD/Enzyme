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
  store float 1.000000e+00, float* %"grad_square$DX$_5", align 8, !tbaa !9
  %func_result = call reassoc ninf nsz arcp contract afn float @square__enzyme_fwddiff_(float (...)* bitcast (float (float*)* @square to float (...)*), float* nonnull %X, float* nonnull %"grad_square$DX$_5") #0
  ret float %func_result
}

declare float @square__enzyme_fwddiff_(float (...)* noalias, float* noalias readonly dereferenceable(4), float* noalias dereferenceable(4)) local_unnamed_addr

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

; CHECK: define internal float @fwddiffesquare(float* noalias nocapture readonly dereferenceable(4) %X, float* nocapture %"X'") #0 {
; CHECK-NEXT: alloca_1:
; CHECK-NEXT:   %"X_PTR$_4'ipa" = alloca float*, align 8
; CHECK-NEXT:   %"X_PTR$_4" = alloca float*, align 8
; CHECK-NEXT:   store float* null, float** %"X_PTR$_4'ipa", align 8, !tbaa !0, !alias.scope !13, !noalias !16
; CHECK-NEXT:   store float* null, float** %"X_PTR$_4", align 8, !tbaa !0, !alias.scope !16, !noalias !13
; CHECK-NEXT:   %"(i8**)X_PTR$_4$'ipc" = bitcast float** %"X_PTR$_4'ipa" to i8**
; CHECK-NEXT:   %"(i8**)X_PTR$_4$" = bitcast float** %"X_PTR$_4" to i8**
; CHECK-NEXT:   %0 = call i32 @for_alloc_allocatable(i64 4, i8** nonnull %"(i8**)X_PTR$_4$'ipc", i32 262144)
; CHECK-NEXT:   %1 = bitcast float** %"X_PTR$_4'ipa" to i32**
; CHECK-NEXT:   %2 = load i32*, i32** %1, align 8
; CHECK-NEXT:   store i32 0, i32* %2, align 1
; CHECK-NEXT:   %func_result = call i32 @for_alloc_allocatable(i64 4, i8** nonnull %"(i8**)X_PTR$_4$", i32 262144) #4
; CHECK-NEXT:   %"X_fetch.1'ipl" = load float, float* %"X'", align 1, !tbaa !5, !alias.scope !18, !noalias !21
; CHECK-NEXT:   %X_fetch.1 = load float, float* %X, align 1, !tbaa !5, !alias.scope !21, !noalias !18
; CHECK-NEXT:   %"X_PTR$_4_fetch.2" = load float*, float** %"X_PTR$_4", align 8, !tbaa !0, !alias.scope !16, !noalias !13
; CHECK-NEXT:   %rel.not = icmp eq float* %"X_PTR$_4_fetch.2", null
; CHECK-NEXT:   br i1 %rel.not, label %LHS_not_allocated_lab29, label %LHS_allocated_lab28

; CHECK: LHS_not_allocated_lab29:                          ; preds = %alloca_1
; CHECK-NEXT:   %3 = call i32 @for_alloc_allocatable_handle(i64 4, i8** nonnull %"(i8**)X_PTR$_4$'ipc", i32 262144, i8* null)
; CHECK-NEXT:   %4 = bitcast float** %"X_PTR$_4'ipa" to i32**
; CHECK-NEXT:   %5 = load i32*, i32** %4, align 8
; CHECK-NEXT:   store i32 0, i32* %5, align 1
; CHECK-NEXT:   %func_result4 = call i32 @for_alloc_allocatable_handle(i64 4, i8** nonnull %"(i8**)X_PTR$_4$", i32 262144, i8* null) #4
; CHECK-NEXT:   %"X_PTR$_4_fetch.3.pre" = load float*, float** %"X_PTR$_4", align 8, !tbaa !0, !alias.scope !16, !noalias !13
; CHECK-NEXT:   br label %LHS_allocated_lab28

; CHECK: LHS_allocated_lab28:                              ; preds = %LHS_not_allocated_lab29, %alloca_1
; CHECK-NEXT:   %.in = phi i32* [ %5, %LHS_not_allocated_lab29 ], [ %2, %alloca_1 ]
; CHECK-NEXT:   %"X_PTR$_4_fetch.3" = phi float* [ %"X_PTR$_4_fetch.3.pre", %LHS_not_allocated_lab29 ], [ %"X_PTR$_4_fetch.2", %alloca_1 ]
; CHECK-NEXT:   %6 = bitcast i32* %.in to float*
; CHECK-NEXT:   store float %"X_fetch.1'ipl", float* %6, align 8, !tbaa !7, !alias.scope !23, !noalias !26
; CHECK-NEXT:   store float %X_fetch.1, float* %"X_PTR$_4_fetch.3", align 8, !tbaa !7, !alias.scope !26, !noalias !23
; CHECK-NEXT:   %"X_PTR$_4_fetch.4'ipl" = load float, float* %6, align 1, !tbaa !5, !alias.scope !23, !noalias !26
; CHECK-NEXT:   %"(i8*)X_PTR$_4_fetch.8$'ipc" = bitcast i32* %.in to i8*
; CHECK-NEXT:   %"(i8*)X_PTR$_4_fetch.8$" = bitcast float* %"X_PTR$_4_fetch.3" to i8*
; CHECK-NEXT:   %func_result6 = tail call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)X_PTR$_4_fetch.8$", i32 262144) #4
; CHECK-NEXT:   %.not = icmp eq float* %"X_PTR$_4_fetch.3", %6
; CHECK-NEXT:   br i1 %.not, label %__enzyme_checked_free_1.exit, label %free0.i

; CHECK: free0.i:                                          ; preds = %LHS_allocated_lab28
; CHECK-NEXT:   %7 = call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)X_PTR$_4_fetch.8$'ipc", i32 262144) #4
; CHECK-NEXT:   br label %__enzyme_checked_free_1.exit

; CHECK: __enzyme_checked_free_1.exit:                     ; preds = %LHS_allocated_lab28, %free0.i
; CHECK-NEXT:   %rel8 = icmp eq i32 %func_result6, 0
; CHECK-NEXT:   br i1 %rel8, label %dealloc.list.end37, label %dealloc.list.then36

; CHECK: dealloc.list.then36:                              ; preds = %__enzyme_checked_free_1.exit
; CHECK-NEXT:   %func_result12 = tail call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)X_PTR$_4_fetch.8$", i32 262144) #4
; CHECK-NEXT:   %.not3 = icmp eq float* %"X_PTR$_4_fetch.3", %6
; CHECK-NEXT:   br i1 %.not3, label %dealloc.list.end37, label %free0.i1

; CHECK: free0.i1:                                         ; preds = %dealloc.list.then36
; CHECK-NEXT:   %8 = call i32 @for_dealloc_allocatable(i8* nonnull %"(i8*)X_PTR$_4_fetch.8$'ipc", i32 262144) #4
; CHECK-NEXT:   br label %dealloc.list.end37

; CHECK: dealloc.list.end37:                               ; preds = %free0.i1, %dealloc.list.then36, %__enzyme_checked_free_1.exit
; CHECK-NEXT:   %9 = fadd fast float %"X_PTR$_4_fetch.4'ipl", %"X_PTR$_4_fetch.4'ipl"
; CHECK-NEXT:   %10 = fmul fast float %9, %X_fetch.1
; CHECK-NEXT:   ret float %10
; CHECK-NEXT: }

; CHECK: define internal void @__enzyme_checked_free_1(i8* nocapture %0, i8* nocapture %1, i32 %2) #2 {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %.not = icmp eq i8* %0, %1
; CHECK-NEXT:   br i1 %.not, label %end, label %free0

; CHECK: free0:                                            ; preds = %entry
; CHECK-NEXT:   %3 = call i32 @for_dealloc_allocatable(i8* nonnull %1, i32 %2) #4
; CHECK-NEXT:   br label %end

; CHECK: end:                                              ; preds = %free0, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
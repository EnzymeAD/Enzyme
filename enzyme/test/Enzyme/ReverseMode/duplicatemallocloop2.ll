; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -early-cse -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,early-cse,adce)" -S | FileCheck %s

define dso_local double @f(double* nocapture readonly %a0) local_unnamed_addr #0 {
entry:
  %a2 = load double, double* %a0, align 8
  %m2 = fmul double %a2, %a2
  ret double %m2
}

define dso_local void @malloced(double* noalias nocapture %a0, double* noalias nocapture readonly %a1, i32 %a2) #1 {
entry:
  %a4 = shl i32 %a2, 3
  br label %loop

loop:  
  %a9 = phi i32 [ 0, %entry ], [ %a14, %loop ]
  %a5 = call noalias i8* @malloc(i32 8) #5
  %a6 = bitcast i8* %a5 to double*
  %a10 = getelementptr inbounds double, double* %a1, i32 %a9
  %a11 = load double, double* %a10, align 8
  store double %a11, double* %a6, align 8
  %a12 = call double @f(double* nonnull %a6)
  %a13 = getelementptr inbounds double, double* %a0, i32 %a9
  store double %a12, double* %a13, align 8
  %a14 = add nuw nsw i32 %a9, 1
  %a15 = icmp eq i32 %a14, 10
  call void @free(i8* %a5)
  br i1 %a15, label %exit, label %loop

exit:                                                ; preds = %8
  ret void
}

declare dso_local noalias i8* @malloc(i32) local_unnamed_addr #2

declare dso_local void @free(i8* nocapture) local_unnamed_addr #3

define dso_local void @derivative(double* %a0, double* %a1, double* %a2, double* %a3, i32 %a4) local_unnamed_addr #1 {
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (void (double*, double*, i32)* @malloced to i8*), double* %a0, double* %a1, double* %a2, double* %a3, i32 %a4) #6
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...) local_unnamed_addr #4

attributes #0 = { noinline norecurse nounwind readonly }
attributes #1 = { nounwind }
attributes #2 = { inaccessiblememonly nounwind }
attributes #3 = { inaccessiblemem_or_argmemonly nounwind }
attributes #6 = { nounwind }


; CHECK: define internal void @diffemalloced(double* noalias nocapture %a0, double* nocapture %"a0'", double* noalias nocapture readonly %a1, double* nocapture %"a1'", i32 %a2)
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %0 = trunc i64 %iv to i32
; CHECK-NEXT:   %a5 = call noalias i8* @malloc(i32 8)
; CHECK-NEXT:   %a6 = bitcast i8* %a5 to double*
; CHECK-NEXT:   %a10 = getelementptr inbounds double, double* %a1, i32 %0
; CHECK-NEXT:   %a11 = load double, double* %a10, align 8
; CHECK-NEXT:   store double %a11, double* %a6, align 8
; CHECK-NEXT:   %a12 = call fast double @augmented_f(double* %a6, double* undef)
; CHECK-NEXT:   %a13 = getelementptr inbounds double, double* %a0, i32 %0
; CHECK-NEXT:   store double %a12, double* %a13, align 8
; CHECK-NEXT:   %a14 = add nuw nsw i32 %0, 1
; CHECK-NEXT:   %a15 = icmp eq i32 %a14, 10
; CHECK-NEXT:   call void @free(i8* %a5)
; CHECK-NEXT:   br i1 %a15, label %remat_enter, label %loop

; CHECK: invertentry:                                      ; preds = %remat_enter
; CHECK-NEXT:   ret void

; CHECK: incinvertloop:                                    ; preds = %remat_enter
; CHECK-NEXT:   %1 = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: remat_enter:                                      ; preds = %loop, %incinvertloop
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %1, %incinvertloop ], [ 9, %loop ]
; CHECK-NEXT:   %remat_a5 = call noalias i8* @malloc(i32 8) 
; CHECK-NEXT:   %"a5'mi" = call noalias nonnull i8* @malloc(i32 8) 
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"a5'mi", i8 0, i64 8, i1 false)
; CHECK-DAG:   %[[a6_unwrap1:.+]] = bitcast i8* %remat_a5 to double*
; CHECK-DAG:   %[[_unwrap2:.+]] = trunc i64 %"iv'ac.0" to i32
; CHECK-DAG:   %[[a10_unwrap:.+]] = getelementptr inbounds double, double* %a1, i32 %[[_unwrap2]]
; CHECK-DAG:   %[[a11_unwrap:.+]] = load double, double* %[[a10_unwrap]], align 8
; CHECK-NEXT:   store double %[[a11_unwrap]], double* %[[a6_unwrap1]], align 8
; CHECK-NEXT:   %"a13'ipg_unwrap" = getelementptr inbounds double, double* %"a0'", i32 %[[_unwrap2]]
; CHECK-NEXT:   %2 = load double, double* %"a13'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a13'ipg_unwrap", align 8
; CHECK-NEXT:   %"a6'ipc_unwrap" = bitcast i8* %"a5'mi" to double*
; CHECK-NEXT:   call void @diffef(double* %[[a6_unwrap1]], double* %"a6'ipc_unwrap", double %2)
; CHECK-NEXT:   %3 = load double, double* %"a6'ipc_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a6'ipc_unwrap", align 8
; CHECK-NEXT:   %"a10'ipg_unwrap" = getelementptr inbounds double, double* %"a1'", i32 %[[_unwrap2]]
; CHECK-NEXT:   %4 = load double, double* %"a10'ipg_unwrap", align 8
; CHECK-NEXT:   %5 = fadd fast double %4, %3
; CHECK-NEXT:   store double %5, double* %"a10'ipg_unwrap", align 8
; CHECK-NEXT:   call void @free(i8* nonnull %"a5'mi")
; CHECK-NEXT:   call void @free(i8* %remat_a5)
; CHECK-NEXT:   %6 = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %6, label %invertentry, label %incinvertloop
; CHECK-NEXT: }

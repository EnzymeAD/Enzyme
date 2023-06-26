; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

@.str = private unnamed_addr constant [28 x i8] c"original =%f derivative=%f\0A\00", align 1

define dso_local double* @f(double** nocapture readonly %a0) local_unnamed_addr #0 {
entry:
  %a2 = load double*, double** %a0, align 8
  ret double* %a2
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define dso_local double @submalloced(double* noalias %out, double* noalias %a0) local_unnamed_addr #1 {
entry: 
  %p2 = call noalias i8* @malloc(i32 8)
  br label %loop

loop:  
  %a9 = phi i32 [ 0, %entry ], [ %a14, %loop ]
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %p2)
  %p3 = bitcast i8* %p2 to double**
  %a10 = getelementptr inbounds double, double* %a0, i32 %a9
  store double* %a10, double** %p3, align 8
  %a4 = call double* @f(double** nonnull %p3)
  %r = load double, double* %a4
  %m2 = fmul double %r, %r
  %a13 = getelementptr inbounds double, double* %out, i32 %a9
  store double %m2, double* %a13, align 8
  %a14 = add nuw nsw i32 %a9, 1
  %a15 = icmp eq i32 %a14, 10
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %p2)
  br i1 %a15, label %exit, label %loop

exit:                                                ; preds = %8
  call void @free(i8* %p2)
  ret double 0.000000e+00
}

declare dso_local noalias i8* @malloc(i32) local_unnamed_addr #2

declare dso_local void @free(i8* nocapture) local_unnamed_addr #3

define dso_local double @malloced(double* %o, double* %a0) #1 {
entry:
  %a3 = call double @submalloced(double* %o, double* %a0)
  %a4 = fmul double %a3, %a3
  ret double %a4
}

define dso_local void @derivative(double* %o, double* %do, double* %a0, double* %a1) local_unnamed_addr #4 {
entry:
  call void (i8*, ...) @__enzyme_autodiff(i8* bitcast (double (double*, double*)* @malloced to i8*), double* %o, double* %do, double* %a0, double* %a1)
  ret void
}

declare dso_local void @__enzyme_autodiff(i8*, ...)

attributes #0 = { noinline norecurse nounwind readonly }
attributes #1 = { nounwind }
attributes #2 = { inaccessiblememonly nounwind }
attributes #3 = { inaccessiblemem_or_argmemonly nounwind }
attributes #4 = { nounwind }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind }
attributes #9 = { nounwind }


; CHECK: define internal { { double**, double* }, double } @augmented_submalloced(double* noalias %out, double* %"out'", double* noalias %a0, double* %"a0'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { { double**, double* }, double }
; CHECK-NEXT:   %1 = getelementptr inbounds { { double**, double* }, double }, { { double**, double* }, double }* %0, i32 0, i32 0
; CHECK-NEXT:   %p2 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   %"p2'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"p2'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   %malloccall = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* bitcast (i8* (i32)* @malloc to i8* (i64)*)(i64 80)
; CHECK-NEXT:   %"a4'ac_malloccache" = bitcast i8* %malloccall to double**
; CHECK-NEXT:   %2 = getelementptr inbounds { double**, double* }, { double**, double* }* %1, i32 0, i32 0
; CHECK-NEXT:   store double** %"a4'ac_malloccache", double*** %2
; CHECK-NEXT:   %[[malloccall4:.+]] = tail call noalias nonnull dereferenceable(80) dereferenceable_or_null(80) i8* bitcast (i8* (i32)* @malloc to i8* (i64)*)(i64 80)
; CHECK-NEXT:   %r_malloccache = bitcast i8* %[[malloccall4]] to double*
; CHECK-NEXT:   %3 = getelementptr inbounds { double**, double* }, { double**, double* }* %1, i32 0, i32 1
; CHECK-NEXT:   store double* %r_malloccache, double** %3
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %4 = trunc i64 %iv to i32
; CHECK-NEXT:   %"p3'ipc" = bitcast i8* %"p2'mi" to double**
; CHECK-NEXT:   %p3 = bitcast i8* %p2 to double**
; CHECK-NEXT:   %"a10'ipg" = getelementptr inbounds double, double* %"a0'", i32 %4
; CHECK-NEXT:   %a10 = getelementptr inbounds double, double* %a0, i32 %4
; CHECK-NEXT:   store double* %"a10'ipg", double** %"p3'ipc", align 8
; CHECK-NEXT:   store double* %a10, double** %p3, align 8
; CHECK-NEXT:   %a4_augmented = call { double*, double* } @augmented_f(double** %p3, double** %"p3'ipc")
; CHECK-NEXT:   %a4 = extractvalue { double*, double* } %a4_augmented, 0
; CHECK-NEXT:   %"a4'ac" = extractvalue { double*, double* } %a4_augmented, 1
; CHECK-NEXT:   %r = load double, double* %a4
; CHECK-NEXT:   %m2 = fmul double %r, %r
; CHECK-NEXT:   %a13 = getelementptr inbounds double, double* %out, i32 %4
; CHECK-NEXT:   store double %m2, double* %a13, align 8
; CHECK-NEXT:   %5 = getelementptr inbounds double, double* %r_malloccache, i64 %iv
; CHECK-NEXT:   store double %r, double* %5, align 8
; CHECK-NEXT:   %6 = getelementptr inbounds double*, double** %"a4'ac_malloccache", i64 %iv
; CHECK-NEXT:   store double* %"a4'ac", double** %6, align 8
; CHECK-NEXT:   %a14 = add nuw nsw i32 %4, 1
; CHECK-NEXT:   %a15 = icmp eq i32 %a14, 10
; CHECK-NEXT:   br i1 %a15, label %exit, label %loop

; CHECK: exit:                                             ; preds = %loop
; CHECK-NEXT:   call void @free(i8* %p2)
; CHECK-NEXT:   call void @free(i8* %"p2'mi")
; CHECK-NEXT:   %7 = getelementptr inbounds { { double**, double* }, double }, { { double**, double* }, double }* %0, i32 0, i32 1
; CHECK-NEXT:   store double 0.000000e+00, double* %7
; CHECK-NEXT:   %8 = load { { double**, double* }, double }, { { double**, double* }, double }* %0
; CHECK-NEXT:   ret { { double**, double* }, double } %8
; CHECK-NEXT: }

; CHECK: define internal void @diffesubmalloced(double* noalias %out, double* %"out'", double* noalias %a0, double* %"a0'", double %differeturn, { double**, double* } %tapeArg) 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue { double**, double* } %tapeArg, 0
; CHECK-NEXT:   %1 = extractvalue { double**, double* } %tapeArg, 1
; CHECK-NEXT:   %p2 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   %"p2'mi" = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i32 8)
; CHECK-NEXT:   call void @llvm.memset.p0i8.i64(i8* nonnull dereferenceable(8) dereferenceable_or_null(8) %"p2'mi", i8 0, i64 8, i1 false)
; CHECK-NEXT:   br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK-NEXT:   %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
; CHECK-NEXT:   %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:   %2 = trunc i64 %iv to i32
; CHECK-NEXT:   %"p3'ipc" = bitcast i8* %"p2'mi" to double**
; CHECK-NEXT:   %"a10'ipg" = getelementptr inbounds double, double* %"a0'", i32 %2
; REMOVED:   %a10 = getelementptr inbounds double, double* %a0, i32 %2
; CHECK-NEXT:   store double* %"a10'ipg", double** %"p3'ipc", align 8
; REMOVED:   store double* %a10, double** %p3, align 8
; CHECK-NEXT:   %a14 = add nuw nsw i32 %2, 1
; CHECK-NEXT:   %a15 = icmp eq i32 %a14, 10
; CHECK-NEXT:   br i1 %a15, label %remat_enter, label %loop

; CHECK: invertentry:  
; CHECK-NEXT:   call void @free(i8* nonnull %"p2'mi")
; CHECK-NEXT:   call void @free(i8* nonnull %p2)
; CHECK-NEXT:   %3 = bitcast double** %0 to i8*
; CHECK-NEXT:   call void @free(i8* nonnull %3)
; CHECK-NEXT:   %4 = bitcast double* %1 to i8*
; CHECK-NEXT:   call void @free(i8* nonnull %4)
; CHECK-NEXT:   ret void

; CHECK: incinvertloop: 
; CHECK-NEXT:   %[[i16:.+]] = add nsw i64 %"iv'ac.0", -1
; CHECK-NEXT:   br label %remat_enter

; CHECK: remat_enter: 
; CHECK-NEXT:   %"iv'ac.0" = phi i64 [ %[[i16]], %incinvertloop ], [ 9, %loop ]
; CHECK-DAG:   %[[p3_unwrap9:.+]] = bitcast i8* %p2 to double**
; CHECK-DAG:   %[[_unwrap10:.+]] = trunc i64 %"iv'ac.0" to i32
; CHECK-DAG:   %[[a10_unwrap:.+]] = getelementptr inbounds double, double* %a0, i32 %[[_unwrap10]]
; CHECK-NEXT:   store double* %[[a10_unwrap]], double** %[[p3_unwrap9]], align 8
; CHECK-NEXT:   %"a10'ipg_unwrap" = getelementptr inbounds double, double* %"a0'", i32 %[[_unwrap10]]
; CHECK-NEXT:   %[[p3ipc_unwrap11:.+]] = bitcast i8* %"p2'mi" to double**
; CHECK-NEXT:   store double* %"a10'ipg_unwrap", double** %[[p3ipc_unwrap11]]
; CHECK-NEXT:   %_unwrap = trunc i64 %"iv'ac.0" to i32
; CHECK-NEXT:   %"a13'ipg_unwrap" = getelementptr inbounds double, double* %"out'", i32 %_unwrap
; CHECK-NEXT:   %[[i5:.+]] = load double, double* %"a13'ipg_unwrap", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a13'ipg_unwrap", align 8
; CHECK-NEXT:   %[[i6:.+]] = extractvalue { double**, double* } %tapeArg, 1
; CHECK-NEXT:   %[[i7:.+]] = getelementptr inbounds double, double* %[[i6]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[i8:.+]] = load double, double* %[[i7]], align 8
; CHECK-NEXT:   %[[m0differ:.+]] = fmul fast double %[[i5]], %[[i8]]
; CHECK-NEXT:   %[[m1differ:.+]] = fmul fast double %[[i5]], %[[i8]]
; CHECK-NEXT:   %[[i9:.+]] = fadd fast double %[[m0differ]], %[[m1differ]]
; CHECK-NEXT:   %[[i10:.+]] = extractvalue { double**, double* } %tapeArg, 0
; CHECK-NEXT:   %[[i11:.+]] = getelementptr inbounds double*, double** %[[i10]], i64 %"iv'ac.0"
; CHECK-NEXT:   %[[i12:.+]] = load double*, double** %[[i11]], align 8
; CHECK-NEXT:   %[[i13:.+]] = load double, double* %[[i12]]
; CHECK-NEXT:   %[[i14:.+]] = fadd fast double %[[i13]], %[[i9]]
; CHECK-NEXT:   store double %[[i14]], double* %[[i12]]
; CHECK-NEXT:   %p3_unwrap = bitcast i8* %p2 to double**
; CHECK-NEXT:   %"p3'ipc_unwrap" = bitcast i8* %"p2'mi" to double**
; CHECK-NEXT:   call void @diffef(double** %p3_unwrap, double** %"p3'ipc_unwrap")
; CHECK-NEXT:   %[[i15:.+]] = icmp eq i64 %"iv'ac.0", 0
; CHECK-NEXT:   br i1 %[[i15]], label %invertentry, label %incinvertloop

; CHECK-NEXT: }

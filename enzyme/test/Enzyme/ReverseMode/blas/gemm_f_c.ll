;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -S | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly) 

define void @f(i8* %C, i8* %A, i8* %B) {
entry:
  %transa = alloca i8, align 1
  %transb = alloca i8, align 1
  %m = alloca i64, align 16
  %m_p = bitcast i64* %m to i8*
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %k = alloca i64, align 16
  %k_p = bitcast i64* %k to i8*
  %alpha = alloca double, align 16
  %alpha_p = bitcast double* %alpha to i8*
  %lda = alloca i64, align 16
  %lda_p = bitcast i64* %lda to i8*
  %ldb = alloca i64, align 16
  %ldb_p = bitcast i64* %ldb to i8*
  %beta = alloca double, align 16
  %beta_p = bitcast double* %beta to i8*
  %ldc = alloca i64, align 16
  %ldc_p = bitcast i64* %ldc to i8*
  store i8 78, i8* %transa, align 1
  store i8 78, i8* %transb, align 1
  store i64 4, i64* %m, align 16
  store i64 4, i64* %n, align 16
  store i64 8, i64* %k, align 16
  store double 1.000000e+00, double* %alpha, align 16
  store i64 4, i64* %lda, align 16
  store i64 8, i64* %ldb, align 16
  store double 0.000000e+00, double* %beta
  store i64 4, i64* %ldc, align 16
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p) 
  %ptr = bitcast i8* %A to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA, i8* %B, i8* %dB) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %dA, metadata !"enzyme_dup", i8* %B, i8* %dB)
  ret void
}

; CHECK: define internal void @diffef(i8* %C, i8* %"C'", i8* %A, i8* %"A'", i8* %B, i8* %"B'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64
; CHECK-NEXT:   %[[byrefconstantint4:.+]] = alloca i64
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double
; CHECK-NEXT:   %[[byrefconstantint5:.+]] = alloca i64
; CHECK-NEXT:   %transa = alloca i8, align 1
; CHECK-NEXT:   %transb = alloca i8, align 1
; CHECK-NEXT:   %m = alloca i64, align 16
; CHECK-NEXT:   %m_p = bitcast i64* %m to i8*
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   %k = alloca i64, align 16
; CHECK-NEXT:   %k_p = bitcast i64* %k to i8*
; CHECK-NEXT:   %alpha = alloca double, align 16
; CHECK-NEXT:   %alpha_p = bitcast double* %alpha to i8*
; CHECK-NEXT:   %lda = alloca i64, align 16
; CHECK-NEXT:   %lda_p = bitcast i64* %lda to i8*
; CHECK-NEXT:   %ldb = alloca i64, align 16
; CHECK-NEXT:   %ldb_p = bitcast i64* %ldb to i8*
; CHECK-NEXT:   %beta = alloca double, align 16
; CHECK-NEXT:   %beta_p = bitcast double* %beta to i8*
; CHECK-NEXT:   %ldc = alloca i64, align 16
; CHECK-NEXT:   %ldc_p = bitcast i64* %ldc to i8*
; CHECK-NEXT:   store i8 78, i8* %transa, align 1
; CHECK-NEXT:   store i8 78, i8* %transb, align 1
; CHECK-NEXT:   store i64 4, i64* %m, align 16
; CHECK-NEXT:   store i64 4, i64* %n, align 16
; CHECK-NEXT:   store i64 8, i64* %k, align 16
; CHECK-NEXT:   store double 1.000000e+00, double* %alpha, align 16
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store i64 8, i64* %ldb, align 16
; CHECK-NEXT:   store double 0.000000e+00, double* %beta
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   %trans_check = load i8, i8* %transa, align 1
; CHECK-NEXT:   %trans_check1 = load i8, i8* %transb, align 1
; CHECK-NEXT:   %0 = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %1 = bitcast i8* %k_p to i64*
; CHECK-NEXT:   %2 = load i64, i64* %0
; CHECK-NEXT:   %3 = load i64, i64* %1
; CHECK-NEXT:   %4 = mul i64 %2, %3
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %4, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK-NEXT:   %5 = bitcast i8* %lda_p to i64*
; CHECK-NEXT:   %6 = load i64, i64* %5
; CHECK-NEXT:   %7 = bitcast i8* %A to double*
; CHECK:   %mul.i = add nuw nsw i64 %2, %3
; CHECK-NEXT:   %8 = icmp eq i64 %mul.i, 0
; CHECK-NEXT:   br i1 %8, label %__enzyme_memcpy_double_mat_64.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %init.end.i, %entry
; CHECK-NEXT:   %j.i = phi i64 [ 0, %entry ], [ %j.next.i, %init.end.i ]
; CHECK-NEXT:   br label %for.body.i

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %i.i = phi i64 [ 0, %init.idx.i ], [ %i.next.i, %for.body.i ]
; CHECK-NEXT:   %9 = mul nuw nsw i64 %j.i, %2
; CHECK-NEXT:   %10 = add nuw nsw i64 %i.i, %9
; CHECK-NEXT:   %dst.i.i = getelementptr inbounds double, double* %cache.A, i64 %10
; CHECK-NEXT:   %11 = mul nuw nsw i64 %j.i, %6
; CHECK-NEXT:   %12 = add nuw nsw i64 %i.i, %11
; CHECK-NEXT:   %dst.i1.i = getelementptr inbounds double, double* %7, i64 %12
; CHECK-NEXT:   %src.i.l.i = load double, double* %dst.i1.i
; CHECK-NEXT:   store double %src.i.l.i, double* %dst.i.i
; CHECK-NEXT:   %i.next.i = add nuw nsw i64 %i.i, 1
; CHECK-NEXT:   %13 = icmp eq i64 %i.next.i, %2
; CHECK-NEXT:   br i1 %13, label %init.end.i, label %for.body.i

; CHECK: init.end.i:                                       ; preds = %for.body.i
; CHECK-NEXT:   %j.next.i = add nuw nsw i64 %j.i, 1
; CHECK-NEXT:   %14 = icmp eq i64 %j.next.i, %3
; CHECK-NEXT:   br i1 %14, label %__enzyme_memcpy_double_mat_64.exit, label %init.idx.i

; CHECK: __enzyme_memcpy_double_mat_64.exit:               ; preds = %entry, %init.end.i
; CHECK-NEXT:   %15 = bitcast i8* %k_p to i64*
; CHECK-NEXT:   %16 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %17 = load i64, i64* %15
; CHECK-NEXT:   %18 = load i64, i64* %16
; CHECK-NEXT:   %19 = mul i64 %17, %18
; CHECK-NEXT:   %[[mallocsize1:.+]] = mul nuw nsw i64 %19, 8
; CHECK-NEXT:   %[[malloccall2:.+]] = tail call noalias nonnull i8* @malloc(i64 %[[mallocsize1]])
; CHECK-NEXT:   %cache.B = bitcast i8* %[[malloccall2]] to double*
; CHECK-NEXT:   %20 = bitcast i8* %ldb_p to i64*
; CHECK-NEXT:   %21 = load i64, i64* %20
; CHECK-NEXT:   %22 = bitcast i8* %B to double*
; CHECK:   %[[muli6:.+]] = add nuw nsw i64 %17, %18
; CHECK-NEXT:   %23 = icmp eq i64 %[[muli6]], 0
; CHECK-NEXT:   br i1 %23, label %[[__enzyme_memcpy_double_mat_64exit17:.+]], label %[[init:.+]]

; CHECK: [[init]]:                                      ; preds = %[[forend:.+]], %__enzyme_memcpy_double_mat_64.exit
; CHECK-NEXT:   %[[ji6:.+]] = phi i64 [ 0, %__enzyme_memcpy_double_mat_64.exit ], [ %[[ji14:.+]], %[[forend]] ]
; CHECK-NEXT:   br label %[[forbody:.+]]

; CHECK: [[forbody]]:                                     ; preds = %[[forbody]], %[[init]]
; CHECK-NEXT:   %[[ii8:.+]] = phi i64 [ 0, %[[init]] ], [ %[[ii12:.+]], %[[forbody]] ]
; CHECK-NEXT:   %24 = mul nuw nsw i64 %[[ji6]], %17
; CHECK-NEXT:   %25 = add nuw nsw i64 %[[ii8]], %24
; CHECK-NEXT:   %[[dstii9:.+]] = getelementptr inbounds double, double* %cache.B, i64 %25
; CHECK-NEXT:   %26 = mul nuw nsw i64 %[[ji6]], %21
; CHECK-NEXT:   %27 = add nuw nsw i64 %[[ii8]], %26
; CHECK-NEXT:   %[[dsti10:.+]] = getelementptr inbounds double, double* %22, i64 %27
; CHECK-NEXT:   %[[srci11:.+]] = load double, double* %[[dsti10]]
; CHECK-NEXT:   store double %[[srci11]], double* %[[dstii9]]
; CHECK-NEXT:   %[[ii12]] = add nuw nsw i64 %[[ii8]], 1
; CHECK-NEXT:   %28 = icmp eq i64 %[[ii12]], %17
; CHECK-NEXT:   br i1 %28, label %[[forend]], label %[[forbody]]

; CHECK: [[forend]]:                                     ; preds = %[[forbody]]
; CHECK-NEXT:   %[[ji14]] = add nuw nsw i64 %[[ji6]], 1
; CHECK-NEXT:   %29 = icmp eq i64 %[[ji14]], %18
; CHECK-NEXT:   br i1 %29, label %[[__enzyme_memcpy_double_mat_64exit17]], label %[[init]]

; CHECK: [[__enzyme_memcpy_double_mat_64exit17]]:             ; preds = %__enzyme_memcpy_double_mat_64.exit, %[[forend]]
; CHECK-NEXT:   %30 = insertvalue { double*, double* } undef, double* %cache.A, 0
; CHECK-NEXT:   %31 = insertvalue { double*, double* } %30, double* %cache.B, 1
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta_p, i8* %C, i8* %ldc_p)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %A to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !10, !noalias !13
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %[[__enzyme_memcpy_double_mat_64exit17]]
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !13, !noalias !10
; CHECK-NEXT:   %tape.ext.A = extractvalue { double*, double* } %31, 0
; CHECK-NEXT:   %32 = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   %tape.ext.B = extractvalue { double*, double* } %31, 1
; CHECK-NEXT:   %33 = bitcast double* %tape.ext.B to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-NEXT:   %34 = icmp eq i8 %ld.transa, 110
; CHECK-NEXT:   %35 = select i1 %34, i8 116, i8 0
; CHECK-NEXT:   %36 = icmp eq i8 %ld.transa, 78
; CHECK-NEXT:   %37 = select i1 %36, i8 84, i8 %35
; CHECK-NEXT:   %38 = icmp eq i8 %ld.transa, 116
; CHECK-NEXT:   %39 = select i1 %38, i8 110, i8 %37
; CHECK-NEXT:   %40 = icmp eq i8 %ld.transa, 84
; CHECK-NEXT:   %41 = select i1 %40, i8 78, i8 %39
; CHECK-NEXT:   store i8 %41, i8* %byref.transpose.transa
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-NEXT:   %42 = icmp eq i8 %ld.transb, 110
; CHECK-NEXT:   %43 = select i1 %42, i8 116, i8 0
; CHECK-NEXT:   %44 = icmp eq i8 %ld.transb, 78
; CHECK-NEXT:   %45 = select i1 %44, i8 84, i8 %43
; CHECK-NEXT:   %46 = icmp eq i8 %ld.transb, 116
; CHECK-NEXT:   %47 = select i1 %46, i8 110, i8 %45
; CHECK-NEXT:   %48 = icmp eq i8 %ld.transb, 84
; CHECK-NEXT:   %49 = select i1 %48, i8 78, i8 %47
; CHECK-NEXT:   store i8 %49, i8* %byref.transpose.transb
; CHECK-NEXT:   %get.cached.ld.trans = load i8, i8* %transb
; CHECK-NEXT:   %50 = icmp eq i8 %get.cached.ld.trans, 78
; CHECK-NEXT:   %51 = select i1 %50, i8* %k_p, i8* %n_p
; CHECK-NEXT:   %52 = icmp eq i8 %get.cached.ld.trans, 110
; CHECK-NEXT:   %53 = select i1 %52, i8* %k_p, i8* %51
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %byref.transpose.transb, i8* %m_p, i8* %k_p, i8* %n_p, i8* %alpha_p, i8* %"C'", i8* %ldc_p, i8* %33, i8* %53, i8* %beta_p, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   %[[cachedtrans3:.+]] = load i8, i8* %transa
; CHECK-NEXT:   %54 = icmp eq i8 %[[cachedtrans3]], 78
; CHECK-NEXT:   %55 = select i1 %54, i8* %m_p, i8* %k_p
; CHECK-NEXT:   %56 = icmp eq i8 %[[cachedtrans3]], 110
; CHECK-NEXT:   %57 = select i1 %56, i8* %m_p, i8* %55
; CHECK-NEXT:   call void @dgemm_64_(i8* %byref.transpose.transa, i8* %transb, i8* %k_p, i8* %n_p, i8* %m_p, i8* %alpha_p, i8* %32, i8* %57, i8* %"C'", i8* %ldc_p, i8* %beta_p, i8* %"B'", i8* %ldb_p)
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %[[byrefconstantint4]]
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   store i64 0, i64* %[[byrefconstantint5]]
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %[[byrefconstantint4]], double* %byref.constant.fp.1.0, i8* %beta_p, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %[[byrefconstantint5]])
; CHECK-NEXT:   %[[i50:.+]] = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i50]])
; CHECK-NEXT:   %[[i51:.+]] = bitcast double* %tape.ext.B to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i51]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -S | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly) 

define void @f(i8* noalias %C, i8* noalias %A, i8* noalias %B, i8* noalias %beta) {
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
  store i64 4, i64* %ldc, align 16
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %C, i8* %ldc_p) 
  %ptr = bitcast i8* %A to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA, i8* %B, i8* %dB, i8* %beta, i8* %dbeta) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %dA, metadata !"enzyme_dup", i8* %B, i8* %dB, metadata !"enzyme_dup", i8* %beta, i8* %dbeta)
  ret void
}

; CHECK: define internal void @diffef(i8* noalias %C, i8* %"C'", i8* noalias %A, i8* %"A'", i8* noalias %B, i8* %"B'", i8* noalias %beta, i8*
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.constant.one.i = alloca i64
; CHECK-NEXT:   %byref.mat.size.i = alloca i64
; CHECK-NEXT:   %[[byrefgarbage:.+]] = alloca i8
; CHECK-NEXT:   %[[byrefgarbage2:.+]] = alloca i8
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.constant.char.G = alloca i8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64
; CHECK-NEXT:   %[[int04:.+]] = alloca i64
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double
; CHECK-NEXT:   %[[int05:.+]] = alloca i64
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
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   %loaded.trans = load i8, i8* %transa
; CHECK-DAG:   %[[i0:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[i1:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %2 = or i1 %[[i1]], %[[i0]]
; CHECK-NEXT:   %3 = select i1 %2, i8* %m_p, i8* %k_p
; CHECK-NEXT:   %4 = select i1 %2, i8* %k_p, i8* %m_p
; CHECK-NEXT:   %[[i5:.+]] = bitcast i8* %3 to i64*
; CHECK-NEXT:   %[[i6:.+]] = load i64, i64* %[[i5]]
; CHECK-NEXT:   %[[i7:.+]] = bitcast i8* %4 to i64*
; CHECK-NEXT:   %[[i8:.+]] = load i64, i64* %[[i7]]
; CHECK-NEXT:   %9 = mul i64 %[[i6]], %[[i8]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %9, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.A = bitcast i8* %malloccall to double*
; CHECK-NEXT:   store i8 0, i8* %[[byrefgarbage]]
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage, i8* %3, i8* %4, i8* %A, i8* %lda_p, double* %cache.A, i8* %3)
; CHECK-NEXT:   %10 = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %11 = load i64, i64* %10
; CHECK-NEXT:   %12 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %13 = load i64, i64* %12
; CHECK-NEXT:   %14 = mul i64 %11, %13
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i64 %14, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i64 %mallocsize1)
; CHECK-NEXT:   %cache.C = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   store i8 0, i8* %byref.copy.garbage3
; CHECK-NEXT:   call void @dlacpy_64_(i8* %byref.copy.garbage3, i8* %m_p, i8* %n_p, i8* %C, i8* %ldc_p, double* %cache.C, i8* %m_p)
; CHECK-NEXT:   %15 = insertvalue { double*, double* } undef, double* %cache.A, 0
; CHECK-NEXT:   %[[i23:.+]] = insertvalue { double*, double* } %15, double* %cache.C, 1
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha_p, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %C, i8* %ldc_p)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %A to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %tape.ext.A = extractvalue { double*, double* } %[[i23]], 0
; CHECK-NEXT:   %[[matA:.+]] = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   %tape.ext.C = extractvalue { double*, double* } %[[i23]], 1
; CHECK-NEXT:   %[[matC0:.+]] = bitcast double* %tape.ext.C to i8*
; CHECK-NEXT:   %tape.ext.C4 = extractvalue { double*, double* } %[[i23]], 1
; CHECK-NEXT:   %[[matC:.+]] = bitcast double* %tape.ext.C4 to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-DAG:    %[[i26:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[i27:.+]] = select i1 %[[i26]], i8 116, i8 0
; CHECK-DAG:    %[[i28:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[i29:.+]] = select i1 %[[i28]], i8 84, i8 %[[i27]]
; CHECK-DAG:    %[[i30:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[i31:.+]] = select i1 %[[i30]], i8 110, i8 %[[i29]]
; CHECK-DAG:    %[[i32:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[i33:.+]] = select i1 %[[i32]], i8 78, i8 %[[i31]]
; CHECK-NEXT:   store i8 %[[i33]], i8* %byref.transpose.transa
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-DAG:    %[[i34:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:    %[[i35:.+]] = select i1 %[[i34]], i8 116, i8 0
; CHECK-DAG:    %[[i36:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:    %[[i37:.+]] = select i1 %[[i36]], i8 84, i8 %[[i35]]
; CHECK-DAG:    %[[i38:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:    %[[i39:.+]] = select i1 %[[i38]], i8 110, i8 %[[i37]]
; CHECK-DAG:    %[[i40:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:    %[[i41:.+]] = select i1 %[[i40]], i8 78, i8 %[[i39]]
; CHECK-NEXT:   store i8 %[[i41]], i8* %byref.transpose.transb
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %byref.transpose.transb, i8* %m_p, i8* %k_p, i8* %n_p, i8* %alpha_p, i8* %"C'", i8* %ldc_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   %loaded.trans5 = load i8, i8* %transa
; CHECK-DAG:   %[[i46:.+]] = icmp eq i8 %loaded.trans5, 78
; CHECK-DAG:   %[[i47:.+]] = icmp eq i8 %loaded.trans5, 110
; CHECK-NEXT:   %[[i48:.+]] = or i1 %[[i47]], %[[i46]]
; CHECK-NEXT:   %39 = select i1 %[[i48]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   call void @dgemm_64_(i8* %byref.transpose.transa, i8* %transb, i8* %k_p, i8* %n_p, i8* %m_p, i8* %alpha_p, i8* %17, i8* %39, i8* %"C'", i8* %ldc_p, i8* %beta, i8* %"B'", i8* %ldb_p)
; CHECK:   %40 = bitcast i64* %byref.constant.one.i to i8*
; CHECK:   %41 = bitcast i64* %byref.mat.size.i to i8*
; CHECK:   store i64 1, i64* %byref.constant.one.i
; CHECK-NEXT:   %intcast.constant.one.i = bitcast i64* %byref.constant.one.i to i8*
; CHECK-DAG:   %[[i52:.+]] = load i64, i64* %m
; CHECK-DAG:   %[[i53:.+]] = load i64, i64* %n
; CHECK-DAG:   %mat.size.i = mul nuw i64 %[[i52]], %[[i53]]
; CHECK-NEXT:   store i64 %mat.size.i, i64* %byref.mat.size.i
; CHECK-NEXT:   %intcast.mat.size.i = bitcast i64* %byref.mat.size.i to i8*
; CHECK-NEXT:   %44 = icmp eq i64 %mat.size.i, 0
; CHECK-NEXT:   br i1 %44, label %__enzyme_inner_prodd_64_.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %invertentry
; CHECK-NEXT:   %45 = load i64, i64* %ldc
; CHECK-NEXT:   %46 = bitcast i8* %"C'" to double*
; CHECK-NEXT:   %47 = icmp eq i64 %42, %45
; CHECK-NEXT:   br i1 %47, label %fast.path.i, label %for.body.i

; CHECK: fast.path.i:                                      ; preds = %init.idx.i
; CHECK-NEXT:   %48 = call fast double @ddot_64_(i8* %intcast.mat.size.i, i8* %"C'", i8* %intcast.constant.one.i, i8* %18, i8* %intcast.constant.one.i)
; CHECK-NEXT:   br label %__enzyme_inner_prodd_64_.exit

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %Aidx.i = phi i64 [ 0, %init.idx.i ], [ %Aidx.next.i, %for.body.i ]
; CHECK-NEXT:   %Bidx.i = phi i64 [ 0, %init.idx.i ], [ %Bidx.next.i, %for.body.i ]
; CHECK-NEXT:   %iteration.i = phi i64 [ 0, %init.idx.i ], [ %iter.next.i, %for.body.i ]
; CHECK-NEXT:   %sum.i = phi{{( fast)?}} double [ 0.000000e+00, %init.idx.i ], [ %52, %for.body.i ]
; CHECK-NEXT:   %A.i.i = getelementptr inbounds double, double* %46, i64 %Aidx.i
; CHECK-NEXT:   %B.i.i = getelementptr inbounds double, double* %tape.ext.C, i64 %Bidx.i
; CHECK-NEXT:   %49 = bitcast double* %A.i.i to i8*
; CHECK-NEXT:   %50 = bitcast double* %B.i.i to i8*
; CHECK-NEXT:   %51 = call fast double @ddot_64_(i8* %m_p, i8* %49, i8* %intcast.constant.one.i, i8* %50, i8* %intcast.constant.one.i)
; CHECK-NEXT:   %Aidx.next.i = add nuw i64 %Aidx.i, %45
; CHECK-NEXT:   %Bidx.next.i = add nuw i64 %Aidx.i, %42
; CHECK-NEXT:   %iter.next.i = add i64 %iteration.i, 1
; CHECK-NEXT:   %52 = fadd fast double %sum.i, %51
; CHECK-NEXT:   %53 = icmp eq i64 %iteration.i, %43
; CHECK-NEXT:   br i1 %53, label %__enzyme_inner_prodd_64_.exit, label %for.body.i

; CHECK: __enzyme_inner_prodd_64_.exit:                    ; preds = %invertentry, %fast.path.i, %for.body.i
; CHECK-NEXT:   %res.i = phi double [ 0.000000e+00, %invertentry ], [ %sum.i, %for.body.i ], [ %48, %fast.path.i ]
; CHECK-NEXT:   %54 = bitcast i64* %byref.constant.one.i to i8*
; CHECK:   %55 = bitcast i64* %byref.mat.size.i to i8*
; CHECK:   %56 = bitcast i8* %"beta'" to double*
; CHECK-NEXT:   %57 = load double, double* %56
; CHECK-NEXT:   %58 = fadd fast double %57, %res.i
; CHECK-NEXT:   store double %58, double* %56
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   store i64 0, i64* %[[int04]]
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   store i64 0, i64* %[[int05]]
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i64* %byref.constant.int.0, i64* %[[int04]], i8* %fpcast.constant.fp.1.0, i8* %beta, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i64* %[[int05]])
; CHECK-NEXT:   %[[i30:.+]] = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i30]])
; CHECK-NEXT:   %[[i31:.+]] = bitcast double* %tape.ext.C4 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i31]])
; CHECK-NEXT:   %[[i32:.+]] = bitcast double* %tape.ext.C to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i32]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double @__enzyme_inner_prodd_64_(i8* %blasm, i8* %blasn, i8* noalias nocapture readonly %A, i8* %lda, i8* noalias nocapture readonly %B
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.constant.one = alloca i64
; CHECK-NEXT:   store i64 1, i64* %byref.constant.one
; CHECK-NEXT:   %intcast.constant.one = bitcast i64* %byref.constant.one to i8*
; CHECK-NEXT:   %0 = bitcast i8* %blasm to i64*
; CHECK-NEXT:   %1 = load i64, i64* %0
; CHECK-NEXT:   %2 = bitcast i8* %blasn to i64*
; CHECK-NEXT:   %3 = load i64, i64* %2
; CHECK-NEXT:   %mat.size = mul nuw i64 %1, %3
; CHECK-NEXT:   %byref.mat.size = alloca i64
; CHECK-NEXT:   store i64 %mat.size, i64* %byref.mat.size
; CHECK-NEXT:   %intcast.mat.size = bitcast i64* %byref.mat.size to i8*
; CHECK-NEXT:   %4 = icmp eq i64 %mat.size, 0
; CHECK-NEXT:   br i1 %4, label %for.end, label %init.idx

; CHECK: init.idx:                                         ; preds = %entry
; CHECK-NEXT:   %5 = bitcast i8* %lda to i64*
; CHECK-NEXT:   %6 = load i64, i64* %5
; CHECK-NEXT:   %7 = bitcast i8* %A to double*
; CHECK-NEXT:   %8 = bitcast i8* %B to double*
; CHECK-NEXT:   %9 = icmp eq i64 %1, %6
; CHECK-NEXT:   br i1 %9, label %fast.path, label %for.body

; CHECK: fast.path:                                        ; preds = %init.idx
; CHECK-NEXT:   %10 = call fast double @ddot_64_(i8* %intcast.mat.size, i8* %A, i8* %intcast.constant.one, i8* %B, i8* %intcast.constant.one)
; CHECK-NEXT:   br label %for.end

; CHECK: for.body:                                         ; preds = %for.body, %init.idx
; CHECK-NEXT:   %Aidx = phi i64 [ 0, %init.idx ], [ %Aidx.next, %for.body ]
; CHECK-NEXT:   %Bidx = phi i64 [ 0, %init.idx ], [ %Bidx.next, %for.body ]
; CHECK-NEXT:   %iteration = phi i64 [ 0, %init.idx ], [ %iter.next, %for.body ]
; CHECK-NEXT:   %sum = phi{{( fast)?}} double [ 0.000000e+00, %init.idx ], [ %14, %for.body ]
; CHECK-NEXT:   %A.i = getelementptr inbounds double, double* %7, i64 %Aidx
; CHECK-NEXT:   %B.i = getelementptr inbounds double, double* %8, i64 %Bidx
; CHECK-NEXT:   %11 = bitcast double* %A.i to i8*
; CHECK-NEXT:   %12 = bitcast double* %B.i to i8*
; CHECK-NEXT:   %13 = call fast double @ddot_64_(i8* %blasm, i8* %11, i8* %intcast.constant.one, i8* %12, i8* %intcast.constant.one)
; CHECK-NEXT:   %Aidx.next = add nuw i64 %Aidx, %6
; CHECK-NEXT:   %Bidx.next = add nuw i64 %Aidx, %1
; CHECK-NEXT:   %iter.next = add i64 %iteration, 1
; CHECK-NEXT:   %14 = fadd fast double %sum, %13
; CHECK-NEXT:   %15 = icmp eq i64 %iteration, %3
; CHECK-NEXT:   br i1 %15, label %for.end, label %for.body

; CHECK: for.end:                                          ; preds = %for.body, %fast.path, %entry
; CHECK-NEXT:   %res = phi double [ 0.000000e+00, %entry ], [ %sum, %for.body ], [ %10, %fast.path ]
; CHECK-NEXT:   ret double %res
; CHECK-NEXT: }

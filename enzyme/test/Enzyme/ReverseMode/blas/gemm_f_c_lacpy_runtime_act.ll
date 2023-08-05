;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-lapack-copy=1 -enzyme-runtime-activity=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-lapack-copy=1 -enzyme-runtime-activity=1 -S | FileCheck %s

declare void @dgemm_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8*, i8* nocapture readonly) 

define void @f(i8* noalias %C, i8* noalias %A, i8* noalias %B, i8* noalias %alpha, i8* noalias %beta) {
entry:
  %transa = alloca i8, align 1
  %transb = alloca i8, align 1
  %m = alloca i64, align 16
  %m_p = bitcast i64* %m to i8*
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %k = alloca i64, align 16
  %k_p = bitcast i64* %k to i8*
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
  store i64 4, i64* %lda, align 16
  store i64 8, i64* %ldb, align 16
  store i64 4, i64* %ldc, align 16
  call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %C, i8* %ldc_p) 
  %ptr = bitcast i8* %A to double*
  store double 0.0000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %C, i8* %dC, i8* %A, i8* %dA, i8* %B, i8* %dB, i8* %alpha, i8* %dalpha, i8* %beta, i8* %dbeta) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*,i8*,i8*)* @f, metadata !"enzyme_dup", i8* %C, i8* %dC, metadata !"enzyme_dup", i8* %A, i8* %A, metadata !"enzyme_dup", i8* %B, i8* %dB, metadata !"enzyme_dup", i8* %alpha, i8* %dalpha, metadata !"enzyme_dup", i8* %beta, i8* %beta)
  ret void
}

; CHECK: define internal void @diffef(i8* noalias %C, i8* %"C'", i8* noalias %A, i8* %"A'", i8* noalias %B, i8* %"B'", i8* noalias %alpha, i8* %"alpha'", i8* noalias %beta, i8*
; CHECK-NEXT: entry:
; CHECK-NEXT:   %byref.constant.one.i15 = alloca i64
; CHECK-NEXT:   %byref.mat.size.i18 = alloca i64
; CHECK-NEXT:   %byref.constant.one.i = alloca i64
; CHECK-NEXT:   %byref.mat.size.i = alloca i64
; CHECK-NEXT:   %[[byrefgarbage:.+]] = alloca i8
; CHECK-NEXT:   %[[byrefgarbage2:.+]] = alloca i8
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %byref.transpose.transb = alloca i8
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-NEXT:   %byref.constant.fp.1.0 = alloca double
; CHECK-NEXT:   %byref.constant.fp.0.0 = alloca double
; CHECK-NEXT:   %byref.constant.char.G = alloca i8
; CHECK-NEXT:   %byref.constant.int.0 = alloca i64
; CHECK-NEXT:   %byref.constant.int.09 = alloca i64
; CHECK-NEXT:   %byref.constant.fp.1.011 = alloca double
; CHECK-NEXT:   %byref.constant.int.013 = alloca i64
; CHECK-NEXT:   %transa = alloca i8, align 1
; CHECK-NEXT:   %transb = alloca i8, align 1
; CHECK-NEXT:   %m = alloca i64, align 16
; CHECK-NEXT:   %m_p = bitcast i64* %m to i8*
; CHECK-NEXT:   %n = alloca i64, align 16
; CHECK-NEXT:   %n_p = bitcast i64* %n to i8*
; CHECK-NEXT:   %k = alloca i64, align 16
; CHECK-NEXT:   %k_p = bitcast i64* %k to i8*
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
; CHECK-NEXT:   store i64 4, i64* %lda, align 16
; CHECK-NEXT:   store i64 8, i64* %ldb, align 16
; CHECK-NEXT:   store i64 4, i64* %ldc, align 16
; CHECK-NEXT:   %rt.inactive.alpha = icmp eq i8* %"alpha'", %alpha
; CHECK-NEXT:   %rt.inactive.A = icmp eq i8* %"A'", %A
; CHECK-NEXT:   %rt.inactive.B = icmp eq i8* %"B'", %B
; CHECK-NEXT:   %rt.inactive.beta = icmp eq i8* %"beta'", %beta
; CHECK-NEXT:   %rt.inactive.C = icmp eq i8* %"C'", %C
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
; CHECK-NEXT:   %[[i16:.+]] = insertvalue { double*, double* } %15, double* %cache.C, 1
; CHECK-NEXT:   %17 = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %18 = load i64, i64* %17
; CHECK-NEXT:   %19 = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %20 = load i64, i64* %19
; CHECK-NEXT:   %size_AB = mul nuw i64 %18, %20
; CHECK-NEXT:   %mallocsize5 = mul nuw nsw i64 %size_AB, 8
; CHECK-NEXT:   %malloccall6 = tail call noalias nonnull i8* @malloc(i64 %mallocsize5)
; CHECK-NEXT:   %mat_AB = bitcast i8* %malloccall6 to double*
; CHECK-NEXT:   %21 = bitcast double* %mat_AB to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %C, i8* %ldc_p)
; CHECK-NEXT:   %"ptr'ipc" = bitcast i8* %"A'" to double*
; CHECK-NEXT:   %ptr = bitcast i8* %A to double*
; CHECK-NEXT:   store double 0.000000e+00, double* %ptr, align 8, !alias.scope !0, !noalias !3
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double 0.000000e+00, double* %"ptr'ipc", align 8, !alias.scope !3, !noalias !0
; CHECK-NEXT:   %tape.ext.A = extractvalue { double*, double* } %[[i16]], 0
; CHECK-NEXT:   %[[matA:.+]] = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   %tape.ext.C = extractvalue { double*, double* } %[[i16]], 1
; CHECK-NEXT:   %[[matC0:.+]] = bitcast double* %tape.ext.C to i8*
; CHECK-NEXT:   %tape.ext.C4 = extractvalue { double*, double* } %[[i16]], 1
; CHECK-NEXT:   %[[matC:.+]] = bitcast double* %tape.ext.C4 to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %transa
; CHECK-DAG:    %[[i25:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[i26:.+]] = select i1 %[[i25]], i8 116, i8 0
; CHECK-DAG:    %[[i27:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[i28:.+]] = select i1 %[[i27]], i8 84, i8 %[[i26]]
; CHECK-DAG:    %[[i29:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[i30:.+]] = select i1 %[[i29]], i8 110, i8 %[[i28]]
; CHECK-DAG:    %[[i31:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[i32:.+]] = select i1 %[[i31]], i8 78, i8 %[[i30]]
; CHECK-NEXT:   store i8 %[[i32]], i8* %byref.transpose.transa
; CHECK-NEXT:   %ld.transb = load i8, i8* %transb
; CHECK-DAG:    %[[i33:.+]] = icmp eq i8 %ld.transb, 110
; CHECK-DAG:    %[[i34:.+]] = select i1 %[[i33]], i8 116, i8 0
; CHECK-DAG:    %[[i35:.+]] = icmp eq i8 %ld.transb, 78
; CHECK-DAG:    %[[i36:.+]] = select i1 %[[i35]], i8 84, i8 %[[i34]]
; CHECK-DAG:    %[[i37:.+]] = icmp eq i8 %ld.transb, 116
; CHECK-DAG:    %[[i38:.+]] = select i1 %[[i37]], i8 110, i8 %[[i36]]
; CHECK-DAG:    %[[i39:.+]] = icmp eq i8 %ld.transb, 84
; CHECK-DAG:    %[[i40:.+]] = select i1 %[[i39]], i8 78, i8 %[[i38]]
; CHECK-NEXT:   store i8 %[[i40]], i8* %byref.transpose.transb
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %intcast.int.one = bitcast i64* %byref.int.one to i8*
; CHECK-NEXT:   br i1 %rt.inactive.alpha, label %invertentry.alpha.done, label %invertentry.alpha.active

; CHECK: invertentry.alpha.active:                         ; preds = %invertentry
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   %loaded.trans7 = load i8, i8* %transa
; CHECK-DAG:   %[[i41:.+]] = icmp eq i8 %loaded.trans7, 78
; CHECK-DAG:   %[[i42:.+]] = icmp eq i8 %loaded.trans7, 110
; CHECK-NEXT:   %[[i43:.+]] = or i1 %[[i42]], %[[i41]]
; CHECK-NEXT:   %[[i44:.+]] = select i1 %43, i8* %m_p, i8* %k_p
; CHECK-NEXT:   store double 0.000000e+00, double* %byref.constant.fp.0.0
; CHECK-NEXT:   %fpcast.constant.fp.0.0 = bitcast double* %byref.constant.fp.0.0 to i8*
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %transb, i8* %m_p, i8* %n_p, i8* %k_p, i8* %fpcast.constant.fp.1.0, i8* %22, i8* %[[i44]], i8* %B, i8* %ldb_p, i8* %fpcast.constant.fp.0.0, i8* %21, i8* %m_p)
; CHECK:   %45 = bitcast i64* %byref.constant.one.i to i8*
; CHECK:   %46 = bitcast i64* %byref.mat.size.i to i8*
; CHECK:   store i64 1, i64* %byref.constant.one.i
; CHECK-NEXT:   %intcast.constant.one.i = bitcast i64* %byref.constant.one.i to i8*
; CHECK-DAG:   %[[i47:.+]] = load i64, i64* %m
; CHECK-DAG:   %[[i48:.+]] = load i64, i64* %n
; CHECK-DAG:   %mat.size.i = mul nuw i64 %[[i47]], %[[i48]]
; CHECK-NEXT:   store i64 %mat.size.i, i64* %byref.mat.size.i
; CHECK-NEXT:   %intcast.mat.size.i = bitcast i64* %byref.mat.size.i to i8*
; CHECK-NEXT:   %[[i49:.+]] = icmp eq i64 %mat.size.i, 0
; CHECK-NEXT:   br i1 %[[i49]], label %__enzyme_inner_prodd_64_.exit, label %init.idx.i

; CHECK: init.idx.i:                                       ; preds = %invertentry.alpha.active
; CHECK-NEXT:   %[[i50:.+]] = load i64, i64* %ldc
; CHECK-NEXT:   %[[i51:.+]] = bitcast i8* %"C'" to double*
; CHECK-NEXT:   %[[i52:.+]] = icmp eq i64 %[[i47]], %[[i50]]
; CHECK-NEXT:   br i1 %[[i52]], label %fast.path.i, label %for.body.i

; CHECK: fast.path.i:                                      ; preds = %init.idx.i
; CHECK-NEXT:   %[[i53:.+]] = call fast double @ddot_64_(i8* %intcast.mat.size.i, i8* %"C'", i8* %intcast.constant.one.i, i8* %21, i8* %intcast.constant.one.i)
; CHECK-NEXT:   br label %__enzyme_inner_prodd_64_.exit

; CHECK: for.body.i:                                       ; preds = %for.body.i, %init.idx.i
; CHECK-NEXT:   %Aidx.i = phi i64 [ 0, %init.idx.i ], [ %Aidx.next.i, %for.body.i ]
; CHECK-NEXT:   %Bidx.i = phi i64 [ 0, %init.idx.i ], [ %Bidx.next.i, %for.body.i ]
; CHECK-NEXT:   %iteration.i = phi i64 [ 0, %init.idx.i ], [ %iter.next.i, %for.body.i ]
; CHECK-NEXT:   %sum.i = phi{{( fast)?}} double [ 0.000000e+00, %init.idx.i ], [ %57, %for.body.i ]
; CHECK-NEXT:   %A.i.i = getelementptr inbounds double, double* %51, i64 %Aidx.i
; CHECK-NEXT:   %B.i.i = getelementptr inbounds double, double* %mat_AB, i64 %Bidx.i
; CHECK-NEXT:   %54 = bitcast double* %A.i.i to i8*
; CHECK-NEXT:   %55 = bitcast double* %B.i.i to i8*
; CHECK-NEXT:   %56 = call fast double @ddot_64_(i8* %m_p, i8* %54, i8* %intcast.constant.one.i, i8* %55, i8* %intcast.constant.one.i)
; CHECK-NEXT:   %Aidx.next.i = add nuw i64 %Aidx.i, %50
; CHECK-NEXT:   %Bidx.next.i = add nuw i64 %Aidx.i, %47
; CHECK-NEXT:   %iter.next.i = add i64 %iteration.i, 1
; CHECK-NEXT:   %57 = fadd fast double %sum.i, %56
; CHECK-NEXT:   %58 = icmp eq i64 %iteration.i, %48
; CHECK-NEXT:   br i1 %58, label %__enzyme_inner_prodd_64_.exit, label %for.body.i

; CHECK: __enzyme_inner_prodd_64_.exit:                    ; preds = %invertentry.alpha.active, %fast.path.i, %for.body.i
; CHECK-NEXT:   %res.i = phi double [ 0.000000e+00, %invertentry.alpha.active ], [ %sum.i, %for.body.i ], [ %[[i53]], %fast.path.i ]
; CHECK-NEXT:   %59 = bitcast i64* %byref.constant.one.i to i8*
; CHECK:   %60 = bitcast i64* %byref.mat.size.i to i8*
; CHECK:   %61 = bitcast i8* %"alpha'" to double*
; CHECK-NEXT:   %62 = load double, double* %61
; CHECK-NEXT:   %63 = fadd fast double %62, %res.i
; CHECK-NEXT:   store double %63, double* %61
; CHECK-NEXT:   br label %invertentry.alpha.done

; CHECK: invertentry.alpha.done:                           ; preds = %__enzyme_inner_prodd_64_.exit, %invertentry
; CHECK-NEXT:   br i1 %rt.inactive.A, label %invertentry.A.done, label %invertentry.A.active

; CHECK: invertentry.A.active:                             ; preds = %invertentry.alpha.done
; CHECK-NEXT:   call void @dgemm_64_(i8* %transa, i8* %byref.transpose.transb, i8* %m_p, i8* %k_p, i8* %n_p, i8* %alpha, i8* %"C'", i8* %ldc_p, i8* %B, i8* %ldb_p, i8* %beta, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   br label %invertentry.A.done

; CHECK: invertentry.A.done:                               ; preds = %invertentry.A.active, %invertentry.alpha.done
; CHECK-NEXT:   br i1 %rt.inactive.B, label %invertentry.B.done, label %invertentry.B.active

; CHECK: invertentry.B.active:                             ; preds = %invertentry.A.done
; CHECK-NEXT:   %loaded.trans8 = load i8, i8* %transa
; CHECK-DAG:   %[[i64:.+]] = icmp eq i8 %loaded.trans8, 78
; CHECK-DAG:   %[[i65:.+]] = icmp eq i8 %loaded.trans8, 110
; CHECK-DAG:   %[[i66:.+]] = or i1 %[[i65]], %[[i64]]
; CHECK-NEXT:   %[[i67:.+]] = select i1 %[[i66]], i8* %m_p, i8* %k_p
; CHECK-NEXT:   call void @dgemm_64_(i8* %byref.transpose.transa, i8* %transb, i8* %k_p, i8* %n_p, i8* %m_p, i8* %alpha, i8* %22, i8* %[[i67]], i8* %"C'", i8* %ldc_p, i8* %beta, i8* %"B'", i8* %ldb_p)
; CHECK-NEXT:   br label %invertentry.B.done

; CHECK: invertentry.B.done:                               ; preds = %invertentry.B.active, %invertentry.A.done
; CHECK-NEXT:   br i1 %rt.inactive.beta, label %invertentry.beta.done, label %invertentry.beta.active

; CHECK: invertentry.beta.active:                          ; preds = %invertentry.B.done
; CHECK:   %68 = bitcast i64* %byref.constant.one.i15 to i8*
; CHECK:   %69 = bitcast i64* %byref.mat.size.i18 to i8*
; CHECK:   store i64 1, i64* %byref.constant.one.i15
; CHECK-NEXT:   %intcast.constant.one.i16 = bitcast i64* %byref.constant.one.i15 to i8*
; CHECK-NEXT:   %70 = load i64, i64* %m
; CHECK-NEXT:   %71 = load i64, i64* %n
; CHECK-NEXT:   %mat.size.i17 = mul nuw i64 %70, %71
; CHECK-NEXT:   store i64 %mat.size.i17, i64* %byref.mat.size.i18
; CHECK-NEXT:   %intcast.mat.size.i19 = bitcast i64* %byref.mat.size.i18 to i8*
; CHECK-NEXT:   %72 = icmp eq i64 %mat.size.i17, 0
; CHECK-NEXT:   br i1 %72, label %__enzyme_inner_prodd_64_.exit33, label %init.idx.i20

; CHECK: init.idx.i20:                                     ; preds = %invertentry.beta.active
; CHECK-NEXT:   %73 = load i64, i64* %ldc
; CHECK-NEXT:   %74 = bitcast i8* %"C'" to double*
; CHECK-NEXT:   %75 = icmp eq i64 %70, %73
; CHECK-NEXT:   br i1 %75, label %fast.path.i21, label %for.body.i31

; CHECK: fast.path.i21:                                    ; preds = %init.idx.i20
; CHECK-NEXT:   %76 = call fast double @ddot_64_(i8* %intcast.mat.size.i19, i8* %"C'", i8* %intcast.constant.one.i16, i8* %23, i8* %intcast.constant.one.i16)
; CHECK-NEXT:   br label %__enzyme_inner_prodd_64_.exit33

; CHECK: for.body.i31:                                     ; preds = %for.body.i31, %init.idx.i20
; CHECK-NEXT:   %Aidx.i22 = phi i64 [ 0, %init.idx.i20 ], [ %Aidx.next.i28, %for.body.i31 ]
; CHECK-NEXT:   %Bidx.i23 = phi i64 [ 0, %init.idx.i20 ], [ %Bidx.next.i29, %for.body.i31 ]
; CHECK-NEXT:   %iteration.i24 = phi i64 [ 0, %init.idx.i20 ], [ %iter.next.i30, %for.body.i31 ]
; CHECK-NEXT:   %sum.i25 = phi{{( fast)?}} double [ 0.000000e+00, %init.idx.i20 ], [ %80, %for.body.i31 ]
; CHECK-NEXT:   %A.i.i26 = getelementptr inbounds double, double* %74, i64 %Aidx.i22
; CHECK-NEXT:   %B.i.i27 = getelementptr inbounds double, double* %tape.ext.C, i64 %Bidx.i23
; CHECK-NEXT:   %77 = bitcast double* %A.i.i26 to i8*
; CHECK-NEXT:   %78 = bitcast double* %B.i.i27 to i8*
; CHECK-NEXT:   %79 = call fast double @ddot_64_(i8* %m_p, i8* %77, i8* %intcast.constant.one.i16, i8* %78, i8* %intcast.constant.one.i16)
; CHECK-NEXT:   %Aidx.next.i28 = add nuw i64 %Aidx.i22, %73
; CHECK-NEXT:   %Bidx.next.i29 = add nuw i64 %Aidx.i22, %70
; CHECK-NEXT:   %iter.next.i30 = add i64 %iteration.i24, 1
; CHECK-NEXT:   %80 = fadd fast double %sum.i25, %79
; CHECK-NEXT:   %81 = icmp eq i64 %iteration.i24, %71
; CHECK-NEXT:   br i1 %81, label %__enzyme_inner_prodd_64_.exit33, label %for.body.i31

; CHECK: __enzyme_inner_prodd_64_.exit33:                  ; preds = %invertentry.beta.active, %fast.path.i21, %for.body.i31
; CHECK-NEXT:   %res.i32 = phi double [ 0.000000e+00, %invertentry.beta.active ], [ %sum.i25, %for.body.i31 ], [ %76, %fast.path.i21 ]
; CHECK-NEXT:   %82 = bitcast i64* %byref.constant.one.i15 to i8*
; CHECK:   %83 = bitcast i64* %byref.mat.size.i18 to i8*
; CHECK:   %84 = bitcast i8* %"beta'" to double*
; CHECK-NEXT:   %85 = load double, double* %84
; CHECK-NEXT:   %86 = fadd fast double %85, %res.i32
; CHECK-NEXT:   store double %86, double* %84
; CHECK-NEXT:   br label %invertentry.beta.done

; CHECK: invertentry.beta.done:                            ; preds = %__enzyme_inner_prodd_64_.exit33, %invertentry.B.done
; CHECK-NEXT:   br i1 %rt.inactive.C, label %invertentry.C.done, label %invertentry.C.active

; CHECK: invertentry.C.active:                             ; preds = %invertentry.beta.done
; CHECK-NEXT:   store i8 71, i8* %byref.constant.char.G
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.0
; CHECK-NEXT:   %intcast.constant.int.0 = bitcast i64* %byref.constant.int.0 to i8*
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.09
; CHECK-NEXT:   %intcast.constant.int.010 = bitcast i64* %byref.constant.int.09 to i8*
; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.011
; CHECK-NEXT:   %fpcast.constant.fp.1.012 = bitcast double* %byref.constant.fp.1.011 to i8*
; CHECK-NEXT:   store i64 0, i64* %byref.constant.int.013
; CHECK-NEXT:   %intcast.constant.int.014 = bitcast i64* %byref.constant.int.013 to i8*
; CHECK-NEXT:   call void @dlascl_64_(i8* %byref.constant.char.G, i8* %intcast.constant.int.0, i8* %intcast.constant.int.010, i8* %fpcast.constant.fp.1.012, i8* %beta, i8* %m_p, i8* %n_p, i8* %"C'", i8* %ldc_p, i8* %intcast.constant.int.014)
; CHECK-NEXT:   br label %invertentry.C.done

; CHECK: invertentry.C.done:                               ; preds = %invertentry.C.active, %invertentry.beta.done
; CHECK-NEXT:   %[[i87:.+]] = bitcast double* %tape.ext.A to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i87]])
; CHECK-NEXT:   %[[i88:.+]] = bitcast double* %tape.ext.C4 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[i88]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

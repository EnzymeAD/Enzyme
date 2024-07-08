;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-blas-copy=1 -enzyme-lapack-copy=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-blas-copy=1  -enzyme-lapack-copy=1 -S | FileCheck %s

; Here we don't transpose the matrix a (78 equals 'N' in ASCII) and we therefore also don't transpose x.
; Therfore the first arg to dcopy is n_p, as opposed to the gemv_transpose test.
;                       trans,                  M,                       N,                     alpha,                  A,    lda,                    x,  , incx,                  beta,                    y,  incy
declare void @dgemv_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* , i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8* , i8* nocapture readonly, i64) 

define void @f(i8* noalias %y, i8* noalias %A, i8* noalias %x, i8* noalias %alpha, i8* noalias %beta) {
entry:
  %transa = alloca i8, align 1
  %m = alloca i64, align 16
  %m_p = bitcast i64* %m to i8*
  %n = alloca i64, align 16
  %n_p = bitcast i64* %n to i8*
  %lda = alloca i64, align 16
  %lda_p = bitcast i64* %lda to i8*
  %incx = alloca i64, align 16
  %incx_p = bitcast i64* %incx to i8*
  %incy = alloca i64, align 16
  %incy_p = bitcast i64* %incy to i8*
  store i8 78, i8* %transa, align 1
  store i64 4, i64* %m, align 16
  store i64 4, i64* %n, align 16
  store i64 4, i64* %lda, align 16
  store i64 2, i64* %incx, align 16
  store i64 1, i64* %incy, align 16
  call void @dgemv_64_(i8* %transa, i8* %m_p, i8* %n_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta, i8* %y, i8* %incy_p, i64 1)
  ret void
}

define void @g(i8* noalias %y, i8* noalias %A, i8* noalias %x, i8* noalias %alpha, i8* noalias %beta) {
entry:
  call void @f(i8* %y, i8* %A, i8* %x, i8* %alpha, i8* %beta)
  %ptr = bitcast i8* %x to double*
  store double 0.000000e+00, double* %ptr, align 8
  ret void
}

declare dso_local void @__enzyme_autodiff(...)

define void @active(i8* %y, i8* %dy, i8* %A, i8* %dA, i8* %x, i8* %dx, i8* %alpha, i8* %dalpha, i8* %beta, i8* %dbeta) {
entry:
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*,i8*,i8*)* @g, metadata !"enzyme_dup", i8* %y, i8* %dy, metadata !"enzyme_dup", i8* %A, i8* %dA, metadata !"enzyme_dup", i8* %x, i8* %dx, metadata !"enzyme_dup", i8* %alpha, i8* %dalpha, metadata !"enzyme_dup", i8* %beta, i8* %dbeta)
  ret void
}

; CHECK: define internal { double*, double* } @augmented_f(i8* noalias %y, i8* %"y'", i8* noalias %A, i8* %"A'", i8* noalias %x, i8* %"x'", i8* noalias %alpha, i8* %"alpha'", i8* noalias %beta, i8* %"beta'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { double*, double* }
; CHECK-NEXT:   %byref. = alloca i64
; CHECK-NEXT:   %byref.10 = alloca i64
; CHECK-NEXT:   %incy = alloca i64, i64 1, align 16
; CHECK-NEXT:   %1 = bitcast i64* %incy to i8*
; CHECK-NEXT:   %incx = alloca i64, i64 1, align 16
; CHECK-NEXT:   %2 = bitcast i64* %incx to i8*
; CHECK-NEXT:   %lda = alloca i64, i64 1, align 16
; CHECK-NEXT:   %3 = bitcast i64* %lda to i8*
; CHECK-NEXT:   %n = alloca i64, i64 1, align 16
; CHECK-NEXT:   %4 = bitcast i64* %n to i8*
; CHECK-NEXT:   %m = alloca i64, i64 1, align 16
; CHECK-NEXT:   %5 = bitcast i64* %m to i8*
; CHECK-NEXT:   %malloccall = alloca i8, i64 1, align 1
; CHECK-NEXT:   %6 = bitcast i8* %5 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %m_p = bitcast i64* %6 to i8*
; CHECK-NEXT:   %7 = bitcast i8* %4 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %n_p = bitcast i64* %7 to i8*
; CHECK-NEXT:   %8 = bitcast i8* %3 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %lda_p = bitcast i64* %8 to i8*
; CHECK-NEXT:   %9 = bitcast i8* %2 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %incx_p = bitcast i64* %9 to i8*
; CHECK-NEXT:   %10 = bitcast i8* %1 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %incy_p = bitcast i64* %10 to i8*
; CHECK-NEXT:   store i8 78, i8* %malloccall, align 1
; CHECK-NEXT:   store i64 4, i64* %6, align 16
; CHECK-NEXT:   store i64 4, i64* %7, align 16
; CHECK-NEXT:   store i64 4, i64* %8, align 16
; CHECK-NEXT:   store i64 2, i64* %9, align 16
; CHECK-NEXT:   store i64 1, i64* %10, align 16
; CHECK-NEXT:   %loaded.trans = load i8, i8* %malloccall
; CHECK-DAG:   %[[r11:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[r12:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %[[r13:.+]] = or i1 %[[r12]], %[[r11]]
; CHECK-NEXT:   %[[r14:.+]] = select i1 %[[r13]], i8* %n_p, i8* %m_p
; CHECK-NEXT:   %[[r15:.+]] = bitcast i8* %[[r14]] to i64*
; CHECK-NEXT:   %[[r16:.+]] = load i64, i64* %[[r15]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[r16]], 8
; CHECK-NEXT:   %malloccall6 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall6 to double*
; CHECK-NEXT:   store i64 1, i64* %byref.
; CHECK-NEXT:   call void @dcopy_64_(i8* %[[r14]], i8* %x, i8* %incx_p, double* %cache.x, i64* %byref.)
; CHECK-NEXT:   %loaded.trans7 = load i8, i8* %malloccall
; CHECK-DAG:   %[[r17:.+]] = icmp eq i8 %loaded.trans7, 78
; CHECK-DAG:   %[[r18:.+]] = icmp eq i8 %loaded.trans7, 110
; CHECK-NEXT:   %[[r19:.+]] = or i1 %[[r18]], %[[r17]]
; CHECK-NEXT:   %[[r20:.+]] = select i1 %[[r19]], i8* %m_p, i8* %n_p
; CHECK-NEXT:   %[[r21:.+]] = bitcast i8* %[[r20]] to i64*
; CHECK-NEXT:   %[[r22:.+]] = load i64, i64* %[[r21]]
; CHECK-NEXT:   %mallocsize8 = mul nuw nsw i64 %[[r22]], 8
; CHECK-NEXT:   %malloccall9 = tail call noalias nonnull i8* @malloc(i64 %mallocsize8)
; CHECK-NEXT:   %cache.y = bitcast i8* %malloccall9 to double*
; CHECK-NEXT:   store i64 1, i64* %byref.10
; CHECK-NEXT:   call void @dcopy_64_(i8* %[[r20]], i8* %y, i8* %incy_p, double* %cache.y, i64* %byref.10)
; CHECK-NEXT:   %23 = insertvalue { double*, double* } undef, double* %cache.x, 0
; CHECK-NEXT:   %24 = insertvalue { double*, double* } %23, double* %cache.y, 1
; CHECK-NEXT:   store { double*, double* } %24, { double*, double* }* %0
; CHECK-NEXT:   call void @dgemv_64_(i8* %malloccall, i8* %m_p, i8* %n_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta, i8* %y, i8* %incy_p, i64 1)
; CHECK-NEXT:   %25 = load { double*, double* }, { double*, double* }* %0
; CHECK-NEXT:   ret { double*, double* } %25
; CHECK-NEXT: }

; CHECK: define internal void @diffef(i8* noalias %y, i8* %"y'", i8* noalias %A, i8* %"A'", i8* noalias %x, i8* %"x'", i8* noalias %alpha, i8* %"alpha'", i8* noalias %beta, i8* %"beta'", { double*, double* }
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.int.one = alloca i64
; CHECK-DAG:    %byref.constant.fp.1.0 = alloca double
; CHECK-DAG:    %byref.constant.char.N = alloca i8, align 1
; CHECK-DAG:    %byref.constant.fp.0.0 = alloca double
; CHECK-NEXT:   %byref.constant.int.1 = alloca i64
; CHECK-NEXT:   %byref.constant.int.17 = alloca i64
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %[[N9:.+]] = alloca i8, align 1
; CHECK-NEXT:   %[[byrefconstantfp1:.+]] = alloca double
; CHECK-NEXT:   %incy = alloca i64, i64 1, align 16
; CHECK-NEXT:   %1 = bitcast i64* %incy to i8*
; CHECK-NEXT:   %incx = alloca i64, i64 1, align 16
; CHECK-NEXT:   %2 = bitcast i64* %incx to i8*
; CHECK-NEXT:   %lda = alloca i64, i64 1, align 16
; CHECK-NEXT:   %3 = bitcast i64* %lda to i8*
; CHECK-NEXT:   %n = alloca i64, i64 1, align 16
; CHECK-NEXT:   %4 = bitcast i64* %n to i8*
; CHECK-NEXT:   %m = alloca i64, i64 1, align 16
; CHECK-NEXT:   %5 = bitcast i64* %m to i8*
; CHECK-NEXT:   %malloccall = alloca i8, i64 1, align 1
; CHECK-NEXT:   %6 = bitcast i8* %5 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %m_p = bitcast i64* %6 to i8*
; CHECK-NEXT:   %7 = bitcast i8* %4 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %n_p = bitcast i64* %7 to i8*
; CHECK-NEXT:   %8 = bitcast i8* %3 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %lda_p = bitcast i64* %8 to i8*
; CHECK-NEXT:   %9 = bitcast i8* %2 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %incx_p = bitcast i64* %9 to i8*
; CHECK-NEXT:   %10 = bitcast i8* %1 to i64*, !enzyme_caststack !5
; CHECK-NEXT:   %incy_p = bitcast i64* %10 to i8*
; CHECK-NEXT:   store i8 78, i8* %malloccall, align 1
; CHECK-NEXT:   store i64 4, i64* %6, align 16
; CHECK-NEXT:   store i64 4, i64* %7, align 16
; CHECK-NEXT:   store i64 4, i64* %8, align 16
; CHECK-NEXT:   store i64 2, i64* %9, align 16
; CHECK-NEXT:   store i64 1, i64* %10, align 16
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %tape.ext.x = extractvalue { double*, double* } %0, 0
; CHECK-NEXT:   %[[r20:.+]] = bitcast double* %tape.ext.x to i8*
; CHECK-NEXT:   %tape.ext.y = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   %[[r21:.+]] = bitcast double* %tape.ext.y to i8*
; CHECK-NEXT:   %tape.ext.y1 = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   %[[r22:.+]] = bitcast double* %tape.ext.y1 to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.int.one

; CHECK-NEXT:   %[[r11:.+]] = bitcast i8* %m_p to i64*
; CHECK-NEXT:   %[[r12:.+]] = load i64, i64* %[[r11]]
; CHECK-NEXT:   %[[r13:.+]] = bitcast i8* %n_p to i64*
; CHECK-NEXT:   %[[r14:.+]] = load i64, i64* %[[r13]]
; CHECK-NEXT:   %loaded.trans = load i8, i8* %malloccall
; CHECK-DAG:   %[[r15:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[r16:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %[[r17:.+]] = or i1 %[[r16]], %[[r15]]
; CHECK-NEXT:   %[[r18:.+]] = select i1 %[[r17]], i64 %[[r12]], i64 %[[r14]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[r18]], 8
; CHECK-NEXT:   %malloccall6 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %[[matAx:.+]] = bitcast i8* %malloccall6 to double*
; CHECK-NEXT:   %[[r19:.+]] = bitcast double* %[[matAx]] to i8*

; CHECK-NEXT:   store double 1.000000e+00, double* %byref.constant.fp.1.0
; CHECK-NEXT:   %fpcast.constant.fp.1.0 = bitcast double* %byref.constant.fp.1.0 to i8*
; CHECK-NEXT:   store i8 78, i8* %byref.constant.char.N, align 1
; CHECK-NEXT:   store double 0.000000e+00, double* %byref.constant.fp.0.0
; CHECK-NEXT:   %fpcast.constant.fp.0.0 = bitcast double* %byref.constant.fp.0.0 to i8*
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.1
; CHECK-NEXT:   call void bitcast (void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64)* @dgemv_64_ to void (i8*, i8*, i8*, i8*, i8*, i8*, i8*, i64*, i8*, i8*, i64*, i64)*)(i8* %malloccall, i8* %m_p, i8* %n_p, i8* %fpcast.constant.fp.1.0, i8* %A, i8* %lda_p, i8* %[[r20]], i64* %byref.int.one, i8* %fpcast.constant.fp.0.0, i8* %[[r19]], i64* %byref.constant.int.1, i64 1)
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %malloccall
; CHECK-DAG:    %[[c1:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-DAG:    %[[c2:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[c3:.+]] = or i1 %[[c2]], %[[c1]]
; CHECK-NEXT:   %[[r34:.+]] = select i1 %[[c3]], i8* %m_p, i8* %n_p
; CHECK-NEXT:   store i64 1, i64* %byref.constant.int.17
; CHECK-NEXT:   %[[r35:.+]] = call fast double @ddot_64_(i8* %[[r34]], i8* %"y'", i8* %incy_p, i8* %[[r19]], i64* %byref.constant.int.17)
; CHECK-NEXT:   %[[r36:.+]] = bitcast i8* %"alpha'" to double*
; CHECK-NEXT:   %[[r37:.+]] = load double, double* %[[r36]]
; CHECK-NEXT:   %[[r38:.+]] = fadd fast double %[[r37]], %[[r35]]
; CHECK-NEXT:   store double %[[r38]], double* %[[r36]]

; CHECK-NEXT:   %[[forfree:.+]] = bitcast double* %22 to i8* 
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[forfree]])

; CHECK-NEXT:   %[[ldrowtrans8:.+]] = load i8, i8* %malloccall, align 1
; CHECK-DAG:   %[[r39:.+]] = icmp eq i8 %[[ldrowtrans9:.+]], 110
; CHECK-DAG:   %[[r40:.+]] = icmp eq i8 %[[ldrowtrans9:.+]], 78
; CHECK-NEXT:   %[[r41:.+]] = or i1 %[[r40]], %[[r39]]
; CHECK-NEXT:   %[[r42:.+]] = select i1 %[[r41]], i8* %"y'", i8* %[[r20]]
; CHECK-NEXT:   %[[intcast:.+]] = bitcast i64* %byref.int.one to i8*
; CHECK-NEXT:   %[[r43:.+]] = select i1 %[[r41]], i8* %incy_p, i8* %[[intcast]]
; CHECK-NEXT:   %[[r44:.+]] = select i1 %[[r41]], i8* %[[r20]], i8* %"y'"
; CHECK-NEXT:   %[[intcast2:.+]] = bitcast i8* %incy_p to i64*
; CHECK-NEXT:   %[[r45:.+]] = select i1 %[[r41]], i64* %byref.int.one, i64* %[[intcast2]]
; CHECK-NEXT:   call void @dger_64_(i8* %m_p, i8* %n_p, i8* %alpha, i8* %[[r42]], i8* %[[r43]], i8* %[[r44]], i64* %[[r45]], i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   %ld.transa = load i8, i8* %malloccall
; CHECK-DAG:    %[[r0:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[r1:.+]] = select i1 %[[r0]], i8 116, i8 78
; CHECK-DAG:    %[[r2:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[r3:.+]] = select i1 %[[r2]], i8 84, i8 %[[r1]]
; CHECK-DAG:    %[[r4:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[r5:.+]] = select i1 %[[r4]], i8 110, i8 %[[r3]]
; CHECK-DAG:    %[[r6:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[r7:.+]] = select i1 %[[r6]], i8 78, i8 %[[r5]]
; CHECK-NEXT:   store i8 %[[r7]], i8* %byref.transpose.transa
; CHECK-NEXT:   store i8 78, i8* %[[N9]], align 1
; CHECK-NEXT:   store double 1.000000e+00, double* %[[byrefconstantfp1]]
; CHECK-NEXT:   %[[fpcast14:.+]] = bitcast double* %[[byrefconstantfp1]] to i8*
; CHECK-NEXT:   call void @dgemv_64_(i8* %byref.transpose.transa, i8* %m_p, i8* %n_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %"y'", i8* %incy_p, i8* %[[fpcast14]], i8* %"x'", i8* %incx_p, i64 1)
; CHECK-NEXT:   %[[ldrowtrans13:.+]] = load i8, i8* %malloccall
; CHECK-DAG:   %[[r39:.+]] = icmp eq i8 %[[ldrowtrans13:.+]], 110
; CHECK-DAG:   %[[r40:.+]] = icmp eq i8 %[[ldrowtrans13:.+]], 78
; CHECK-NEXT:   %[[r41:.+]] = or i1 %[[r40]], %[[r39]]
; CHECK-NEXT:   %[[r42:.+]] = select i1 %[[r41]], i8* %m_p, i8* %n_p
; CHECK-NEXT:   %[[r43:.+]] = call fast double @ddot_64_(i8* %[[r42]], i8* %"y'", i8* %incy_p, i8* %[[r21]], i64* %byref.int.one)
; CHECK-NEXT:   %[[r44:.+]] = bitcast i8* %"beta'" to double*
; CHECK-NEXT:   %[[r45:.+]] = load double, double* %[[r44]]
; CHECK-NEXT:   %[[r46:.+]] = fadd fast double %[[r45]], %[[r43]]
; CHECK-NEXT:   store double %[[r46]], double* %[[r44]]
; CHECK-NEXT:   %[[ldrowtrans14:.+]] = load i8, i8* %malloccall
; CHECK-DAG:   %[[r47:.+]] = icmp eq i8 %[[ldrowtrans14]], 110
; CHECK-DAG:   %[[r48:.+]] = icmp eq i8 %[[ldrowtrans14]], 78
; CHECK-NEXT:   %[[r49:.+]] = or i1 %[[r48]], %[[r47]]
; CHECK-NEXT:   %[[r50:.+]] = select i1 %[[r49]], i8* %m_p, i8* %n_p
; CHECK-NEXT:   call void @dscal_64_(i8* %[[r50]], i8* %beta, i8* %"y'", i8* %incy_p)
; CHECK-NEXT:   %[[r51:.+]] = bitcast double* %tape.ext.x to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[r51]])
; CHECK-NEXT:   %[[r52:.+]] = bitcast double* %tape.ext.y1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[r52]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

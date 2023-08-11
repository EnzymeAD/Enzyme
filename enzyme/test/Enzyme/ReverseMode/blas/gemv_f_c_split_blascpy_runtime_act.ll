;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-blas-copy=1 -enzyme-lapack-copy=1 -enzyme-runtime-activity=1 -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-blas-copy=1  -enzyme-lapack-copy=1 -enzyme-runtime-activity=1 -S | FileCheck %s

; Here we don't transpose the matrix a (78 equals 'N' in ASCII) and we therefore also don't transpose x.
; Therfore the first arg to dcopy is n_p, as opposed to the gemv_transpose test.
;                       trans,                  M,                       N,                     alpha,                  A,    lda,                    x,  , incx,                  beta,                    y,  incy
declare void @dgemv_64_(i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* nocapture readonly, i8* , i8* nocapture readonly, i8*, i8* nocapture readonly, i8* nocapture readonly, i8* , i8* nocapture readonly) 

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
  call void @dgemv_64_(i8* %transa, i8* %m_p, i8* %n_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta, i8* %y, i8* %incy_p)
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
  call void (...) @__enzyme_autodiff(void (i8*,i8*,i8*,i8*,i8*)* @g, metadata !"enzyme_dup", i8* %y, i8* %dy, metadata !"enzyme_dup", i8* %A, i8* %dA, metadata !"enzyme_const", i8* %x, metadata !"enzyme_const", i8* %alpha, metadata !"enzyme_dup", i8* %beta, i8* %dbeta)
  ret void
}

; CHECK: define internal { double*, double* } @augmented_f(i8* noalias %y, i8* %"y'", i8* noalias %A, i8* %"A'", i8* noalias %x, i8* noalias %alpha, i8* noalias %beta, i8* %"beta'")
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
; CHECK-NEXT:   %6 = bitcast i8* %5 to i64*
; CHECK-NEXT:   %m_p = bitcast i64* %6 to i8*
; CHECK-NEXT:   %7 = bitcast i8* %4 to i64*
; CHECK-NEXT:   %n_p = bitcast i64* %7 to i8*
; CHECK-NEXT:   %8 = bitcast i8* %3 to i64*
; CHECK-NEXT:   %lda_p = bitcast i64* %8 to i8*
; CHECK-NEXT:   %9 = bitcast i8* %2 to i64*
; CHECK-NEXT:   %incx_p = bitcast i64* %9 to i8*
; CHECK-NEXT:   %10 = bitcast i8* %1 to i64*
; CHECK-NEXT:   %incy_p = bitcast i64* %10 to i8*
; CHECK-NEXT:   store i8 78, i8* %malloccall, align 1
; CHECK-NEXT:   store i64 4, i64* %6, align 16
; CHECK-NEXT:   store i64 4, i64* %7, align 16
; CHECK-NEXT:   store i64 4, i64* %8, align 16
; CHECK-NEXT:   store i64 2, i64* %9, align 16
; CHECK-NEXT:   store i64 1, i64* %10, align 16
; CHECK-NEXT:   %rt.inactive.A = icmp eq i8* %"A'", %A
; CHECK-NEXT:   %rt.inactive.beta = icmp eq i8* %"beta'", %beta
; CHECK-NEXT:   %rt.inactive.y = icmp eq i8* %"y'", %y
; CHECK-NEXT:   %loaded.trans = load i8, i8* %malloccall
; CHECK-DAG:   %[[i11:.+]] = icmp eq i8 %loaded.trans, 78
; CHECK-DAG:   %[[i12:.+]] = icmp eq i8 %loaded.trans, 110
; CHECK-NEXT:   %[[i13:.+]] = or i1 %[[i12]], %[[i11]]
; CHECK-NEXT:   %[[i14:.+]] = select i1 %[[i13]], i8* %n_p, i8* %m_p
; CHECK-NEXT:   %[[i15:.+]] = bitcast i8* %[[i14]] to i64*
; CHECK-NEXT:   %[[i16:.+]] = load i64, i64* %[[i15]]
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %[[i16]], 8
; CHECK-NEXT:   %malloccall6 = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall6 to double*
; CHECK-NEXT:   store i64 1, i64* %byref.
; CHECK-NEXT:   call void @dcopy_64_(i8* %[[i14]], i8* %x, i8* %incx_p, double* %cache.x, i64* %byref.)
; CHECK-NEXT:   %loaded.trans7 = load i8, i8* %malloccall
; CHECK-DAG:   %[[i17:.+]] = icmp eq i8 %loaded.trans7, 78
; CHECK-DAG:   %[[i18:.+]] = icmp eq i8 %loaded.trans7, 110
; CHECK-NEXT:   %[[i19:.+]] = or i1 %[[i18]], %[[i17]]
; CHECK-NEXT:   %[[i20:.+]] = select i1 %[[i19]], i8* %m_p, i8* %n_p
; CHECK-NEXT:   %[[i21:.+]] = bitcast i8* %[[i20]] to i64*
; CHECK-NEXT:   %[[i22:.+]] = load i64, i64* %[[i21]]
; CHECK-NEXT:   %mallocsize8 = mul nuw nsw i64 %[[i22]], 8
; CHECK-NEXT:   %malloccall9 = tail call noalias nonnull i8* @malloc(i64 %mallocsize8)
; CHECK-NEXT:   %cache.y = bitcast i8* %malloccall9 to double*
; CHECK-NEXT:   store i64 1, i64* %byref.10
; CHECK-NEXT:   call void @dcopy_64_(i8* %20, i8* %y, i8* %incy_p, double* %cache.y, i64* %byref.10)
; CHECK-NEXT:   %23 = insertvalue { double*, double* } undef, double* %cache.x, 0
; CHECK-NEXT:   %24 = insertvalue { double*, double* } %23, double* %cache.y, 1
; CHECK-NEXT:   store { double*, double* } %24, { double*, double* }* %0
; CHECK-NEXT:   call void @dgemv_64_(i8* %malloccall, i8* %m_p, i8* %n_p, i8* %alpha, i8* %A, i8* %lda_p, i8* %x, i8* %incx_p, i8* %beta, i8* %y, i8* %incy_p)
; CHECK-NEXT:   %25 = load { double*, double* }, { double*, double* }* %0
; CHECK-NEXT:   ret { double*, double* } %25
; CHECK-NEXT: }

; CHECK: define internal void @diffef(i8* noalias %y, i8* %"y'", i8* noalias %A, i8* %"A'", i8* noalias %x, i8* noalias %alpha, i8* noalias %beta, i8* %"beta'", { double*, double* } 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca double
; CHECK-NEXT:   %byref.transpose.transa = alloca i8
; CHECK-NEXT:   %byref.int.one = alloca i64
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
; CHECK-NEXT:   %6 = bitcast i8* %5 to i64*
; CHECK-NEXT:   %m_p = bitcast i64* %6 to i8*
; CHECK-NEXT:   %7 = bitcast i8* %4 to i64*
; CHECK-NEXT:   %n_p = bitcast i64* %7 to i8*
; CHECK-NEXT:   %8 = bitcast i8* %3 to i64*
; CHECK-NEXT:   %lda_p = bitcast i64* %8 to i8*
; CHECK-NEXT:   %9 = bitcast i8* %2 to i64*
; CHECK-NEXT:   %incx_p = bitcast i64* %9 to i8*
; CHECK-NEXT:   %10 = bitcast i8* %1 to i64*
; CHECK-NEXT:   %incy_p = bitcast i64* %10 to i8*
; CHECK-NEXT:   store i8 78, i8* %malloccall, align 1
; CHECK-NEXT:   store i64 4, i64* %6, align 16
; CHECK-NEXT:   store i64 4, i64* %7, align 16
; CHECK-NEXT:   store i64 4, i64* %8, align 16
; CHECK-NEXT:   store i64 2, i64* %9, align 16
; CHECK-NEXT:   store i64 1, i64* %10, align 16
; CHECK-NEXT:   %rt.inactive.A = icmp eq i8* %"A'", %A
; CHECK-NEXT:   %rt.inactive.beta = icmp eq i8* %"beta'", %beta
; CHECK-NEXT:   %rt.inactive.y = icmp eq i8* %"y'", %y
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   %tape.ext.x = extractvalue { double*, double* } %0, 0
; CHECK-NEXT:   %[[i11:.+]] = bitcast double* %tape.ext.x to i8*
; CHECK-NEXT:   %tape.ext.y = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   %[[i12:.+]] = bitcast double* %tape.ext.y to i8*
; CHECK-NEXT:   %tape.ext.y1 = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   %[[i13:.+]] = bitcast double* %tape.ext.y1 to i8*
; CHECK-NEXT:   %ld.transa = load i8, i8* %malloccall
; CHECK-DAG:    %[[r0:.+]] = icmp eq i8 %ld.transa, 110
; CHECK-DAG:    %[[r1:.+]] = select i1 %[[r0]], i8 116, i8 0
; CHECK-DAG:    %[[r2:.+]] = icmp eq i8 %ld.transa, 78
; CHECK-DAG:    %[[r3:.+]] = select i1 %[[r2]], i8 84, i8 %[[r1]]
; CHECK-DAG:    %[[r4:.+]] = icmp eq i8 %ld.transa, 116
; CHECK-DAG:    %[[r5:.+]] = select i1 %[[r4]], i8 110, i8 %[[r3]]
; CHECK-DAG:    %[[r6:.+]] = icmp eq i8 %ld.transa, 84
; CHECK-DAG:    %[[r7:.+]] = select i1 %[[r6]], i8 78, i8 %[[r5]]
; CHECK-NEXT:   store i8 %[[r7]], i8* %byref.transpose.transa
; CHECK-NEXT:   store i64 1, i64* %byref.int.one
; CHECK-NEXT:   %intcast.int.one = bitcast i64* %byref.int.one to i8*
; CHECK-NEXT:   br i1 %rt.inactive.A, label %invertentry.A.done, label %invertentry.A.active

; CHECK: invertentry.A.active:                             ; preds = %invertentry
; CHECK-NEXT:   call void @dger_64_(i8* %m_p, i8* %n_p, i8* %alpha, i8* %"y'", i8* %incy_p, i8* %[[i11]], i8* %intcast.int.one, i8* %"A'", i8* %lda_p)
; CHECK-NEXT:   br label %invertentry.A.done

; CHECK: invertentry.A.done:                               ; preds = %invertentry.A.active, %invertentry
; CHECK-NEXT:   br i1 %rt.inactive.beta, label %invertentry.beta.done, label %invertentry.beta.active

; CHECK: invertentry.beta.active:                          ; preds = %invertentry.A.done
; CHECK-NEXT:   %ld.row.trans = load i8, i8* %byref.transpose.transa
; CHECK-DAG:   %[[r39:.+]] = icmp eq i8 %ld.row.trans, 110
; CHECK-DAG:   %[[r40:.+]] = icmp eq i8 %ld.row.trans, 78
; CHECK-NEXT:   %[[r41:.+]] = or i1 %[[r40]], %[[r39]]
; CHECK-NEXT:   %[[r42:.+]] = select i1 %[[r41]], i8* %m_p, i8* %n_p
; CHECK-NEXT:   %[[r43:.+]] = call fast double @ddot_64_(i8* %[[r42]], i8* %"y'", i8* %incy_p, i8* %[[i12]], i8* %intcast.int.one)
; CHECK-NEXT:   %[[r44:.+]] = bitcast i8* %"beta'" to double*
; CHECK-NEXT:   %[[r45:.+]] = load double, double* %[[r44]]
; CHECK-NEXT:   %[[r46:.+]] = fadd fast double %[[r45]], %[[r43]]
; CHECK-NEXT:   store double %[[r46]], double* %[[r44]]
; CHECK-NEXT:   br label %invertentry.beta.done

; CHECK: invertentry.beta.done:                            ; preds = %invertentry.beta.active, %invertentry.A.done
; CHECK-NEXT:   br i1 %rt.inactive.y, label %invertentry.y.done, label %invertentry.y.active

; CHECK: invertentry.y.active:                             ; preds = %invertentry.beta.done
; CHECK-NEXT:   %ld.row.trans2 = load i8, i8* %byref.transpose.transa
; CHECK-DAG:   %[[r47:.+]] = icmp eq i8 %ld.row.trans2, 110
; CHECK-DAG:   %[[r48:.+]] = icmp eq i8 %ld.row.trans2, 78
; CHECK-NEXT:   %[[r49:.+]] = or i1 %[[r48]], %[[r47]]
; CHECK-NEXT:   %[[r50:.+]] = select i1 %[[r49]], i8* %m_p, i8* %n_p
; CHECK-NEXT:   call void @dscal_64_(i8* %[[r50]], i8* %beta, i8* %"y'", i8* %incy_p)
; CHECK-NEXT:   br label %invertentry.y.done

; CHECK: invertentry.y.done:                               ; preds = %invertentry.y.active, %invertentry.beta.done
; CHECK-NEXT:   %[[r51:.+]] = bitcast double* %tape.ext.x to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[r51]])
; CHECK-NEXT:   %[[r52:.+]] = bitcast double* %tape.ext.y1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %[[r52]])
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

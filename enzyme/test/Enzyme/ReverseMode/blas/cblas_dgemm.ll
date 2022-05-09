;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc) {
entry:
  tail call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc)
  ret void
}

declare void @cblas_dgemm(i32, i32, i32, i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)

define void @wrapperModA(i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc) {
entry:
  tail call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc)
  store double 0.000000e+00, double* %a, align 8
  ret void
}

define void @wrapperModB(i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc) {
entry:
  tail call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc)
  store double 0.000000e+00, double* %b, align 8
  ret void
}

define void @wrapperModABC(i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc) {
entry:
  tail call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc)
  store double 0.000000e+00, double* %c, align 8
  store double 0.000000e+00, double* %b, align 8
  store double 0.000000e+00, double* %a, align 8
  ret void
}

define void @active(i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb, double %beta, double* %c, double* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)* @wrapper to i8*), i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb, double %beta, double* %c, double* %_c, i32 %ldc)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, i32, i32, double, double*, double*, i32, double*, double*, i32, double, double*, double*, i32)

define void @activeModA(i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb, double %beta, double* %c, double* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)* @wrapperModA to i8*), i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb, double %beta, double* %c, double* %_c, i32 %ldc)
  ret void
}

define void @activeModB(i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb, double %beta, double* %c, double* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)* @wrapperModB to i8*), i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb, double %beta, double* %c, double* %_c, i32 %ldc)
  ret void
}

define void @activeModABC(i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb, double %beta, double* %c, double* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)* @wrapperModABC to i8*), i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %_a, i32 %lda, double* %b, double* %_b, i32 %ldb, double %beta, double* %c, double* %_c, i32 %ldc)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[active:.+]](

;CHECK: define void @activeModA
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[activeModA:.+]](

;CHECK: define void @activeModB
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[activeModB:.+]](

;CHECK: define void @activeModABC
;CHECK-NEXT: entry
;CHECK-NEXT: call { double, double } @[[activeModABC:.+]](

;CHECK:define internal { double, double } @[[active]](i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %"a'", i32 %lda, double* %b, double* %"b'", i32 %ldb, double %beta, double* %c, double* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc)
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 111, i32 112, i32 %m, i32 %k, i32 %n, double %alpha, double* %"c'", i32 %ldc, double* %b, i32 %ldb, double 1.000000e+00, double* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 112, i32 111, i32 %k, i32 %n, i32 %m, double %alpha, double* %a, i32 %lda, double* %"c'", i32 %ldc, double 1.000000e+00, double* %"b'", i32 %ldb)
;CHECK-NEXT:  call void @__enzyme_memcpy_doublematrix_scal(i32 102, i32 %m, i32 %k, double %beta, double* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { double, double } zeroinitializer
;CHECK-NEXT:}

;CHECK:define internal { double, double } @[[activeModA]](i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %"a'", i32 %lda, double* %b, double* %"b'", i32 %ldb, double %beta, double* %c, double* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = mul i32 %m, %k
;CHECK-NEXT:  %1 = zext i32 %0 to i64
;CHECK-NEXT:  %mallocsize = mul i64 %1, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %2 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0matrix(double* %2, double* %a, i32 %m, i32 %k, i32 %lda, i32 102)
;CHECK-NEXT:  %3 = mul i32 %k, %n
;CHECK-NEXT:  %4 = zext i32 %3 to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %4, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %5 = bitcast i8* %malloccall2 to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0matrix(double* %5, double* %b, i32 %k, i32 %n, i32 %ldb, i32 102)
;CHECK-NEXT:  tail call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc)
;CHECK-NEXT:  store double 0.000000e+00, double* %a, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"a'", align 8
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 111, i32 112, i32 %m, i32 %k, i32 %n, double %alpha, double* %"c'", i32 %ldc, double* %5, i32 %ldb, double 1.000000e+00, double* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 112, i32 111, i32 %k, i32 %n, i32 %m, double %alpha, double* %2, i32 %lda, double* %"c'", i32 %ldc, double 1.000000e+00, double* %"b'", i32 %ldb)
;CHECK-NEXT:  call void @__enzyme_memcpy_doublematrix_scal(i32 102, i32 %m, i32 %k, double %beta, double* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { double, double } zeroinitializer
;CHECK-NEXT:}

;CHECK:define internal { double, double } @[[activeModB]](i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %"a'", i32 %lda, double* %b, double* %"b'", i32 %ldb, double %beta, double* %c, double* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = mul i32 %m, %k
;CHECK-NEXT:  %1 = zext i32 %0 to i64
;CHECK-NEXT:  %mallocsize = mul i64 %1, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %2 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0matrix(double* %2, double* %a, i32 %m, i32 %k, i32 %lda, i32 102)
;CHECK-NEXT:  %3 = mul i32 %k, %n
;CHECK-NEXT:  %4 = zext i32 %3 to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %4, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %5 = bitcast i8* %malloccall2 to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0matrix(double* %5, double* %b, i32 %k, i32 %n, i32 %ldb, i32 102)
;CHECK-NEXT:  tail call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc)
;CHECK-NEXT:  store double 0.000000e+00, double* %b, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"b'", align 8
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 111, i32 112, i32 %m, i32 %k, i32 %n, double %alpha, double* %"c'", i32 %ldc, double* %5, i32 %ldb, double 1.000000e+00, double* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 112, i32 111, i32 %k, i32 %n, i32 %m, double %alpha, double* %2, i32 %lda, double* %"c'", i32 %ldc, double 1.000000e+00, double* %"b'", i32 %ldb)
;CHECK-NEXT:  call void @__enzyme_memcpy_doublematrix_scal(i32 102, i32 %m, i32 %k, double %beta, double* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { double, double } zeroinitializer
;CHECK-NEXT:}

;CHECK:define internal { double, double } @[[activeModABC]](i32 %m, i32 %n, i32 %k, double %alpha, double* %a, double* %"a'", i32 %lda, double* %b, double* %"b'", i32 %ldb, double %beta, double* %c, double* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = mul i32 %m, %k
;CHECK-NEXT:  %1 = zext i32 %0 to i64
;CHECK-NEXT:  %mallocsize = mul i64 %1, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %2 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0matrix(double* %2, double* %a, i32 %m, i32 %k, i32 %lda, i32 102)
;CHECK-NEXT:  %3 = mul i32 %k, %n
;CHECK-NEXT:  %4 = zext i32 %3 to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %4, ptrtoint (double* getelementptr (double, double* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %5 = bitcast i8* %malloccall2 to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0matrix(double* %5, double* %b, i32 %k, i32 %n, i32 %ldb, i32 102)
;CHECK-NEXT:  tail call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, double %alpha, double* %a, i32 %lda, double* %b, i32 %ldb, double %beta, double* %c, i32 %ldc)
;CHECK-NEXT:  store double 0.000000e+00, double* %c, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %b, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %a, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"a'", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"b'", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"c'", align 8
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 111, i32 112, i32 %m, i32 %k, i32 %n, double %alpha, double* %"c'", i32 %ldc, double* %5, i32 %ldb, double 1.000000e+00, double* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 112, i32 111, i32 %k, i32 %n, i32 %m, double %alpha, double* %2, i32 %lda, double* %"c'", i32 %ldc, double 1.000000e+00, double* %"b'", i32 %ldb)
;CHECK-NEXT:  call void @__enzyme_memcpy_doublematrix_scal(i32 102, i32 %m, i32 %k, double %beta, double* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { double, double } zeroinitializer
;CHECK-NEXT:}

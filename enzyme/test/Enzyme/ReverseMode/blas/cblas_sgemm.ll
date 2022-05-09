;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

define void @wrapper(i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc) {
entry:
  tail call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc)
  ret void
}

declare void @cblas_sgemm(i32, i32, i32, i32, i32, i32, float, float*, i32, float*, i32, float, float*, i32)

define void @wrapperModA(i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc) {
entry:
  tail call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc)
  store float 0.000000e+00, float* %a, align 4
  ret void
}

define void @wrapperModB(i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc) {
entry:
  tail call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc)
  store float 0.000000e+00, float* %b, align 4
  ret void
}

define void @wrapperModABC(i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc) {
entry:
  tail call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc)
  store float 0.000000e+00, float* %c, align 4
  store float 0.000000e+00, float* %b, align 4
  store float 0.000000e+00, float* %a, align 4
  ret void
}

define void @active(i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb, float %beta, float* %c, float* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, i32, float, float*, i32, float*, i32, float, float*, i32)* @wrapper to i8*), i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb, float %beta, float* %c, float* %_c, i32 %ldc)
  ret void
}

declare void @__enzyme_autodiff(i8*, i32, i32, i32, float, float*, float*, i32, float*, float*, i32, float, float*, float*, i32)

define void @activeModA(i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb, float %beta, float* %c, float* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, i32, float, float*, i32, float*, i32, float, float*, i32)* @wrapperModA to i8*), i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb, float %beta, float* %c, float* %_c, i32 %ldc)
  ret void
}

define void @activeModB(i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb, float %beta, float* %c, float* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, i32, float, float*, i32, float*, i32, float, float*, i32)* @wrapperModB to i8*), i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb, float %beta, float* %c, float* %_c, i32 %ldc)
  ret void
}

define void @activeModABC(i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb, float %beta, float* %c, float* %_c, i32 %ldc) {
entry:
  tail call void @__enzyme_autodiff(i8* bitcast (void (i32, i32, i32, float, float*, i32, float*, i32, float, float*, i32)* @wrapperModABC to i8*), i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %_a, i32 %lda, float* %b, float* %_b, i32 %ldb, float %beta, float* %c, float* %_c, i32 %ldc)
  ret void
}

;CHECK: define void @active
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[active:.+]](

;CHECK: define void @activeModA
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[activeModA:.+]](

;CHECK: define void @activeModB
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[activeModB:.+]](

;CHECK: define void @activeModABC
;CHECK-NEXT: entry
;CHECK-NEXT: call { float, float } @[[activeModABC:.+]](

;CHECK:define internal { float, float } @[[active]](i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %"a'", i32 %lda, float* %b, float* %"b'", i32 %ldb, float %beta, float* %c, float* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc)
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 111, i32 112, i32 %m, i32 %k, i32 %n, float %alpha, float* %"c'", i32 %ldc, float* %b, i32 %ldb, float 1.000000e+00, float* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 112, i32 111, i32 %k, i32 %n, i32 %m, float %alpha, float* %a, i32 %lda, float* %"c'", i32 %ldc, float 1.000000e+00, float* %"b'", i32 %ldb)
;CHECK-NEXT:  call void @__enzyme_memcpy_floatmatrix_scal(i32 102, i32 %m, i32 %k, float %beta, float* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { float, float } zeroinitializer
;CHECK-NEXT:}

;CHECK:define internal { float, float } @[[activeModA]](i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %"a'", i32 %lda, float* %b, float* %"b'", i32 %ldb, float %beta, float* %c, float* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = mul i32 %m, %k
;CHECK-NEXT:  %1 = zext i32 %0 to i64
;CHECK-NEXT:  %mallocsize = mul i64 %1, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %2 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0matrix(float* %2, float* %a, i32 %m, i32 %k, i32 %lda, i32 102)
;CHECK-NEXT:  %3 = mul i32 %k, %n
;CHECK-NEXT:  %4 = zext i32 %3 to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %4, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %5 = bitcast i8* %malloccall2 to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0matrix(float* %5, float* %b, i32 %k, i32 %n, i32 %ldb, i32 102)
;CHECK-NEXT:  tail call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc)
;CHECK-NEXT:  store float 0.000000e+00, float* %a, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"a'", align 4
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 111, i32 112, i32 %m, i32 %k, i32 %n, float %alpha, float* %"c'", i32 %ldc, float* %5, i32 %ldb, float 1.000000e+00, float* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 112, i32 111, i32 %k, i32 %n, i32 %m, float %alpha, float* %2, i32 %lda, float* %"c'", i32 %ldc, float 1.000000e+00, float* %"b'", i32 %ldb)
;CHECK-NEXT:  call void @__enzyme_memcpy_floatmatrix_scal(i32 102, i32 %m, i32 %k, float %beta, float* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { float, float } zeroinitializer
;CHECK-NEXT:}

;CHECK:define internal { float, float } @[[activeModB]](i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %"a'", i32 %lda, float* %b, float* %"b'", i32 %ldb, float %beta, float* %c, float* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = mul i32 %m, %k
;CHECK-NEXT:  %1 = zext i32 %0 to i64
;CHECK-NEXT:  %mallocsize = mul i64 %1, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %2 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0matrix(float* %2, float* %a, i32 %m, i32 %k, i32 %lda, i32 102)
;CHECK-NEXT:  %3 = mul i32 %k, %n
;CHECK-NEXT:  %4 = zext i32 %3 to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %4, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %5 = bitcast i8* %malloccall2 to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0matrix(float* %5, float* %b, i32 %k, i32 %n, i32 %ldb, i32 102)
;CHECK-NEXT:  tail call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc)
;CHECK-NEXT:  store float 0.000000e+00, float* %b, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"b'", align 4
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 111, i32 112, i32 %m, i32 %k, i32 %n, float %alpha, float* %"c'", i32 %ldc, float* %5, i32 %ldb, float 1.000000e+00, float* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 112, i32 111, i32 %k, i32 %n, i32 %m, float %alpha, float* %2, i32 %lda, float* %"c'", i32 %ldc, float 1.000000e+00, float* %"b'", i32 %ldb)
;CHECK-NEXT:  call void @__enzyme_memcpy_floatmatrix_scal(i32 102, i32 %m, i32 %k, float %beta, float* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { float, float } zeroinitializer
;CHECK-NEXT:}

;CHECK:define internal { float, float } @[[activeModABC]](i32 %m, i32 %n, i32 %k, float %alpha, float* %a, float* %"a'", i32 %lda, float* %b, float* %"b'", i32 %ldb, float %beta, float* %c, float* %"c'", i32 %ldc)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %0 = mul i32 %m, %k
;CHECK-NEXT:  %1 = zext i32 %0 to i64
;CHECK-NEXT:  %mallocsize = mul i64 %1, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 %mallocsize)
;CHECK-NEXT:  %2 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0matrix(float* %2, float* %a, i32 %m, i32 %k, i32 %lda, i32 102)
;CHECK-NEXT:  %3 = mul i32 %k, %n
;CHECK-NEXT:  %4 = zext i32 %3 to i64
;CHECK-NEXT:  %mallocsize1 = mul i64 %4, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 %mallocsize1)
;CHECK-NEXT:  %5 = bitcast i8* %malloccall2 to float*
;CHECK-NEXT:  call void @__enzyme_memcpy_floatda0sa0matrix(float* %5, float* %b, i32 %k, i32 %n, i32 %ldb, i32 102)
;CHECK-NEXT:  tail call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 %m, i32 %n, i32 %k, float %alpha, float* %a, i32 %lda, float* %b, i32 %ldb, float %beta, float* %c, i32 %ldc)
;CHECK-NEXT:  store float 0.000000e+00, float* %c, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %b, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %a, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"a'", align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"b'", align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %"c'", align 4
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 111, i32 112, i32 %m, i32 %k, i32 %n, float %alpha, float* %"c'", i32 %ldc, float* %5, i32 %ldb, float 1.000000e+00, float* %"a'", i32 %lda)
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 112, i32 111, i32 %k, i32 %n, i32 %m, float %alpha, float* %2, i32 %lda, float* %"c'", i32 %ldc, float 1.000000e+00, float* %"b'", i32 %ldb)
;CHECK-NEXT:  call void @__enzyme_memcpy_floatmatrix_scal(i32 102, i32 %m, i32 %k, float %beta, float* %"c'", i32 %ldc)
;CHECK-NEXT:  ret { float, float } zeroinitializer
;CHECK-NEXT:}

;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern double __enzyme_autodiff(void *, double *, double *, double *, double *, double *, double*, double, double);
;
;void g(double *restrict A, double *restrict B, double *C, double alpha, double beta) {
;    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, 2, 4, 3, alpha, A, 2, B, 3, beta, C, 4);
;}
;
;int main() {
;    double A[] = {1, 4,
;                 2, 5,
;                 3, 6};
;    double B[] = {21, 0.3, 0.7,
;                 0.9, 1, 26,
;                 30, 31, 32,
;                 33, 34, 35};
;    double C[] = { 0.00, 0.00, 0.0, 0.0,
;                0.00, 0.00, 0.0, 0.0};
;    double A1[] = {0, 0, 0,
;                  0, 0, 0};
;    double B1[] = {0, 0, 0, 0,
;                  0, 0, 0, 0,
;                  0, 0, 0, 0};
;    double C1[] = {1, 1, 1, 1,
;                  1, 1, 1, 1};
;    __enzyme_autodiff((void*)g, A, A1, B, B1, C, C1, 2.0, 2.0);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define dso_local void @g(double* %alpha, double* %A, double* %x, double* %beta, double* %y) {
entry:
  %trans = alloca i32, align 4
  %M = alloca i32, align 4
  %N = alloca i32, align 4
  %lda = alloca i32, align 4
  %incx = alloca i32, align 4
  %incy = alloca i32, align 4

  call void @dgemv(i32* %trans, i32* %M, i32* %N, double* %alpha, double* %A, i32* %lda, double* %x, i32* %incx, double* %beta, double* %y, i32* %incy)
  ret void
}

declare dso_local void @dgemv(i32*, i32*, i32*, double*, double*, i32*, double*, i32*, double*, double*, i32*)

; declare dso_local void @dgemm(i32, i32, i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)

define dso_local i32 @main(double* %A, double* %dA, double* %x, double* %dx, double* %y, double* %dy, double* %dalpha, double* %dbeta) {
  %alpha = alloca double, align 8
  %beta = alloca double, align 8
  store double 2.000000e+00, double* %alpha
  store double 2.000000e+00, double* %beta 
  %call = call double @__enzyme_autodiff(i8* bitcast (void (double*, double*, double*, double*, double*)* @g to i8*), 
  double* %A, double* %dA, double* %x, double* %dx, double* %y, double* %dy, double* %alpha, double* %dalpha , double* %beta, double* %dbeta)
  ret i32 0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*)

;CHECK:define internal { double, double } @diffeg(double* noalias %A, double* %"A'", double* noalias %B, double* %"B'", double* %C, double* %"C'", double %alpha, double %beta) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  call void @cblas_dgemm(i32 101, i32 112, i32 112, i32 2, i32 4, i32 3, double %alpha, double* nocapture readonly %A, i32 2, double* nocapture readonly %B, i32 3, double %beta, double* %C, i32 4)
;CHECK-NEXT:  call void @cblas_dgemm(i32 101, i32 111, i32 111, i32 2, i32 3, i32 4, double %alpha, double* nocapture readonly %"C'", i32 4, double* nocapture readonly %B, i32 3, double 1.000000e+00, double* %"A'", i32 3)
;CHECK-NEXT:  call void @cblas_dgemm(i32 101, i32 111, i32 111, i32 3, i32 4, i32 2, double %alpha, double* nocapture readonly %A, i32 2, double* nocapture readonly %"C'", i32 4, double 1.000000e+00, double* %"B'", i32 4)
;CHECK-NEXT:  call void @cblas_dscal(i32 8, double %beta, double* %"C'", i32 1)
;CHECK-NEXT:  ret { double, double } zeroinitializer
;CHECK-NEXT:}

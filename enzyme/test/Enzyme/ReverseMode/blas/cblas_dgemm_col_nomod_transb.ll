;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern double __enzyme_autodiff(void *, double *, double *, double *, double *, double *, double*, double, double);
;
;void g(double *restrict A, double *restrict B, double *C, double alpha, double beta) {
;    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 4, 3, 2, alpha, A, 4, B, 3, beta, C, 4);
;}
;
;int main() {
;    double A[] = {0.11, 0.12, 0.13, 0.14,
;                 0.21, 0.22, 0.23, 0.24};
;    double B[] = {1011, 1021, 1031,
;                 1012, 1022, 1032};
;    double C[] = {0.00, 0.00, 0.00, 0.00,
;                 0.00, 0.00, 0.00, 0.00,
;                 0.00, 0.00, 0.00, 0.00};
;    double A1[] = {0, 0, 0, 0, 0, 0, 0, 0};
;    double B1[] = {0, 0, 0, 0, 0, 0};
;    double C1[] = {1, 1, 1, 1,
;                  1, 1, 1, 1,
;                  1, 1, 1, 1};
;    __enzyme_autodiff((void*)g, A, A1, B, B1, C, C1, 2.0, 3.0);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.A = private unnamed_addr constant [8 x double] [double 1.100000e-01, double 1.200000e-01, double 1.300000e-01, double 1.400000e-01, double 2.100000e-01, double 2.200000e-01, double 2.300000e-01, double 2.400000e-01], align 16
@__const.main.C1 = private unnamed_addr constant [12 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], align 16

define dso_local void @g(double* noalias %A, double* noalias %B, double* %C, double %alpha, double %beta) {
entry:
  call void @cblas_dgemm(i32 102, i32 111, i32 112, i32 4, i32 3, i32 2, double %alpha, double* %A, i32 4, double* %B, i32 3, double %beta, double* %C, i32 4)
  ret void
}

declare dso_local void @cblas_dgemm(i32, i32, i32, i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)

define dso_local i32 @main() {
entry:
  %A = alloca [8 x double], align 16
  %B = alloca [6 x double], align 16
  %C = alloca [12 x double], align 16
  %A1 = alloca [8 x double], align 16
  %B1 = alloca [6 x double], align 16
  %C1 = alloca [12 x double], align 16
  %0 = bitcast [8 x double]* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast ([8 x double]* @__const.main.A to i8*), i64 64, i1 false)
  %1 = bitcast [6 x double]* %B to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %1, i8 0, i64 48, i1 false)
  %2 = bitcast i8* %1 to [6 x double]*
  %3 = getelementptr inbounds [6 x double], [6 x double]* %2, i32 0, i32 0
  store double 1.011000e+03, double* %3, align 16
  %4 = getelementptr inbounds [6 x double], [6 x double]* %2, i32 0, i32 1
  store double 1.021000e+03, double* %4, align 8
  %5 = getelementptr inbounds [6 x double], [6 x double]* %2, i32 0, i32 2
  store double 1.031000e+03, double* %5, align 16
  %6 = getelementptr inbounds [6 x double], [6 x double]* %2, i32 0, i32 3
  store double 1.012000e+03, double* %6, align 8
  %7 = getelementptr inbounds [6 x double], [6 x double]* %2, i32 0, i32 4
  store double 1.022000e+03, double* %7, align 16
  %8 = getelementptr inbounds [6 x double], [6 x double]* %2, i32 0, i32 5
  store double 1.032000e+03, double* %8, align 8
  %9 = bitcast [12 x double]* %C to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %9, i8 0, i64 96, i1 false)
  %10 = bitcast [8 x double]* %A1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %10, i8 0, i64 64, i1 false)
  %11 = bitcast [6 x double]* %B1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %11, i8 0, i64 48, i1 false)
  %12 = bitcast [12 x double]* %C1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %12, i8* align 16 bitcast ([12 x double]* @__const.main.C1 to i8*), i64 96, i1 false)
  %arraydecay = getelementptr inbounds [8 x double], [8 x double]* %A, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [8 x double], [8 x double]* %A1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [6 x double], [6 x double]* %B, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [6 x double], [6 x double]* %B1, i32 0, i32 0
  %arraydecay4 = getelementptr inbounds [12 x double], [12 x double]* %C, i32 0, i32 0
  %arraydecay5 = getelementptr inbounds [12 x double], [12 x double]* %C1, i32 0, i32 0
  %call = call double @__enzyme_autodiff(i8* bitcast (void (double*, double*, double*, double, double)* @g to i8*), double* %arraydecay, double* %arraydecay1, double* %arraydecay2, double* %arraydecay3, double* %arraydecay4, double* %arraydecay5, double 2.000000e+00, double 3.000000e+00)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*, double*, double*, double, double)

;CHECK:define internal { double, double } @diffeg(double* noalias %A, double* %"A'", double* noalias %B, double* %"B'", double* %C, double* %"C'", double %alpha, double %beta) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 111, i32 112, i32 4, i32 3, i32 2, double %alpha, double* nocapture readonly %A, i32 4, double* nocapture readonly %B, i32 3, double %beta, double* %C, i32 4)
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 111, i32 111, i32 4, i32 2, i32 3, double %alpha, double* nocapture readonly %"C'", i32 4, double* nocapture readonly %B, i32 3, double 1.000000e+00, double* %"A'", i32 4)
;CHECK-NEXT:  call void @cblas_dgemm(i32 102, i32 112, i32 111, i32 2, i32 3, i32 4, double %alpha, double* nocapture readonly %A, i32 4, double* nocapture readonly %"C'", i32 4, double 1.000000e+00, double* %"B'", i32 2)
;CHECK-NEXT:  call void @cblas_dscal(i32 12, double %beta, double* %"C'", i32 1)
;CHECK-NEXT:  ret { double, double } zeroinitializer
;CHECK-NEXT:}

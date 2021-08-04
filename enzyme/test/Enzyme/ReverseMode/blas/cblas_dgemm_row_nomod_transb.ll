;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern double __enzyme_autodiff(void *, double *, double *, double *, double *, double *, double*, double, double);
;
;void g(double *restrict A, double *restrict B, double *C, double alpha, double beta) {
;    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 2, 4, 3, alpha, A, 3, B, 3, beta, C, 4);
;}
;
;int main() {
;    double A[] = { 1, 2, 3,
;                4, 5, 6};
;    double B[] = {21, 0.3, 0.7,
;                 0.9, 1, 26,
;                 30, 31, 32,
;                 33, 34, 35};
;    double C[] = { 0.00, 0.00, 0.0, 0.0,
;                0.00, 0.00, 0.0, 0.0};
;    double A1[] = {0, 0, 0,
;                  0, 0, 0};
;    double B1[] = {0, 0, 0,
;                  0, 0, 0,
;                  0, 0, 0,
;                  0, 0, 0};
;    double C1[] = {1, 1, 1, 1,
;                  1, 1, 1, 1};
;    __enzyme_autodiff((void*)g, A, A1, B, B1, C, C1, 2.0, 2.0);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.B = private unnamed_addr constant [12 x double] [double 2.100000e+01, double 3.000000e-01, double 0x3FE6666666666666, double 9.000000e-01, double 1.000000e+00, double 2.600000e+01, double 3.000000e+01, double 3.100000e+01, double 3.200000e+01, double 3.300000e+01, double 3.400000e+01, double 3.500000e+01], align 16
@__const.main.C1 = private unnamed_addr constant [8 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], align 16

define dso_local void @g(double* noalias %A, double* noalias %B, double* %C, double %alpha, double %beta) {
entry:
  call void @cblas_dgemm(i32 101, i32 111, i32 112, i32 2, i32 4, i32 3, double %alpha, double* %A, i32 3, double* %B, i32 3, double %beta, double* %C, i32 4)
  ret void
}

declare dso_local void @cblas_dgemm(i32, i32, i32, i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)

define dso_local i32 @main() {
entry:
  %A = alloca [6 x double], align 16
  %B = alloca [12 x double], align 16
  %C = alloca [8 x double], align 16
  %A1 = alloca [6 x double], align 16
  %B1 = alloca [12 x double], align 16
  %C1 = alloca [8 x double], align 16
  %0 = bitcast [6 x double]* %A to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %0, i8 0, i64 48, i1 false)
  %1 = bitcast i8* %0 to [6 x double]*
  %2 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 0
  store double 1.000000e+00, double* %2, align 16
  %3 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 1
  store double 2.000000e+00, double* %3, align 8
  %4 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 2
  store double 3.000000e+00, double* %4, align 16
  %5 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 3
  store double 4.000000e+00, double* %5, align 8
  %6 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 4
  store double 5.000000e+00, double* %6, align 16
  %7 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 5
  store double 6.000000e+00, double* %7, align 8
  %8 = bitcast [12 x double]* %B to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %8, i8* align 16 bitcast ([12 x double]* @__const.main.B to i8*), i64 96, i1 false)
  %9 = bitcast [8 x double]* %C to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %9, i8 0, i64 64, i1 false)
  %10 = bitcast [6 x double]* %A1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %10, i8 0, i64 48, i1 false)
  %11 = bitcast [12 x double]* %B1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %11, i8 0, i64 96, i1 false)
  %12 = bitcast [8 x double]* %C1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %12, i8* align 16 bitcast ([8 x double]* @__const.main.C1 to i8*), i64 64, i1 false)
  %arraydecay = getelementptr inbounds [6 x double], [6 x double]* %A, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [6 x double], [6 x double]* %A1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [12 x double], [12 x double]* %B, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [12 x double], [12 x double]* %B1, i32 0, i32 0
  %arraydecay4 = getelementptr inbounds [8 x double], [8 x double]* %C, i32 0, i32 0
  %arraydecay5 = getelementptr inbounds [8 x double], [8 x double]* %C1, i32 0, i32 0
  %call = call double @__enzyme_autodiff(i8* bitcast (void (double*, double*, double*, double, double)* @g to i8*), double* %arraydecay, double* %arraydecay1, double* %arraydecay2, double* %arraydecay3, double* %arraydecay4, double* %arraydecay5, double 2.000000e+00, double 2.000000e+00)
  ret i32 0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*, double*, double*, double, double)

;CHECK:define internal { double, double } @diffeg(double* noalias %A, double* %"A'", double* noalias %B, double* %"B'", double* %C, double* %"C'", double %alpha, double %beta) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  call void @cblas_dgemm(i32 101, i32 111, i32 112, i32 2, i32 4, i32 3, double %alpha, double* nocapture readonly %A, i32 3, double* nocapture readonly %B, i32 3, double %beta, double* %C, i32 4)
;CHECK-NEXT:  call void @cblas_dgemm(i32 101, i32 111, i32 111, i32 2, i32 3, i32 4, double %alpha, double* nocapture readonly %"C'", i32 4, double* nocapture readonly %B, i32 3, double 1.000000e+00, double* %"A'", i32 3)
;CHECK-NEXT:  call void @cblas_dgemm(i32 101, i32 112, i32 111, i32 3, i32 4, i32 2, double %alpha, double* nocapture readonly %A, i32 3, double* nocapture readonly %"C'", i32 4, double 1.000000e+00, double* %"B'", i32 4)
;CHECK-NEXT:  call void @cblas_dscal(i32 8, double %beta, double* %"C'", i32 1)
;CHECK-NEXT:  ret { double, double } zeroinitializer
;CHECK-NEXT:}

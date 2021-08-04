;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern float __enzyme_autodiff(void *, float *, float *, float *, float *, float *, float*, float, float);
;
;void g(float *restrict A, float *restrict B, float *C, float alpha, float beta) {
;    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, 2, 4, 3, alpha, A, 2, B, 3, beta, C, 4);
;}
;
;int main() {
;    float A[] = {1, 4,
;                 2, 5,
;                 3, 6};
;    float B[] = {21, 0.3, 0.7,
;                 0.9, 1, 26,
;                 30, 31, 32,
;                 33, 34, 35};
;    float C[] = { 0.00, 0.00, 0.0, 0.0,
;                0.00, 0.00, 0.0, 0.0};
;    float A1[] = {0, 0, 0,
;                  0, 0, 0};
;    float B1[] = {0, 0, 0, 0,
;                  0, 0, 0, 0,
;                  0, 0, 0, 0};
;    float C1[] = {1, 1, 1, 1,
;                  1, 1, 1, 1};
;    __enzyme_autodiff((void*)g, A, A1, B, B1, C, C1, 2.0, 2.0);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.A = private unnamed_addr constant [6 x float] [float 1.000000e+00, float 4.000000e+00, float 2.000000e+00, float 5.000000e+00, float 3.000000e+00, float 6.000000e+00], align 16
@__const.main.B = private unnamed_addr constant [12 x float] [float 2.100000e+01, float 0x3FD3333340000000, float 0x3FE6666660000000, float 0x3FECCCCCC0000000, float 1.000000e+00, float 2.600000e+01, float 3.000000e+01, float 3.100000e+01, float 3.200000e+01, float 3.300000e+01, float 3.400000e+01, float 3.500000e+01], align 16
@__const.main.C1 = private unnamed_addr constant [8 x float] [float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00], align 16

define dso_local void @g(float* %A, float* %B, float* %C, float %alpha, float %beta) {
entry:
  call void @cblas_sgemm(i32 101, i32 112, i32 112, i32 2, i32 4, i32 3, float %alpha, float* %A, i32 2, float* %B, i32 3, float %beta, float* %C, i32 4)
  ret void
}

declare dso_local void @cblas_sgemm(i32, i32, i32, i32, i32, i32, float, float*, i32, float*, i32, float, float*, i32)

define dso_local i32 @main() {
entry:
  %A = alloca [6 x float], align 16
  %B = alloca [12 x float], align 16
  %C = alloca [8 x float], align 16
  %A1 = alloca [6 x float], align 16
  %B1 = alloca [12 x float], align 16
  %C1 = alloca [8 x float], align 16
  %0 = bitcast [6 x float]* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast ([6 x float]* @__const.main.A to i8*), i64 24, i1 false)
  %1 = bitcast [12 x float]* %B to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %1, i8* align 16 bitcast ([12 x float]* @__const.main.B to i8*), i64 48, i1 false)
  %2 = bitcast [8 x float]* %C to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %2, i8 0, i64 32, i1 false)
  %3 = bitcast [6 x float]* %A1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %3, i8 0, i64 24, i1 false)
  %4 = bitcast [12 x float]* %B1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %4, i8 0, i64 48, i1 false)
  %5 = bitcast [8 x float]* %C1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %5, i8* align 16 bitcast ([8 x float]* @__const.main.C1 to i8*), i64 32, i1 false)
  %arraydecay = getelementptr inbounds [6 x float], [6 x float]* %A, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [6 x float], [6 x float]* %A1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [12 x float], [12 x float]* %B, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [12 x float], [12 x float]* %B1, i32 0, i32 0
  %arraydecay4 = getelementptr inbounds [8 x float], [8 x float]* %C, i32 0, i32 0
  %arraydecay5 = getelementptr inbounds [8 x float], [8 x float]* %C1, i32 0, i32 0
  %call = call float @__enzyme_autodiff(i8* bitcast (void (float*, float*, float*, float, float)* @g to i8*), float* %arraydecay, float* %arraydecay1, float* %arraydecay2, float* %arraydecay3, float* %arraydecay4, float* %arraydecay5, float 2.000000e+00, float 2.000000e+00)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local float @__enzyme_autodiff(i8*, float*, float*, float*, float*, float*, float*, float, float)

;CHECK:define internal { float, float } @diffeg(float* %A, float* %"A'", float* %B, float* %"B'", float* %C, float* %"C'", float %alpha, float %beta) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  call void @cblas_sgemm(i32 101, i32 112, i32 112, i32 2, i32 4, i32 3, float %alpha, float* nocapture readonly %A, i32 2, float* nocapture readonly %B, i32 3, float %beta, float* %C, i32 4)
;CHECK-NEXT:  call void @cblas_sgemm(i32 101, i32 111, i32 111, i32 2, i32 3, i32 4, float %alpha, float* nocapture readonly %"C'", i32 4, float* nocapture readonly %B, i32 3, float 1.000000e+00, float* %"A'", i32 3)
;CHECK-NEXT:  call void @cblas_sgemm(i32 101, i32 111, i32 111, i32 3, i32 4, i32 2, float %alpha, float* nocapture readonly %A, i32 2, float* nocapture readonly %"C'", i32 4, float 1.000000e+00, float* %"B'", i32 4)
;CHECK-NEXT:  call void @cblas_sscal(i32 8, float %beta, float* %"C'", i32 1)
;CHECK-NEXT:  ret { float, float } zeroinitializer
;CHECK-NEXT:}

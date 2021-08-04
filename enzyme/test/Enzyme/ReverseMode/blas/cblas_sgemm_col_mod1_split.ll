;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

;#include <cblas.h>
;
;extern float __enzyme_autodiff(void *, float *, float *, float *, float *, float *, float*, float, float);
;
;void outer(float *Afoo, float *Bfoo, float *Cfoo, float alpha, float beta) {
;    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 3, 2, alpha, Afoo, 4, Bfoo, 2, beta, Cfoo, 4);
;}
;
;void g(float *A, float *B, float *C, float alpha, float beta) {
;    outer(A, B, C, alpha, beta);
;    B[0] = 10;
;}
;
;int main() {
;    float A[] = {0.11, 0.12, 0.13, 0.14,
;                 0.21, 0.22, 0.23, 0.24};
;    float B[] = {1011, 1012,
;                 1021, 1022,
;                 1031, 1032};
;    float C[] = {0.00, 0.00, 0.00, 0.00,
;                 0.00, 0.00, 0.00, 0.00,
;                 0.00, 0.00, 0.00, 0.00};
;    float A1[] = {0, 0, 0, 0, 0, 0, 0, 0};
;    float B1[] = {0, 0, 0, 0, 0, 0};
;    float C1[] = {1, 3, 7, 11,
;                  0.00, 0.00, 0.00, 0.00,
;                  0.00, 0.00, 0.00, 0.00};
;    __enzyme_autodiff((void*)g, A, A1, B, B1, C, C1, 2.0, 3.0);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.A = private unnamed_addr constant [8 x float] [float 0x3FBC28F5C0000000, float 0x3FBEB851E0000000, float 0x3FC0A3D700000000, float 0x3FC1EB8520000000, float 0x3FCAE147A0000000, float 0x3FCC28F5C0000000, float 0x3FCD70A3E0000000, float 0x3FCEB851E0000000], align 16
@__const.main.B = private unnamed_addr constant [6 x float] [float 1.011000e+03, float 1.012000e+03, float 1.021000e+03, float 1.022000e+03, float 1.031000e+03, float 1.032000e+03], align 16

define dso_local void @outer(float* %Afoo, float* %Bfoo, float* %Cfoo, float %alpha, float %beta) {
entry:
  call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 4, i32 3, i32 2, float %alpha, float* %Afoo, i32 4, float* %Bfoo, i32 2, float %beta, float* %Cfoo, i32 4)
  ret void
}

declare dso_local void @cblas_sgemm(i32, i32, i32, i32, i32, i32, float, float*, i32, float*, i32, float, float*, i32)

define dso_local void @g(float* %A, float* %B, float* %C, float %alpha, float %beta) {
entry:
  call void @outer(float* %A, float* %B, float* %C, float %alpha, float %beta)
  %arrayidx = getelementptr inbounds float, float* %B, i64 0
  store float 1.000000e+01, float* %arrayidx, align 4
  ret void
}

define dso_local i32 @main() {
entry:
  %A = alloca [8 x float], align 16
  %B = alloca [6 x float], align 16
  %C = alloca [12 x float], align 16
  %A1 = alloca [8 x float], align 16
  %B1 = alloca [6 x float], align 16
  %C1 = alloca [12 x float], align 16
  %0 = bitcast [8 x float]* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast ([8 x float]* @__const.main.A to i8*), i64 32, i1 false)
  %1 = bitcast [6 x float]* %B to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %1, i8* align 16 bitcast ([6 x float]* @__const.main.B to i8*), i64 24, i1 false)
  %2 = bitcast [12 x float]* %C to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %2, i8 0, i64 48, i1 false)
  %3 = bitcast [8 x float]* %A1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %3, i8 0, i64 32, i1 false)
  %4 = bitcast [6 x float]* %B1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %4, i8 0, i64 24, i1 false)
  %5 = bitcast [12 x float]* %C1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %5, i8 0, i64 48, i1 false)
  %6 = bitcast i8* %5 to <{ float, float, float, float, [8 x float] }>*
  %7 = getelementptr inbounds <{ float, float, float, float, [8 x float] }>, <{ float, float, float, float, [8 x float] }>* %6, i32 0, i32 0
  store float 1.000000e+00, float* %7, align 16
  %8 = getelementptr inbounds <{ float, float, float, float, [8 x float] }>, <{ float, float, float, float, [8 x float] }>* %6, i32 0, i32 1
  store float 3.000000e+00, float* %8, align 4
  %9 = getelementptr inbounds <{ float, float, float, float, [8 x float] }>, <{ float, float, float, float, [8 x float] }>* %6, i32 0, i32 2
  store float 7.000000e+00, float* %9, align 8
  %10 = getelementptr inbounds <{ float, float, float, float, [8 x float] }>, <{ float, float, float, float, [8 x float] }>* %6, i32 0, i32 3
  store float 1.100000e+01, float* %10, align 4
  %arraydecay = getelementptr inbounds [8 x float], [8 x float]* %A, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [8 x float], [8 x float]* %A1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [6 x float], [6 x float]* %B, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [6 x float], [6 x float]* %B1, i32 0, i32 0
  %arraydecay4 = getelementptr inbounds [12 x float], [12 x float]* %C, i32 0, i32 0
  %arraydecay5 = getelementptr inbounds [12 x float], [12 x float]* %C1, i32 0, i32 0
  %call = call float @__enzyme_autodiff(i8* bitcast (void (float*, float*, float*, float, float)* @g to i8*), float* %arraydecay, float* %arraydecay1, float* %arraydecay2, float* %arraydecay3, float* %arraydecay4, float* %arraydecay5, float 2.000000e+00, float 3.000000e+00)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local float @__enzyme_autodiff(i8*, float*, float*, float*, float*, float*, float*, float, float)

;CHECK:define internal { float*, float* } @augmented_outer(float* %Afoo, float* %"Afoo'", float* %Bfoo, float* %"Bfoo'", float* %Cfoo, float* %"Cfoo'", float %alpha, float %beta) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 mul (i64 ptrtoint (float* getelementptr (float, float* null, i32 1) to i64), i64 8))
;CHECK-NEXT:  %0 = bitcast i8* %malloccall to float*
;CHECK-NEXT:  %1 = bitcast float* %Afoo to i8*
;CHECK-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %malloccall, i8* %1, i32 8, i1 false)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 mul (i64 ptrtoint (float* getelementptr (float, float* null, i32 1) to i64), i64 6))
;CHECK-NEXT:  %2 = bitcast i8* %malloccall2 to float*
;CHECK-NEXT:  %3 = bitcast float* %Bfoo to i8*
;CHECK-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %malloccall2, i8* %3, i32 6, i1 false)
;CHECK-NEXT:  %4 = insertvalue { float*, float* } undef, float* %0, 0
;CHECK-NEXT:  %5 = insertvalue { float*, float* } %4, float* %2, 1
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 111, i32 111, i32 4, i32 3, i32 2, float %alpha, float* nocapture readonly %Afoo, i32 4, float* nocapture readonly %Bfoo, i32 2, float %beta, float* %Cfoo, i32 4)
;CHECK-NEXT:  ret { float*, float* } %5
;CHECK-NEXT:}

;CHECK:define internal { float, float } @diffeouter(float* %Afoo, float* %"Afoo'", float* %Bfoo, float* %"Bfoo'", float* %Cfoo, float* %"Cfoo'", float %alpha, float %beta, { float*, float* }
;CHECK-NEXT:entry:
;CHECK-NEXT:  %1 = extractvalue { float*, float* } %0, 0
;CHECK-NEXT:  %2 = extractvalue { float*, float* } %0, 1
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 111, i32 112, i32 4, i32 2, i32 3, float %alpha, float* nocapture readonly %"Cfoo'", i32 4, float* nocapture readonly %2, i32 3, float 1.000000e+00, float* %"Afoo'", i32 4)
;CHECK-NEXT:  %3 = bitcast float* %1 to i8*
;CHECK-NEXT:  tail call void @free(i8* %3)
;CHECK-NEXT:  call void @cblas_sgemm(i32 102, i32 112, i32 111, i32 2, i32 3, i32 4, float %alpha, float* nocapture readonly %1, i32 4, float* nocapture readonly %"Cfoo'", i32 4, float 1.000000e+00, float* %"Bfoo'", i32 2)
;CHECK-NEXT:  %4 = bitcast float* %2 to i8*
;CHECK-NEXT:  tail call void @free(i8* %4)
;CHECK-NEXT:  call void @cblas_sscal(i32 12, float %beta, float* %"Cfoo'", i32 1)
;CHECK-NEXT:  ret { float, float } zeroinitializer
;CHECK-NEXT:}

;CHECK:declare void @cblas_sscal(i32, float, float*, i32)

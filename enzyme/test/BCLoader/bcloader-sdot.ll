;RUN: if [ %llvmver -ge 10 ]; then %clang %s -Xclang -load -Xclang %loadBC -mllvm -bcpath=%BClibdir -S -emit-llvm -o - | %FileCheck %s; fi

;#include <cblas.h>
;
;extern void __enzyme_autodiff(void *, float*, float*, float*, float*);
;
;float g(float *m, float *n) {
;    float x = cblas_sdot(3, m, 1, n, 1);
;    return (x*x);
;}
;
;int main() {
;    float m[3] = {1.0, 2.0, 3.0};
;    float m1[3] = {0.0, 0.0, 0.0};
;    float n[3] = {4.0, 5.0, 6.0};
;    float n1[3] = {0.0, 0.0, 0.0};
;    __enzyme_autodiff((void*)g, m, m1, n, n1);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.m = private unnamed_addr constant [3 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00], align 4
@__const.main.n = private unnamed_addr constant [3 x float] [float 4.000000e+00, float 5.000000e+00, float 6.000000e+00], align 4

define dso_local float @g(float* %m, float* %n) {
entry:
  %m.addr = alloca float*, align 8
  %n.addr = alloca float*, align 8
  %x = alloca float, align 4
  store float* %m, float** %m.addr, align 8
  store float* %n, float** %n.addr, align 8
  %0 = load float*, float** %m.addr, align 8
  %1 = load float*, float** %n.addr, align 8
  %call = call float @cblas_sdot(i32 3, float* %0, i32 1, float* %1, i32 1)
  store float %call, float* %x, align 4
  %2 = load float, float* %x, align 4
  %3 = load float, float* %x, align 4
  %mul = fmul float %2, %3
  ret float %mul
}

declare dso_local float @cblas_sdot(i32, float*, i32, float*, i32)

define dso_local i32 @main() {
entry:
  %m = alloca [3 x float], align 4
  %m1 = alloca [3 x float], align 4
  %n = alloca [3 x float], align 4
  %n1 = alloca [3 x float], align 4
  %0 = bitcast [3 x float]* %m to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 bitcast ([3 x float]* @__const.main.m to i8*), i64 12, i1 false)
  %1 = bitcast [3 x float]* %m1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %1, i8 0, i64 12, i1 false)
  %2 = bitcast [3 x float]* %n to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 bitcast ([3 x float]* @__const.main.n to i8*), i64 12, i1 false)
  %3 = bitcast [3 x float]* %n1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 4 %3, i8 0, i64 12, i1 false)
  %arraydecay = getelementptr inbounds [3 x float], [3 x float]* %m, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [3 x float], [3 x float]* %m1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [3 x float], [3 x float]* %n, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [3 x float], [3 x float]* %n1, i32 0, i32 0
  call void @__enzyme_autodiff(i8* bitcast (float (float*, float*)* @g to i8*), float* %arraydecay, float* %arraydecay1, float* %arraydecay2, float* %arraydecay3)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local void @__enzyme_autodiff(i8*, float*, float*, float*, float*)

;CHECK:define dso_local float @cblas_sdot(i32 %N, float* %X, i32 %incX, float* %Y, i32 %incY)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %N.addr = alloca i32, align 4
;CHECK-NEXT:  %X.addr = alloca float*, align 8
;CHECK-NEXT:  %incX.addr = alloca i32, align 4
;CHECK-NEXT:  %Y.addr = alloca float*, align 8
;CHECK-NEXT:  %incY.addr = alloca i32, align 4
;CHECK-NEXT:  %sum = alloca float, align 4
;CHECK-NEXT:  %i = alloca i32, align 4
;CHECK-NEXT:  store i32 %N, i32* %N.addr, align 4
;CHECK-NEXT:  store float* %X, float** %X.addr, align 8
;CHECK-NEXT:  store i32 %incX, i32* %incX.addr, align 4
;CHECK-NEXT:  store float* %Y, float** %Y.addr, align 8
;CHECK-NEXT:  store i32 %incY, i32* %incY.addr, align 4
;CHECK-NEXT:  store float 0.000000e+00, float* %sum, align 4
;CHECK-NEXT:  store i32 0, i32* %i, align 4
;CHECK-NEXT:  br label %for.cond

;CHECK:for.cond:                                         ; preds = %for.inc, %entry
;CHECK-NEXT:  %0 = load i32, i32* %i, align 4
;CHECK-NEXT:  %1 = load i32, i32* %N.addr, align 4
;CHECK-NEXT:  %cmp = icmp slt i32 %0, %1
;CHECK-NEXT:  br i1 %cmp, label %for.body, label %for.end

;CHECK:for.body:                                         ; preds = %for.cond
;CHECK-NEXT:  %2 = load float*, float** %X.addr, align 8
;CHECK-NEXT:  %3 = load i32, i32* %i, align 4
;CHECK-NEXT:  %idxprom = sext i32 %3 to i64
;CHECK-NEXT:  %arrayidx = getelementptr inbounds float, float* %2, i64 %idxprom
;CHECK-NEXT:  %4 = load float, float* %arrayidx, align 4
;CHECK-NEXT:  %5 = load float*, float** %Y.addr, align 8
;CHECK-NEXT:  %6 = load i32, i32* %i, align 4
;CHECK-NEXT:  %idxprom1 = sext i32 %6 to i64
;CHECK-NEXT:  %arrayidx2 = getelementptr inbounds float, float* %5, i64 %idxprom1
;CHECK-NEXT:  %7 = load float, float* %arrayidx2, align 4
;CHECK-NEXT:  %mul = fmul float %4, %7
;CHECK-NEXT:  %8 = load float, float* %sum, align 4
;CHECK-NEXT:  %add = fadd float %8, %mul
;CHECK-NEXT:  store float %add, float* %sum, align 4
;CHECK-NEXT:  br label %for.inc

;CHECK:for.inc:                                          ; preds = %for.body
;CHECK-NEXT:  %9 = load i32, i32* %i, align 4
;CHECK-NEXT:  %inc = add nsw i32 %9, 1
;CHECK-NEXT:  store i32 %inc, i32* %i, align 4
;CHECK-NEXT:  br label %for.cond

;CHECK:for.end:                                          ; preds = %for.cond
;CHECK-NEXT:  %10 = load float, float* %sum, align 4
;CHECK-NEXT:  ret float %10
;CHECK-NEXT:}

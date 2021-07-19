;RUN: if [ %llvmver -ge 10 ]; then %clang %s -Xclang -load -Xclang %loadBC -mllvm -bcpath=%BClibdir -S -emit-llvm -o - | %FileCheck %s; fi

;#include <cblas.h>
;
;extern void __enzyme_autodiff(void *, double *, double*, double*, double*);
;
;void g(double *m, double *n) {
;    cblas_daxpy(3, 11, m, 1, n, 1);
;}
;
;int main() {
;    double a[3] = {1.0, 2.0, 3.0};
;    double b[3] = {4.0, 5.0, 6.0};
;    double a1[3] = {0, 0, 0};
;    double b1[3] = {0, 0, 0};
;    __enzyme_autodiff((void*)g, a, a1, b, b1);
;    return 1;
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.a = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 16
@__const.main.b = private unnamed_addr constant [3 x double] [double 4.000000e+00, double 5.000000e+00, double 6.000000e+00], align 16

define dso_local void @g(double* %m, double* %n) {
entry:
  %m.addr = alloca double*, align 8
  %n.addr = alloca double*, align 8
  store double* %m, double** %m.addr, align 8
  store double* %n, double** %n.addr, align 8
  %0 = load double*, double** %m.addr, align 8
  %1 = load double*, double** %n.addr, align 8
  call void @cblas_daxpy(i32 3, double 1.100000e+01, double* %0, i32 1, double* %1, i32 1)
  ret void
}

declare dso_local void @cblas_daxpy(i32, double, double*, i32, double*, i32)

define dso_local i32 @main() {
entry:
  %retval = alloca i32, align 4
  %a = alloca [3 x double], align 16
  %b = alloca [3 x double], align 16
  %a1 = alloca [3 x double], align 16
  %b1 = alloca [3 x double], align 16
  store i32 0, i32* %retval, align 4
  %0 = bitcast [3 x double]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 16 bitcast ([3 x double]* @__const.main.a to i8*), i64 24, i1 false)
  %1 = bitcast [3 x double]* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %1, i8* align 16 bitcast ([3 x double]* @__const.main.b to i8*), i64 24, i1 false)
  %2 = bitcast [3 x double]* %a1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %2, i8 0, i64 24, i1 false)
  %3 = bitcast [3 x double]* %b1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %3, i8 0, i64 24, i1 false)
  %arraydecay = getelementptr inbounds [3 x double], [3 x double]* %a, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [3 x double], [3 x double]* %a1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [3 x double], [3 x double]* %b, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [3 x double], [3 x double]* %b1, i32 0, i32 0
  call void @__enzyme_autodiff(i8* bitcast (void (double*, double*)* @g to i8*), double* %arraydecay, double* %arraydecay1, double* %arraydecay2, double* %arraydecay3)
  ret i32 1
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare dso_local void @__enzyme_autodiff(i8*, double*, double*, double*, double*)

;CHECK:define dso_local void @cblas_daxpy(i32 %N, double %alpha, double* %X, i32 %incX, double* %Y, i32 %incY)
;CHECK-NEXT:entry:
;CHECK-NEXT:  %N.addr = alloca i32, align 4
;CHECK-NEXT:  %alpha.addr = alloca double, align 8
;CHECK-NEXT:  %X.addr = alloca double*, align 8
;CHECK-NEXT:  %incX.addr = alloca i32, align 4
;CHECK-NEXT:  %Y.addr = alloca double*, align 8
;CHECK-NEXT:  %incY.addr = alloca i32, align 4
;CHECK-NEXT:  %i = alloca i32, align 4
;CHECK-NEXT:  store i32 %N, i32* %N.addr, align 4
;CHECK-NEXT:  store double %alpha, double* %alpha.addr, align 8
;CHECK-NEXT:  store double* %X, double** %X.addr, align 8
;CHECK-NEXT:  store i32 %incX, i32* %incX.addr, align 4
;CHECK-NEXT:  store double* %Y, double** %Y.addr, align 8
;CHECK-NEXT:  store i32 %incY, i32* %incY.addr, align 4
;CHECK-NEXT:  store i32 0, i32* %i, align 4
;CHECK-NEXT:  br label %for.cond

;CHECK:for.cond:                                         ; preds = %for.inc, %entry
;CHECK-NEXT:  %0 = load i32, i32* %i, align 4
;CHECK-NEXT:  %1 = load i32, i32* %N.addr, align 4
;CHECK-NEXT:  %cmp = icmp slt i32 %0, %1
;CHECK-NEXT:  br i1 %cmp, label %for.body, label %for.end

;CHECK:for.body:                                         ; preds = %for.cond
;CHECK-NEXT:  %2 = load double, double* %alpha.addr, align 8
;CHECK-NEXT:  %3 = load double*, double** %X.addr, align 8
;CHECK-NEXT:  %4 = load i32, i32* %i, align 4
;CHECK-NEXT:  %idxprom = sext i32 %4 to i64
;CHECK-NEXT:  %arrayidx = getelementptr inbounds double, double* %3, i64 %idxprom
;CHECK-NEXT:  %5 = load double, double* %arrayidx, align 8
;CHECK-NEXT:  %mul = fmul double %2, %5
;CHECK-NEXT:  %6 = load double*, double** %Y.addr, align 8
;CHECK-NEXT:  %7 = load i32, i32* %i, align 4
;CHECK-NEXT:  %idxprom1 = sext i32 %7 to i64
;CHECK-NEXT:  %arrayidx2 = getelementptr inbounds double, double* %6, i64 %idxprom1
;CHECK-NEXT:  %8 = load double, double* %arrayidx2, align 8
;CHECK-NEXT:  %add = fadd double %mul, %8
;CHECK-NEXT:  %9 = load double*, double** %Y.addr, align 8
;CHECK-NEXT:  %10 = load i32, i32* %i, align 4
;CHECK-NEXT:  %idxprom3 = sext i32 %10 to i64
;CHECK-NEXT:  %arrayidx4 = getelementptr inbounds double, double* %9, i64 %idxprom3
;CHECK-NEXT:  store double %add, double* %arrayidx4, align 8
;CHECK-NEXT:  br label %for.inc

;CHECK:for.inc:                                          ; preds = %for.body
;CHECK-NEXT:  %11 = load i32, i32* %i, align 4
;CHECK-NEXT:  %inc = add nsw i32 %11, 1
;CHECK-NEXT:  store i32 %inc, i32* %i, align 4
;CHECK-NEXT:  br label %for.cond

;CHECK:for.end:                                          ; preds = %for.cond
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

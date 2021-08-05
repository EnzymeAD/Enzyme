;RUN: if [ %llvmver -ge 8 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi

;#include <cblas.h>
;
;extern double __enzyme_autodiff(void *, double *, double *, double *,
;                                 double *);
;
;void outer(double* out, double *a, double *b) {
;  *out = cblas_ddot(3, a, 2, b, 3);
;}
;
;double g(double *restrict m, double *restrict n) {
;  double x;
;  outer(&x, m, n);
;  m[0] = 11.0;
;  m[1] = 12.0;
;  m[2] = 13.0;
;  double y = x * x;
;  return y;
;}
;
;int main() {
;  double m[6] = {1, 2, 3, 101, 102, 103};
;  double m1[6] = {0, 0, 0, 0, 0, 0};
;  double n[9] = {4, 5, 6, 104, 105, 106, 7, 8, 9};
;  double n1[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
;  __enzyme_autodiff((void*)g, m, m1, n, n1);
;}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.n = private unnamed_addr constant [9 x double] [double 4.000000e+00, double 5.000000e+00, double 6.000000e+00, double 1.040000e+02, double 1.050000e+02, double 1.060000e+02, double 7.000000e+00, double 8.000000e+00, double 9.000000e+00], align 16

define dso_local void @outer(double* %out, double* %a, double* %b) {
entry:
  %out.addr = alloca double*, align 8
  %a.addr = alloca double*, align 8
  %b.addr = alloca double*, align 8
  store double* %out, double** %out.addr, align 8
  store double* %a, double** %a.addr, align 8
  store double* %b, double** %b.addr, align 8
  %0 = load double*, double** %a.addr, align 8
  %1 = load double*, double** %b.addr, align 8
  %call = call double @cblas_ddot(i32 3, double* %0, i32 2, double* %1, i32 3)
  %2 = load double*, double** %out.addr, align 8
  store double %call, double* %2, align 8
  ret void
}

declare dso_local double @cblas_ddot(i32, double*, i32, double*, i32)

define dso_local double @g(double* noalias %m, double* noalias %n) {
entry:
  %m.addr = alloca double*, align 8
  %n.addr = alloca double*, align 8
  %x = alloca double, align 8
  %y = alloca double, align 8
  store double* %m, double** %m.addr, align 8
  store double* %n, double** %n.addr, align 8
  %0 = load double*, double** %m.addr, align 8
  %1 = load double*, double** %n.addr, align 8
  call void @outer(double* %x, double* %0, double* %1)
  %2 = load double*, double** %m.addr, align 8
  %arrayidx = getelementptr inbounds double, double* %2, i64 0
  store double 1.100000e+01, double* %arrayidx, align 8
  %3 = load double*, double** %m.addr, align 8
  %arrayidx1 = getelementptr inbounds double, double* %3, i64 1
  store double 1.200000e+01, double* %arrayidx1, align 8
  %4 = load double*, double** %m.addr, align 8
  %arrayidx2 = getelementptr inbounds double, double* %4, i64 2
  store double 1.300000e+01, double* %arrayidx2, align 8
  %5 = load double, double* %x, align 8
  %6 = load double, double* %x, align 8
  %mul = fmul double %5, %6
  store double %mul, double* %y, align 8
  %7 = load double, double* %y, align 8
  ret double %7
}

define dso_local i32 @main() {
entry:
  %m = alloca [6 x double], align 16
  %m1 = alloca [6 x double], align 16
  %n = alloca [9 x double], align 16
  %n1 = alloca [9 x double], align 16
  %0 = bitcast [6 x double]* %m to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %0, i8 0, i64 48, i1 false)
  %1 = bitcast i8* %0 to [6 x double]*
  %2 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 0
  store double 1.000000e+00, double* %2, align 16
  %3 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 1
  store double 2.000000e+00, double* %3, align 8
  %4 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 2
  store double 3.000000e+00, double* %4, align 16
  %5 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 3
  store double 1.010000e+02, double* %5, align 8
  %6 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 4
  store double 1.020000e+02, double* %6, align 16
  %7 = getelementptr inbounds [6 x double], [6 x double]* %1, i32 0, i32 5
  store double 1.030000e+02, double* %7, align 8
  %8 = bitcast [6 x double]* %m1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %8, i8 0, i64 48, i1 false)
  %9 = bitcast [9 x double]* %n to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %9, i8* align 16 bitcast ([9 x double]* @__const.main.n to i8*), i64 72, i1 false)
  %10 = bitcast [9 x double]* %n1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %10, i8 0, i64 72, i1 false)
  %arraydecay = getelementptr inbounds [6 x double], [6 x double]* %m, i32 0, i32 0
  %arraydecay1 = getelementptr inbounds [6 x double], [6 x double]* %m1, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [9 x double], [9 x double]* %n, i32 0, i32 0
  %arraydecay3 = getelementptr inbounds [9 x double], [9 x double]* %n1, i32 0, i32 0
  %call = call double @__enzyme_autodiff(i8* bitcast (double (double*, double*)* @g to i8*), double* %arraydecay, double* %arraydecay1, double* %arraydecay2, double* %arraydecay3)
  ret i32 0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare dso_local double @__enzyme_autodiff(i8*, double*, double*, double*, double*)

;CHECK:define internal void @diffeg(double* noalias %m, double* %"m'", double* noalias %n, double* %"n'", double %differeturn) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  %"x'ipa" = alloca double, align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"x'ipa", align 8
;CHECK-NEXT:  %x = alloca double, align 8
;CHECK-NEXT:  %_augmented = call { double*, double* } @augmented_outer(double* %x, double* %"x'ipa", double* %m, double* %"m'", double* %n, double* %"n'")
;CHECK-NEXT:  store double 1.100000e+01, double* %m, align 8
;CHECK-NEXT:  %"arrayidx1'ipg" = getelementptr inbounds double, double* %"m'", i64 1
;CHECK-NEXT:  %arrayidx1 = getelementptr inbounds double, double* %m, i64 1
;CHECK-NEXT:  store double 1.200000e+01, double* %arrayidx1, align 8
;CHECK-NEXT:  %"arrayidx2'ipg" = getelementptr inbounds double, double* %"m'", i64 2
;CHECK-NEXT:  %arrayidx2 = getelementptr inbounds double, double* %m, i64 2
;CHECK-NEXT:  store double 1.300000e+01, double* %arrayidx2, align 8
;CHECK-NEXT:  %0 = load double, double* %x, align 8
;CHECK-NEXT:  %1 = load double, double* %x, align 8
;CHECK-NEXT:  %m0diffe = fmul fast double %differeturn, %1
;CHECK-NEXT:  %m1diffe = fmul fast double %differeturn, %0
;CHECK-NEXT:  %2 = load double, double* %"x'ipa", align 8
;CHECK-NEXT:  %3 = fadd fast double %2, %m1diffe
;CHECK-NEXT:  store double %3, double* %"x'ipa", align 8
;CHECK-NEXT:  %4 = load double, double* %"x'ipa", align 8
;CHECK-NEXT:  %5 = fadd fast double %4, %m0diffe
;CHECK-NEXT:  store double %5, double* %"x'ipa", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"arrayidx2'ipg", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"arrayidx1'ipg", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"m'", align 8
;CHECK-NEXT:  call void @diffeouter(double* %x, double* %"x'ipa", double* %m, double* %"m'", double* %n, double* %"n'", { double*, double* } %_augmented)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:define internal { double*, double* } @augmented_outer(double* %out, double* %"out'", double* %a, double* %"a'", double* %b, double* %"b'") {
;CHECK-NEXT:entry:
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 mul (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 3))
;CHECK-NEXT:  %0 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0stride(double* %0, double* %a, i32 3, i32 2)
;CHECK-NEXT:  %malloccall2 = tail call i8* @malloc(i64 mul (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 3))
;CHECK-NEXT:  %1 = bitcast i8* %malloccall2 to double*
;CHECK-NEXT:  call void @__enzyme_memcpy_doubleda0sa0stride(double* %1, double* %b, i32 3, i32 3)
;CHECK-NEXT:  %2 = insertvalue { double*, double* } undef, double* %0, 0
;CHECK-NEXT:  %3 = insertvalue { double*, double* } %2, double* %1, 1
;CHECK-NEXT:  %call = call double @cblas_ddot(i32 3, double* nocapture readonly %a, i32 2, double* nocapture readonly %b, i32 3)
;CHECK-NEXT:  store double %call, double* %out, align 8
;CHECK-NEXT:  ret { double*, double* } %3
;CHECK-NEXT:}

;CHECK:define internal void @diffeouter(double* %out, double* %"out'", double* %a, double* %"a'", double* %b, double* %"b'", { double*, double* }
;CHECK-NEXT:entry:
;CHECK-NEXT:  %1 = extractvalue { double*, double* } %0, 0
;CHECK-NEXT:  %2 = extractvalue { double*, double* } %0, 1
;CHECK-NEXT:  %3 = load double, double* %"out'", align 8
;CHECK-NEXT:  store double 0.000000e+00, double* %"out'", align 8
;CHECK-NEXT:  call void @cblas_daxpy(i32 3, double %3, double* %1, i32 1, double* %"b'", i32 3)
;CHECK-NEXT:  %4 = bitcast double* %1 to i8*
;CHECK-NEXT:  tail call void @free(i8* %4)
;CHECK-NEXT:  call void @cblas_daxpy(i32 3, double %3, double* %2, i32 1, double* %"a'", i32 2)
;CHECK-NEXT:  %5 = bitcast double* %2 to i8*
;CHECK-NEXT:  tail call void @free(i8* %5)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

;CHECK:declare void @cblas_daxpy(i32, double, double*, i32, double*, i32)

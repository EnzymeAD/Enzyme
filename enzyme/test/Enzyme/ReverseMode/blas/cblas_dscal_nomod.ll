;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@__const.main.m = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 16
@__const.main.d_m = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], align 16

define void @g(double* noalias %x, double %a) {
  call void @cblas_dscal(i32 3, double %a, double* %x, i32 1)
  ret void
}

declare void @cblas_dscal(i32, double, double*, i32)

define i32 @main() {
  %1 = alloca [3 x double], align 16
  %2 = alloca [3 x double], align 16
  %pa = alloca double, align 8
  store double 2.000000e+00, double* %pa, align 8
  %3 = bitcast [3 x double]* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %3, i8* align 16 bitcast ([3 x double]* @__const.main.m to i8*), i64 24, i1 false)
  %4 = bitcast [3 x double]* %2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %4, i8* align 16 bitcast ([3 x double]* @__const.main.d_m to i8*), i64 24, i1 false)
  %5 = getelementptr inbounds [3 x double], [3 x double]* %1, i64 0, i64 0
  %6 = getelementptr inbounds [3 x double], [3 x double]* %2, i64 0, i64 0
  %a = load double, double* %pa, align 8
  %call = call double @__enzyme_autodiff(i8* bitcast (void (double*, double)* @g to i8*), double* %5, double* %6, double %a)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

declare double @__enzyme_autodiff(i8*, double*, double*, double)

declare i32 @printf(i8*, ...)

;CHECK:define internal { double } @diffeg(double* noalias %x, double* %"x'", double %a) {
;CHECK-NEXT:invert:
;CHECK-NEXT:  call void @cblas_dscal(i32 3, double %a, double* %x, i32 1)
;CHECK-NEXT:  call void @cblas_dscal(i32 3, double %a, double* %"x'", i32 1)
;CHECK-NEXT:  ret { double } zeroinitializer
;CHECK-NEXT:}

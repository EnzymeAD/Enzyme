;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.17.0"
@__const.main.m = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 16
@__const.main.d_m = private unnamed_addr constant [3 x double] [double 2.000000e+00, double 4.000000e+00, double 6.000000e+00], align 16
@__const.main.n = private unnamed_addr constant [3 x double] [double 4.000000e+00, double 5.000000e+00, double 6.000000e+00], align 16
@__const.main.d_n = private unnamed_addr constant [3 x double] [double 8.000000e+00, double 1.000000e+01, double 1.200000e+01], align 16
define void @f(i32 %n, double %a, double* noalias %x, double* noalias %y) {
entry:
  tail call void @cblas_daxpy(i32 %n, double %a, double* %x, i32 1, double* %y, i32 1)
  ret void
}
declare void @cblas_daxpy(i32, double, double*, i32, double*, i32)
define i32 @main() {
entry:
  %m = alloca [3 x double], align 16
  %d_m = alloca [3 x double], align 16
  %n = alloca [3 x double], align 16
  %d_n = alloca [3 x double], align 16
  %0 = bitcast [3 x double]* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %0, i8* align 16 bitcast ([3 x double]* @__const.main.m to i8*), i64 24, i1 false)
  %1 = bitcast [3 x double]* %d_m to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %1)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %1, i8* align 16 bitcast ([3 x double]* @__const.main.d_m to i8*), i64 24, i1 false)
  %2 = bitcast [3 x double]* %n to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %2)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %2, i8* align 16 bitcast ([3 x double]* @__const.main.n to i8*), i64 24, i1 false)
  %3 = bitcast [3 x double]* %d_n to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %3)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %3, i8* align 16 bitcast ([3 x double]* @__const.main.d_n to i8*), i64 24, i1 false)
  %arraydecay = getelementptr inbounds [3 x double], [3 x double]* %m, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [3 x double], [3 x double]* %d_m, i64 0, i64 0
  %arraydecay2 = getelementptr inbounds [3 x double], [3 x double]* %n, i64 0, i64 0
  %arraydecay3 = getelementptr inbounds [3 x double], [3 x double]* %d_n, i64 0, i64 0
  %call = call double @__enzyme_autodiff(i8* bitcast (void (i32, double, double*, double*)* @f to i8*), i32 3, double 1.000000e+00, double* nonnull %arraydecay, double* nonnull %arraydecay1, double* nonnull %arraydecay2, double* nonnull %arraydecay3)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %3)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %2)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0)
  ret i32 0
}
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
declare double @__enzyme_autodiff(i8*, i32, double, double*, double*, double*, double*)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
;CHECK:define internal { double } @diffef(i32 %n, double %a, double* noalias %x, double* %"x'", double* noalias %y, double* %"y'") {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_daxpy(i32 %n, double %a, double* %x, i32 1, double* %y, i32 1)
;CHECK-NEXT:  call void @cblas_daxpy(i32 %n, double %a, double* %"y'", i32 1, double* %"x'", i32 1)
;CHECK-NEXT:  %0 = call fast double @cblas_ddot(i32 %n, double* nocapture readonly %"y'", i32 1, double* nocapture readonly %x, i32 1)
;CHECK-NEXT:  %1 = insertvalue { double } undef, double %0, 0
;CHECK-NEXT:  ret { double } %1
;CHECK-NEXT:}

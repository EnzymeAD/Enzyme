;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.17.0"
@__const.main.m = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 16
@__const.main.d_m = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], align 16
define void @f(double* noalias %x) {
entry:
  tail call void @cblas_dscal(i32 3, double 2.000000e+00, double* %x, i32 1)
  ret void
}
declare void @cblas_dscal(i32, double, double*, i32)
define i32 @main() {
entry:
  %m = alloca [3 x double], align 16
  %d_m = alloca [3 x double], align 16
  %0 = bitcast [3 x double]* %m to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %0, i8* align 16 bitcast ([3 x double]* @__const.main.m to i8*), i64 24, i1 false)
  %1 = bitcast [3 x double]* %d_m to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %1)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %1, i8* align 16 bitcast ([3 x double]* @__const.main.d_m to i8*), i64 24, i1 false)
  %arraydecay = getelementptr inbounds [3 x double], [3 x double]* %m, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [3 x double], [3 x double]* %d_m, i64 0, i64 0
  call void @__enzyme_autodiff(i8* bitcast (void (double*)* @f to i8*), double* nonnull %arraydecay, double* nonnull %arraydecay1)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0)
  ret i32 0
}
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
declare void @__enzyme_autodiff(i8*, double*, double*)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
;CHECK:define internal void @diffef(double* noalias %x, double* %"x'") {
;CHECK-NEXT:entry:
;CHECK-NEXT:  tail call void @cblas_dscal(i32 3, double 2.000000e+00, double* %x, i32 1)
;CHECK-NEXT:  call void @cblas_dscal(i32 3, double 2.000000e+00, double* %"x'", i32 1)
;CHECK-NEXT:  ret void
;CHECK-NEXT:}

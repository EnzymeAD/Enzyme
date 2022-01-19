;RUN: %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.17.0"
@__const.main.a = private unnamed_addr constant [4 x double] [double 1.000000e+00, double 0.000000e+00, double 0.000000e+00, double 1.000000e+00], align 16
@__const.main.x = private unnamed_addr constant [2 x double] [double 1.000000e+00, double 1.000000e+00], align 16
@__const.main.y = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 16
@__const.main.d_y = private unnamed_addr constant [3 x double] [double 2.000000e+00, double 4.000000e+00, double 6.000000e+00], align 16
define void @f(double* noalias %x, double* noalias %y, double* noalias %a, double %alpha) {
entry:
  tail call void @cblas_dgemv(i32 102, i32 111, i32 2, i32 2, double %alpha, double* %a, i32 2, double* %x, i32 1, double 2.000000e+00, double* %y, i32 1)
  ret void
}
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @cblas_dgemv(i32, i32, i32, i32, double, double*, i32, double*, i32, double, double*, i32)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
define i32 @main() {
entry:
  %a = alloca [4 x double], align 16
  %d_a = alloca [4 x double], align 16
  %x = alloca [2 x double], align 16
  %d_x = alloca [2 x double], align 16
  %y = alloca [3 x double], align 16
  %d_y = alloca [3 x double], align 16
  %0 = bitcast [4 x double]* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %0)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %0, i8* align 16 bitcast ([4 x double]* @__const.main.a to i8*), i64 32, i1 false)
  %1 = bitcast [4 x double]* %d_a to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1)
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %1, i8 0, i64 32, i1 false)
  %2 = bitcast [2 x double]* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %2)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %2, i8* align 16 bitcast ([2 x double]* @__const.main.x to i8*), i64 16, i1 false)
  %3 = bitcast [2 x double]* %d_x to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  call void @llvm.memset.p0i8.i64(i8* nonnull align 16 %3, i8 0, i64 16, i1 false)
  %4 = bitcast [3 x double]* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %4)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %4, i8* align 16 bitcast ([3 x double]* @__const.main.y to i8*), i64 24, i1 false)
  %5 = bitcast [3 x double]* %d_y to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %5)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 16 %5, i8* align 16 bitcast ([3 x double]* @__const.main.d_y to i8*), i64 24, i1 false)
  %arraydecay = getelementptr inbounds [2 x double], [2 x double]* %x, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [2 x double], [2 x double]* %d_x, i64 0, i64 0
  %arraydecay2 = getelementptr inbounds [3 x double], [3 x double]* %y, i64 0, i64 0
  %arraydecay3 = getelementptr inbounds [3 x double], [3 x double]* %d_y, i64 0, i64 0
  %arraydecay4 = getelementptr inbounds [4 x double], [4 x double]* %a, i64 0, i64 0
  %arraydecay5 = getelementptr inbounds [4 x double], [4 x double]* %d_a, i64 0, i64 0
  %call = call double @__enzyme_autodiff(i8* bitcast (void (double*, double*, double*, double)* @f to i8*), double* nonnull %arraydecay, double* nonnull %arraydecay1, double* nonnull %arraydecay2, double* nonnull %arraydecay3, double* nonnull %arraydecay4, double* nonnull %arraydecay5, double 3.000000e+00)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %5)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %4)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %2)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0)
  ret i32 0
}
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)
declare double @__enzyme_autodiff(i8*, double*, double*, double*, double*, double*, double*, double)
;CHECK:define internal { double } @diffef(double* noalias %x, double* %"x'", double* noalias %y, double* %"y'", double* noalias %a, double* %"a'", double %alpha) {
;CHECK-NEXT:entry:
;CHECK-NEXT:  %malloccall = tail call i8* @malloc(i64 mul (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 2))
;CHECK-NEXT:  %0 = bitcast i8* %malloccall to double*
;CHECK-NEXT:  tail call void @cblas_dgemv(i32 102, i32 111, i32 2, i32 2, double %alpha, double* %a, i32 2, double* %x, i32 1, double 2.000000e+00, double* %y, i32 1)
;CHECK-NEXT:  call void @cblas_dscal(i32 2, double 2.000000e+00, double* %"y'", i32 1)
;CHECK-NEXT:  call void @cblas_dgemv(i32 102, i32 112, i32 2, i32 2, double %alpha, double* %a, i32 2, double* %"y'", i32 1, double 1.000000e+00, double* %"x'", i32 1)
;CHECK-NEXT:  call void @cblas_dger(i32 102, i32 2, i32 2, double %alpha, double* %"y'", i32 1, double* %"x'", i32 1, double* %a, i32 2)
;CHECK-NEXT:  call void @cblas_dgemv(i32 102, i32 111, i32 2, i32 2, double 1.000000e+00, double* %a, i32 2, double* %x, i32 1, double 0.000000e+00, double* %0, i32 1)
;CHECK-NEXT:  %1 = call fast double @cblas_ddot(i32 2, double* nocapture readonly %"y'", i32 1, double* nocapture readonly %0, i32 1)
;CHECK-NEXT:  %2 = insertvalue { double } undef, double %1, 0
;CHECK-NEXT:  ret { double } %2
;CHECK-NEXT:}

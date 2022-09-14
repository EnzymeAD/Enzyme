; RUN: if [ %llvmver -ge 9 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -instsimplify -adce -loop-deletion -correlated-propagation -simplifycfg -adce -simplifycfg -S | FileCheck %s; fi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@anon = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @str, i32 0, i32 0) }, align 8

declare dso_local void @_Z17__enzyme_autodiffIdEvPFdPKT_mEiS2_PS0_m(...)

define void @caller(double* %a, double* %da, i64 %b) {
entry:
  tail call void (...) @_Z17__enzyme_autodiffIdEvPFdPKT_mEiS2_PS0_m(void (double**, i64)* noundef nonnull @f, double* %a, double* %da, i64 %b)
  ret void
}

declare void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

define internal void @f(double** %arg, i64 %arg1) {
bb:
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* nonnull @anon, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, double**)* @outlined to void (i32*, i32*, ...)*), i64 %arg1, double** %arg)
  ret void
}

define internal void @outlined(i32* noalias %arg, i32* noalias %arg1, i64 %i10, double** %arg4) {
bb:
  %i14 = icmp eq i64 %i10, 0
  br i1 %i14, label %bb56, label %bb17

bb17:                                             ; preds = %bb
  %i33 = load double*, double** %arg4, align 8
  store double 0.000000e+00, double* %i33, align 8
  br label %bb56

bb56:                                             ; preds = %bb55, %bb15
  ret void
}

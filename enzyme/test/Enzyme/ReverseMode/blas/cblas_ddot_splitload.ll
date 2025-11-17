;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S -enzyme-detect-readthrow=0 | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S -enzyme-detect-readthrow=0 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare double @cblas_ddot(i32, double*, i32, double*, i32)

define double @f(i32 %len, double** noalias %mp, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %m = load double*, double** %mp
  %call = call double @cblas_ddot(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  ret double %call
}

define double @modf(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %mp = alloca double*
  store double* %m, double** %mp
  %call = call double @f(i32 %len, double** %mp, i32 %incm, double* %n, i32 %incn)
  store double 0.000000e+00, double* %m
  store double 0.000000e+00, double* %n
  store double* null, double** %mp
  ret double %call
}

define void @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

; CHECK: define internal { { double*, double* }, double* } @augmented_f(i32 %len, double** noalias %mp, double** %"mp'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = alloca { { double*, double* }, double* }, align 8
; CHECK-NEXT:   %"m'ipl" = load double*, double** %"mp'", align 8
; CHECK-NEXT:   %1 = getelementptr inbounds { { double*, double* }, double* }, { { double*, double* }, double* }* %0, i32 0, i32 1
; CHECK-NEXT:   store double* %"m'ipl", double** %1, align 8
; CHECK-NEXT:   %m = load double*, double** %mp, align 8
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %len, double* %m, i32 %incm, double* %cache.x, i32 1)
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i32 %len, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %cache.y = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %len, double* %n, i32 %incn, double* %cache.y, i32 1)
; CHECK-NEXT:   %2 = insertvalue { double*, double* } undef, double* %cache.x, 0
; CHECK-NEXT:   %3 = insertvalue { double*, double* } %2, double* %cache.y, 1
; CHECK-NEXT:   %4 = getelementptr inbounds { { double*, double* }, double* }, { { double*, double* }, double* }* %0, i32 0, i32 0
; CHECK-NEXT:   store { double*, double* } %3, { double*, double* }* %4, align 8
; CHECK-NEXT:   %5 = load { { double*, double* }, double* }, { { double*, double* }, double* }* %0, align 8
; CHECK-NEXT:   ret { { double*, double* }, double* } %5
; CHECK-NEXT: }

; CHECK: define internal void @diffef(i32 %len, double** noalias %mp, double** %"mp'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn, { { double*, double* }, double* } %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"m'il_phi" = extractvalue { { double*, double* }, double* } %tapeArg, 1
; CHECK-NEXT:   %0 = extractvalue { { double*, double* }, double* } %tapeArg, 0
; CHECK-NEXT:   %tape.ext.x = extractvalue { double*, double* } %0, 0
; CHECK-NEXT:   %tape.ext.y = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %tape.ext.y, i32 1, double* %"m'il_phi", i32 %incm)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %differeturn, double* %tape.ext.x, i32 1, double* %"n'", i32 %incn)
; CHECK-NEXT:   %1 = bitcast double* %tape.ext.x to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   %2 = bitcast double* %tape.ext.y to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
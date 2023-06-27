;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare double @cblas_ddot(i32, double*, i32, double*, i32)

define void @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* %dm, double* %dm, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret void
}

define void @inactiveFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2,  i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret void
}

define void @inactiveSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* noalias %dm, double* %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define void @activeMod(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* %dm, double* %dm, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, metadata !"enzyme_width", i64 2,  i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, metadata !"enzyme_width", i64 2, i32 %len, double* noalias %m, double* noalias %dm, double* %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define double @f(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @cblas_ddot(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  ret double %call
}

define double @modf(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @f(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  store double 0.000000e+00, double* %m
  store double 0.000000e+00, double* %n
  ret double %call
}


; CHECK: define void @active
; CHECK-NEXT: entry
; CHECK: call void @[[active:.+]](

; CHECK: define void @inactiveFirst
; CHECK-NEXT: entry
; CHECK: call void @[[inactiveFirst:.+]](

; CHECK: define void @inactiveSecond
; CHECK-NEXT: entry
; CHECK: call void @[[inactiveSecond:.+]](


; CHECK: define void @activeMod
; CHECK-NEXT: entry
; CHECK: call void @[[activeMod:.+]](

; CHECK: define void @inactiveModFirst
; CHECK-NEXT: entry
; CHECK: call void @[[inactiveModFirst:.+]](

; CHECK: define void @inactiveModSecond
; CHECK-NEXT: entry
; CHECK: call void @[[inactiveModSecond:.+]](


; CHECK: define internal void @[[active]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"m'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double*] %"n'", 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %2, double* %n, i32 %incn, double* %0, i32 %incm)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %2, double* %m, i32 %incm, double* %1, i32 %incn)
; CHECK-NEXT:   %3 = extractvalue [2 x double*] %"m'", 1
; CHECK-NEXT:   %4 = extractvalue [2 x double*] %"n'", 1
; CHECK-NEXT:   %5 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %5, double* %n, i32 %incn, double* %3, i32 %incm)
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %5, double* %m, i32 %incm, double* %4, i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"n'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %1, double* %m, i32 %incm, double* %0, i32 %incn)
; CHECK-NEXT:   %2 = extractvalue [2 x double*] %"n'", 1
; CHECK-NEXT:   %3 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %3, double* %m, i32 %incm, double* %2, i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = extractvalue [2 x double*] %"m'", 0
; CHECK-NEXT:   %1 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %1, double* %n, i32 %incn, double* %0, i32 %incm)
; CHECK-NEXT:   %2 = extractvalue [2 x double*] %"m'", 1
; CHECK-NEXT:   %3 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %3, double* %n, i32 %incn, double* %2, i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call { double*, double* } @[[augMod:.+]](i32 %len, double* %m, [2 x double*] %"m'", i32 %incm, double* %n, [2 x double*] %"n'", i32 %incn)
; CHECK:        call void @[[revMod:.+]](i32 %len, double* %m, [2 x double*] %"m'", i32 %incm, double* %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn, { double*, double* } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @[[augMod]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %len, double* %m, i32 %incm, double* %cache.x, i32 1)
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i32 %len, 8
; CHECK-NEXT:   %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:   %cache.y = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:   call void @cblas_dcopy(i32 %len, double* %n, i32 %incn, double* %cache.y, i32 1)
; CHECK-NEXT:   %0 = insertvalue { double*, double* } undef, double* %cache.x, 0
; CHECK-NEXT:   %1 = insertvalue { double*, double* } %0, double* %cache.y, 1
; CHECK-NEXT:   ret { double*, double* } %1
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, [2 x double*] %"n'", i32 %incn)
; CHECK:        call void @[[revModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:  %cache.x = bitcast i8* %malloccall to double*
; CHECK-NEXT:  call void @cblas_dcopy(i32 %len, double* %m, i32 %incm, double* %cache.x, i32 1)
; CHECK-NEXT:  ret double* %cache.x
; CHECK-NEXT: }

; CHECK: define internal void @[[revModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, [2 x double*] %"n'", i32 %incn, [2 x double] %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue [2 x double*] %"n'", 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %2, double* %0, i32 1, double* %1, i32 %incn)
; CHECK-NEXT:   %3 = extractvalue [2 x double*] %"n'", 1
; CHECK-NEXT:   %4 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %4, double* %0, i32 1, double* %3, i32 %incn)
; CHECK-NEXT:   %5 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %5)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn, [2 x double] %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModSecond:.+]](i32 %len, double* %m, [2 x double*] %"m'", i32 %incm, double* %n, i32 %incn)
; CHECK:        call void @[[revModSecond:.+]](i32 %len, double* %m, [2 x double*] %"m'", i32 %incm, double* %n, i32 %incn, [2 x double] %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:  %cache.y = bitcast i8* %malloccall to double*
; CHECK-NEXT:  call void @cblas_dcopy(i32 %len, double* %n, i32 %incn, double* %cache.y, i32 1)
; CHECK-NEXT:  ret double* %cache.y
; CHECK-NEXT: }


; CHECK: define internal void @[[revModSecond]](i32 %len, double* noalias %m, [2 x double*] %"m'", i32 %incm, double* noalias %n, i32 %incn, [2 x double] %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue [2 x double*] %"m'", 0
; CHECK-NEXT:   %2 = extractvalue [2 x double] %differeturn, 0
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %2, double* %0, i32 1, double* %1, i32 %incm)
; CHECK-NEXT:   %3 = extractvalue [2 x double*] %"m'", 1
; CHECK-NEXT:   %4 = extractvalue [2 x double] %differeturn, 1
; CHECK-NEXT:   call void @cblas_daxpy(i32 %len, double %4, double* %0, i32 1, double* %3, i32 %incm)
; CHECK-NEXT:   %5 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %5)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


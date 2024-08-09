;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare double @cblas_ddot64_(i64, double*, i64, double*, i64)

define void @active(i64 %len, double* noalias %m, double* %dm, i64 %incm, double* noalias %n, double* %dn, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i64, double*, i64, double*, i64)* @f, i64 %len, double* noalias %m, double* %dm, i64 %incm, double* noalias %n, double* %dn, i64 %incn)
  ret void
}

define void @inactiveFirst(i64 %len, double* noalias %m, i64 %incm, double* noalias %n, double* %dn, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i64, double*, i64, double*, i64)* @f, i64 %len, metadata !"enzyme_const", double* noalias %m, i64 %incm, double* noalias %n, double* %dn, i64 %incn)
  ret void
}

define void @inactiveSecond(i64 %len, double* noalias %m, double* noalias %dm, i64 %incm, double* noalias %n, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i64, double*, i64, double*, i64)* @f, i64 %len, double* noalias %m, double* noalias %dm, i64 %incm, metadata !"enzyme_const", double* noalias %n, i64 %incn)
  ret void
}

define void @activeMod(i64 %len, double* noalias %m, double* %dm, i64 %incm, double* noalias %n, double* %dn, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i64, double*, i64, double*, i64)* @modf, i64 %len, double* noalias %m, double* %dm, i64 %incm, double* noalias %n, double* %dn, i64 %incn)
  ret void
}

define void @inactiveModFirst(i64 %len, double* noalias %m, i64 %incm, double* noalias %n, double* %dn, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i64, double*, i64, double*, i64)* @modf, i64 %len, metadata !"enzyme_const", double* noalias %m, i64 %incm, double* noalias %n, double* %dn, i64 %incn)
  ret void
}

define void @inactiveModSecond(i64 %len, double* noalias %m, double* noalias %dm, i64 %incm, double* noalias %n, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i64, double*, i64, double*, i64)* @modf, i64 %len, double* noalias %m, double* noalias %dm, i64 %incm, metadata !"enzyme_const", double* noalias %n, i64 %incn)
  ret void
}

define double @f(i64 %len, double* noalias %m, i64 %incm, double* noalias %n, i64 %incn) {
entry:
  %call = call double @cblas_ddot64_(i64 %len, double* %m, i64 %incm, double* %n, i64 %incn)
  ret double %call
}

define double @modf(i64 %len, double* noalias %m, i64 %incm, double* noalias %n, i64 %incn) {
entry:
  %call = call double @f(i64 %len, double* %m, i64 %incm, double* %n, i64 %incn)
  store double 0.000000e+00, double* %m
  store double 0.000000e+00, double* %n
  ret double %call
}

; COM: Can't check the attrs since number and order depends on llvm version
; COM: ; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind readonly willreturn
; CHECK: declare double @cblas_ddot64_(i64 "enzyme_inactive", double* nocapture readonly, i64 "enzyme_inactive", double* nocapture readonly, i64 "enzyme_inactive")

; CHECK: define void @active
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[active:.+]](

; CHECK: define void @inactiveFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveFirst:.+]](

; CHECK: define void @inactiveSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveSecond:.+]](


; CHECK: define void @activeMod
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[activeMod:.+]](

; CHECK: define void @inactiveModFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveModFirst:.+]](

; CHECK: define void @inactiveModSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveModSecond:.+]](

; CHECK: define internal void @[[active]](i64 %len, double* noalias %m, double* %"m'", i64 %incm, double* noalias %n, double* %"n'", i64 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i64 %len, double %differeturn, double* %n, i64 %incn, double* %"m'", i64 %incm)
; CHECK-NEXT:   call void @cblas_daxpy64_(i64 %len, double %differeturn, double* %m, i64 %incm, double* %"n'", i64 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; COM: Can't check the attrs since number and order depends on llvm version
; COM: ; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn
; CHECK: declare void @cblas_daxpy64_(i64 "enzyme_inactive", double, double* nocapture readonly, i64 "enzyme_inactive", double* nocapture, i64 "enzyme_inactive")

; CHECK: define internal void @[[inactiveFirst]](i64 %len, double* noalias %m, i64 %incm, double* noalias %n, double* %"n'", i64 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i64 %len, double %differeturn, double* %m, i64 %incm, double* %"n'", i64 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i64 %len, double* noalias %m, double* %"m'", i64 %incm, double* noalias %n, i64 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i64 %len, double %differeturn, double* %n, i64 %incn, double* %"m'", i64 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i64 %len, double* noalias %m, double* %"m'", i64 %incm, double* noalias %n, double* %"n'", i64 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call { double*, double* } @[[augMod:.+]](i64 %len, double* %m, double* %"m'", i64 %incm, double* %n, double* %"n'", i64 %incn)
; CHECK:        call void @[[revMod:.+]](i64 %len, double* %m, double* %"m'", i64 %incm, double* %n, double* %"n'", i64 %incn, double %differeturn, { double*, double* } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @[[augMod]](i64 %len, double* noalias %m, double* %"m'", i64 %incm, double* noalias %n, double* %"n'", i64 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i64 %len, 8
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:  %cache.x = bitcast i8* %malloccall to double*
; CHECK-NEXT:  call void @cblas_dcopy64_(i64 %len, double* %m, i64 %incm, double* %cache.x, i64 1)
; CHECK-NEXT:  %mallocsize1 = mul nuw nsw i64 %len, 8
; CHECK-NEXT:  %malloccall2 = tail call noalias nonnull i8* @malloc(i64 %mallocsize1)
; CHECK-NEXT:  %cache.y = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:  call void @cblas_dcopy64_(i64 %len, double* %n, i64 %incn, double* %cache.y, i64 1)
; CHECK-NEXT:  %0 = insertvalue { double*, double* } undef, double* %cache.x, 0
; CHECK-NEXT:  %1 = insertvalue { double*, double* } %0, double* %cache.y, 1
; CHECK-NEXT:  ret { double*, double* } %1
; CHECK-NEXT: }

; CHECK: define internal void @[[revMod]](i64 %len, double* noalias %m, double* %"m'", i64 %incm, double* noalias %n, double* %"n'", i64 %incn, double %differeturn, { double*, double* }
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tape.ext.x = extractvalue { double*, double* } %0, 0
; CHECK-NEXT:   %tape.ext.y = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   call void @cblas_daxpy64_(i64 %len, double %differeturn, double* %tape.ext.y, i64 1, double* %"m'", i64 %incm)
; CHECK-NEXT:   call void @cblas_daxpy64_(i64 %len, double %differeturn, double* %tape.ext.x, i64 1, double* %"n'", i64 %incn)
; CHECK-NEXT:   %1 = bitcast double* %tape.ext.x to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   %2 = bitcast double* %tape.ext.y to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModFirst]](i64 %len, double* noalias %m, i64 %incm, double* noalias %n, double* %"n'", i64 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModFirst:.+]](i64 %len, double* %m, i64 %incm, double* %n, double* %"n'", i64 %incn)
; CHECK:        call void @[[revModFirst:.+]](i64 %len, double* %m, i64 %incm, double* %n, double* %"n'", i64 %incn, double %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @augmented_f.6(i64 %len, double* noalias %m, i64 %incm, double* noalias %n, double* %"n'", i64 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i64 %len, 8
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:  %cache.x = bitcast i8* %malloccall to double*
; CHECK-NEXT:  call void @cblas_dcopy64_(i64 %len, double* %m, i64 %incm, double* %cache.x, i64 1)
; CHECK-NEXT:  ret double* %cache.x
; CHECK-NEXT: }

; CHECK: define internal void @[[revModFirst]](i64 %len, double* noalias %m, i64 %incm, double* noalias %n, double* %"n'", i64 %incn, double %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i64 %len, double %differeturn, double* %0, i64 1, double* %"n'", i64 %incn)
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModSecond]](i64 %len, double* noalias %m, double* %"m'", i64 %incm, double* noalias %n, i64 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModSecond:.+]](i64 %len, double* %m, double* %"m'", i64 %incm, double* %n, i64 %incn)
; CHECK:        call void @[[revModSecond:.+]](i64 %len, double* %m, double* %"m'", i64 %incm, double* %n, i64 %incn, double %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModSecond]](i64 %len, double* noalias %m, double* %"m'", i64 %incm, double* noalias %n, i64 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i64 %len, 8
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i64 %mallocsize)
; CHECK-NEXT:  %cache.y = bitcast i8* %malloccall to double*
; CHECK-NEXT:  call void @cblas_dcopy64_(i64 %len, double* %n, i64 %incn, double* %cache.y, i64 1)
; CHECK-NEXT:  ret double* %cache.y
; CHECK-NEXT: }


; CHECK: define internal void @[[revModSecond]](i64 %len, double* noalias %m, double* %"m'", i64 %incm, double* noalias %n, i64 %incn, double %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i64 %len, double %differeturn, double* %0, i64 1, double* %"m'", i64 %incm)
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


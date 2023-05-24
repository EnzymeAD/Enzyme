;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare double @cblas_ddot64_(i32, double*, i32, double*, i32)

define void @active(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @f, i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define void @activeMod(i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, double* noalias %m, double* %dm, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModFirst(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, metadata !"enzyme_const", double* noalias %m, i32 %incm, double* noalias %n, double* %dn, i32 %incn)
  ret void
}

define void @inactiveModSecond(i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(double (i32, double*, i32, double*, i32)* @modf, i32 %len, double* noalias %m, double* noalias %dm, i32 %incm, metadata !"enzyme_const", double* noalias %n, i32 %incn)
  ret void
}

define double @f(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @cblas_ddot64_(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  ret double %call
}

define double @modf(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, i32 %incn) {
entry:
  %call = call double @f(i32 %len, double* %m, i32 %incm, double* %n, i32 %incn)
  store double 0.000000e+00, double* %m
  store double 0.000000e+00, double* %n
  ret double %call
}

; COM: Can't check the attrs since number and order depends on llvm version
; COM: ; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind readonly willreturn
; CHECK: declare double @cblas_ddot64_(i32, double* nocapture readonly, i32, double* nocapture readonly, i32)

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

; CHECK: define internal void @[[active]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i32 %len, double %differeturn, double* %n, i32 %incn, double* %"m'", i32 %incm)
; CHECK-NEXT:   call void @cblas_daxpy64_(i32 %len, double %differeturn, double* %m, i32 %incm, double* %"n'", i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; COM: Can't check the attrs since number and order depends on llvm version
; COM: ; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn
; CHECK: declare void @cblas_daxpy64_(i32, double, double* nocapture readonly, i32, double* nocapture, i32)

; CHECK: define internal void @[[inactiveFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i32 %len, double %differeturn, double* %m, i32 %incm, double* %"n'", i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i32 %len, double %differeturn, double* %n, i32 %incn, double* %"m'", i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call { double*, double* } @[[augMod:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, double* %"n'", i32 %incn)
; CHECK:        call void @[[revMod:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, double* %"n'", i32 %incn, double %differeturn, { double*, double* } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal { double*, double* } @[[augMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:  %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:  call void @cblas_dcopy64_(i32 %len, double* %m, i32 %incm, double* %0, i32 1)
; CHECK-NEXT:  %mallocsize1 = mul nuw nsw i32 %len, 8
; CHECK-NEXT:  %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:  %1 = bitcast i8* %malloccall2 to double*
; CHECK-NEXT:  call void @cblas_dcopy64_(i32 %len, double* %n, i32 %incn, double* %1, i32 1)
; CHECK-NEXT:  %2 = insertvalue { double*, double* } undef, double* %0, 0
; CHECK-NEXT:  %3 = insertvalue { double*, double* } %2, double* %1, 1
; CHECK-NEXT:  ret { double*, double* } %3
; CHECK-NEXT: }

; CHECK: define internal void @[[revMod]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn, { double*, double* }
; CHECK-NEXT: entry:
; CHECK-NEXT:   %1 = extractvalue { double*, double* } %0, 0
; CHECK-NEXT:   %2 = extractvalue { double*, double* } %0, 1
; CHECK-NEXT:   call void @cblas_daxpy64_(i32 %len, double %differeturn, double* %2, i32 1, double* %"m'", i32 %incm)
; CHECK-NEXT:   call void @cblas_daxpy64_(i32 %len, double %differeturn, double* %1, i32 1, double* %"n'", i32 %incn)
; CHECK-NEXT:   %3 = bitcast double* %1 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %3)
; CHECK-NEXT:   %4 = bitcast double* %2 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %4)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, double* %"n'", i32 %incn)
; CHECK:        call void @[[revModFirst:.+]](i32 %len, double* %m, i32 %incm, double* %n, double* %"n'", i32 %incn, double %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @augmented_f.6(i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:  %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:  call void @cblas_dcopy64_(i32 %len, double* %m, i32 %incm, double* %0, i32 1)
; CHECK-NEXT:  ret double* %0
; CHECK-NEXT: }

; CHECK: define internal void @[[revModFirst]](i32 %len, double* noalias %m, i32 %incm, double* noalias %n, double* %"n'", i32 %incn, double %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i32 %len, double %differeturn, double* %0, i32 1, double* %"n'", i32 %incn)
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call double* @[[augModSecond:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, i32 %incn)
; CHECK:        call void @[[revModSecond:.+]](i32 %len, double* %m, double* %"m'", i32 %incm, double* %n, i32 %incn, double %differeturn, double* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal double* @[[augModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i32 %len, 8
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:  %0 = bitcast i8* %malloccall to double*
; CHECK-NEXT:  call void @cblas_dcopy64_(i32 %len, double* %n, i32 %incn, double* %0, i32 1)
; CHECK-NEXT:  ret double* %0
; CHECK-NEXT: }


; CHECK: define internal void @[[revModSecond]](i32 %len, double* noalias %m, double* %"m'", i32 %incm, double* noalias %n, i32 %incn, double %differeturn, double*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_daxpy64_(i32 %len, double %differeturn, double* %0, i32 1, double* %"m'", i32 %incm)
; CHECK-NEXT:   %1 = bitcast double* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


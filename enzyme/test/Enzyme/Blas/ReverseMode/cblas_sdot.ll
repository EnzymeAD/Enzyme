;RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
;RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare float @cblas_sdot(i32, float*, i32, float*, i32)

define void @active(i32 %len, float* noalias %m, float* %dm, i32 %incm, float* noalias %n, float* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @f, i32 %len, float* noalias %m, float* %dm, i32 %incm, float* noalias %n, float* %dn, i32 %incn)
  ret void
}

define void @inactiveFirst(i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @f, i32 %len, metadata !"enzyme_const", float* noalias %m, i32 %incm, float* noalias %n, float* %dn, i32 %incn)
  ret void
}

define void @inactiveSecond(i32 %len, float* noalias %m, float* noalias %dm, i32 %incm, float* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @f, i32 %len, float* noalias %m, float* noalias %dm, i32 %incm, metadata !"enzyme_const", float* noalias %n, i32 %incn)
  ret void
}

define void @activeMod(i32 %len, float* noalias %m, float* %dm, i32 %incm, float* noalias %n, float* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @modf, i32 %len, float* noalias %m, float* %dm, i32 %incm, float* noalias %n, float* %dn, i32 %incn)
  ret void
}

define void @inactiveModFirst(i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %dn, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @modf, i32 %len, metadata !"enzyme_const", float* noalias %m, i32 %incm, float* noalias %n, float* %dn, i32 %incn)
  ret void
}

define void @inactiveModSecond(i32 %len, float* noalias %m, float* noalias %dm, i32 %incm, float* noalias %n, i32 %incn) {
entry:
  call void (...) @__enzyme_autodiff(float (i32, float*, i32, float*, i32)* @modf, i32 %len, float* noalias %m, float* noalias %dm, i32 %incm, metadata !"enzyme_const", float* noalias %n, i32 %incn)
  ret void
}

define float @f(i32 %len, float* noalias %m, i32 %incm, float* noalias %n, i32 %incn) {
entry:
  %call = call float @cblas_sdot(i32 %len, float* %m, i32 %incm, float* %n, i32 %incn)
  ret float %call
}

define float @modf(i32 %len, float* noalias %m, i32 %incm, float* noalias %n, i32 %incn) {
entry:
  %call = call float @f(i32 %len, float* %m, i32 %incm, float* %n, i32 %incn)
  store float 0.000000e+00, float* %m
  store float 0.000000e+00, float* %n
  ret float %call
}


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


; CHECK: define internal void @[[active]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %n, i32 %incn, float* %"m'", i32 %incm)
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %m, i32 %incm, float* %"n'", i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveFirst]](i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %m, i32 %incm, float* %"n'", i32 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %n, i32 %incn, float* %"m'", i32 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call { float*, float* } @[[augMod:.+]](i32 %len, float* %m, float* %"m'", i32 %incm, float* %n, float* %"n'", i32 %incn)
; CHECK:        call void @[[revMod:.+]](i32 %len, float* %m, float* %"m'", i32 %incm, float* %n, float* %"n'", i32 %incn, float %differeturn, { float*, float* } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
; CHECK: define internal { float*, float* } @[[augMod]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, float* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i32 %len, 4
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:  %cache.x = bitcast i8* %malloccall to float*
; CHECK-NEXT:  call void @cblas_scopy(i32 %len, float* %m, i32 %incm, float* %cache.x, i32 1)
; CHECK-NEXT:  %mallocsize1 = mul nuw nsw i32 %len, 4
; CHECK-NEXT:  %malloccall2 = tail call noalias nonnull i8* @malloc(i32 %mallocsize1)
; CHECK-NEXT:  %cache.y = bitcast i8* %malloccall2 to float*
; CHECK-NEXT:  call void @cblas_scopy(i32 %len, float* %n, i32 %incn, float* %cache.y, i32 1)
; CHECK-NEXT:  %0 = insertvalue { float*, float* } undef, float* %cache.x, 0
; CHECK-NEXT:  %1 = insertvalue { float*, float* } %0, float* %cache.y, 1
; CHECK-NEXT:  ret { float*, float* } %1
; CHECK-NEXT:  }

; CHECK: define internal void @[[revMod]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn, { float*, float* }
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tape.ext.x = extractvalue { float*, float* } %0, 0
; CHECK-NEXT:   %tape.ext.y = extractvalue { float*, float* } %0, 1
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %tape.ext.y, i32 1, float* %"m'", i32 %incm)
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %tape.ext.x, i32 1, float* %"n'", i32 %incn)
; CHECK-NEXT:   %1 = bitcast float* %tape.ext.x to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   %2 = bitcast float* %tape.ext.y to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %2)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModFirst]](i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call float* @[[augModFirst:.+]](i32 %len, float* %m, i32 %incm, float* %n, float* %"n'", i32 %incn)
; CHECK:        call void @[[revModFirst:.+]](i32 %len, float* %m, i32 %incm, float* %n, float* %"n'", i32 %incn, float %differeturn, float* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal float* @[[augModFirst]](i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %"n'", i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %mallocsize = mul nuw nsw i32 %len, 4
; CHECK-NEXT:   %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:   %cache.x = bitcast i8* %malloccall to float*
; CHECK-NEXT:   call void @cblas_scopy(i32 %len, float* %m, i32 %incm, float* %cache.x, i32 1)
; CHECK-NEXT:   ret float* %cache.x
; CHECK-NEXT: }

; CHECK: define internal void @[[revModFirst]](i32 %len, float* noalias %m, i32 %incm, float* noalias %n, float* %"n'", i32 %incn, float %differeturn, float*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %0, i32 1, float* %"n'", i32 %incn)
; CHECK-NEXT:   %1 = bitcast float* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveModSecond]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, i32 %incn, float %differeturn)
; CHECK-NEXT: entry:
; CHECK:        %call_augmented = call float* @[[augModSecond:.+]](i32 %len, float* %m, float* %"m'", i32 %incm, float* %n, i32 %incn)
; CHECK:        call void @[[revModSecond:.+]](i32 %len, float* %m, float* %"m'", i32 %incm, float* %n, i32 %incn, float %differeturn, float* %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal float* @[[augModSecond]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, i32 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:  %mallocsize = mul nuw nsw i32 %len, 4
; CHECK-NEXT:  %malloccall = tail call noalias nonnull i8* @malloc(i32 %mallocsize)
; CHECK-NEXT:  %cache.y = bitcast i8* %malloccall to float*
; CHECK-NEXT:  call void @cblas_scopy(i32 %len, float* %n, i32 %incn, float* %cache.y, i32 1)
; CHECK-NEXT:  ret float* %cache.y
; CHECK-NEXT: }


; CHECK: define internal void @[[revModSecond]](i32 %len, float* noalias %m, float* %"m'", i32 %incm, float* noalias %n, i32 %incn, float %differeturn, float*
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @cblas_saxpy(i32 %len, float %differeturn, float* %0, i32 1, float* %"m'", i32 %incm)
; CHECK-NEXT:   %1 = bitcast float* %0 to i8*
; CHECK-NEXT:   tail call void @free(i8* nonnull %1)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


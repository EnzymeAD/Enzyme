;RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S -enzyme-detect-readthrow=0 | FileCheck %s

; Complex unconjugated dot product, result returned through the last pointer arg.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local void @__enzyme_autodiff(...)

declare void @cblas_zdotu_sub64_(i64, <2 x double>*, i64, <2 x double>*, i64, <2 x double>*)

define void @active(i64 %len, <2 x double>* noalias %m, <2 x double>* %dm, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(<2 x double> (i64, <2 x double>*, i64, <2 x double>*, i64)* @f, i64 %len, <2 x double>* noalias %m, <2 x double>* %dm, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn)
  ret void
}

define void @inactiveFirst(i64 %len, <2 x double>* noalias %m, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(<2 x double> (i64, <2 x double>*, i64, <2 x double>*, i64)* @f, i64 %len, metadata !"enzyme_const", <2 x double>* noalias %m, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn)
  ret void
}

define void @inactiveSecond(i64 %len, <2 x double>* noalias %m, <2 x double>* noalias %dm, i64 %incm, <2 x double>* noalias %n, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(<2 x double> (i64, <2 x double>*, i64, <2 x double>*, i64)* @f, i64 %len, <2 x double>* noalias %m, <2 x double>* noalias %dm, i64 %incm, metadata !"enzyme_const", <2 x double>* noalias %n, i64 %incn)
  ret void
}

define void @activeMod(i64 %len, <2 x double>* noalias %m, <2 x double>* %dm, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn) {
entry:
  call void (...) @__enzyme_autodiff(<2 x double> (i64, <2 x double>*, i64, <2 x double>*, i64)* @modf, i64 %len, <2 x double>* noalias %m, <2 x double>* %dm, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn)
  ret void
}

define <2 x double> @f(i64 %len, <2 x double>* noalias %m, i64 %incm, <2 x double>* noalias %n, i64 %incn) {
entry:
  %res = alloca <2 x double>, align 8
  call void @cblas_zdotu_sub64_(i64 %len, <2 x double>* %m, i64 %incm, <2 x double>* %n, i64 %incn, <2 x double>* %res)
  %r = load <2 x double>, <2 x double>* %res, align 8
  ret <2 x double> %r
}

define <2 x double> @modf(i64 %len, <2 x double>* noalias %m, i64 %incm, <2 x double>* noalias %n, i64 %incn) {
entry:
  %call = call <2 x double> @f(i64 %len, <2 x double>* %m, i64 %incm, <2 x double>* %n, i64 %incn)
  store <2 x double> zeroinitializer, <2 x double>* %m
  store <2 x double> zeroinitializer, <2 x double>* %n
  ret <2 x double> %call
}

; COM: Pointer types and capture attributes are matched loosely since their
; COM: spelling depends on the llvm version.
; CHECK: declare void @cblas_zdotu_sub64_(i64 "enzyme_inactive", [[PTR:(<2 x double>\*|ptr)]] [[RO:(nocapture readonly|readonly captures\(none\))]], i64 "enzyme_inactive", [[PTR]] [[RO]], i64 "enzyme_inactive", [[PTR]] [[NC:(nocapture|captures\(none\))]])

; CHECK: define void @active
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[active:[^(]+]](

; CHECK: define void @inactiveFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveFirst:[^(]+]](

; CHECK: define void @inactiveSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[inactiveSecond:[^(]+]](

; CHECK: define void @activeMod
; CHECK-NEXT: entry
; CHECK-NEXT: call void @[[activeMod:[^(]+]](

; COM: dx += dres * y ; dy += dres * x, with dres taken from (and reset in) the shadow of res.
; CHECK: define internal void @[[active]](i64 %len, [[PTR]] noalias %m, [[PTR]] %"m'", i64 %incm, [[PTR]] noalias %n, [[PTR]] %"n'", i64 %incn, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca <2 x double>, align 16
; CHECK-NEXT:   %res = alloca <2 x double>, align 8
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %res)
; CHECK-NEXT:   store <2 x double> %differeturn, [[PTR]] %ret, align 16
; CHECK-NEXT:   call void @cblas_zaxpy64_(i64 %len, [[PTR]] %ret, [[PTR]] %n, i64 %incn, [[PTR]] %"m'", i64 %incm)
; CHECK-NEXT:   call void @cblas_zaxpy64_(i64 %len, [[PTR]] %ret, [[PTR]] %m, i64 %incm, [[PTR]] %"n'", i64 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; COM: complex alpha is passed by pointer under the cblas abi
; CHECK: declare void @cblas_zaxpy64_(i64 "enzyme_inactive", [[PTR]] [[RO]], [[PTR]] [[RO]], i64 "enzyme_inactive", [[PTR]] [[NC]], i64 "enzyme_inactive")

; CHECK: define internal void @[[inactiveFirst]](i64 %len, [[PTR]] noalias %m, i64 %incm, [[PTR]] noalias %n, [[PTR]] %"n'", i64 %incn, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca <2 x double>, align 16
; CHECK-NEXT:   %res = alloca <2 x double>, align 8
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %res)
; CHECK-NEXT:   store <2 x double> %differeturn, [[PTR]] %ret, align 16
; CHECK-NEXT:   call void @cblas_zaxpy64_(i64 %len, [[PTR]] %ret, [[PTR]] %m, i64 %incm, [[PTR]] %"n'", i64 %incn)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[inactiveSecond]](i64 %len, [[PTR]] noalias %m, [[PTR]] %"m'", i64 %incm, [[PTR]] noalias %n, i64 %incn, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca <2 x double>, align 16
; CHECK-NEXT:   %res = alloca <2 x double>, align 8
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %res)
; CHECK-NEXT:   store <2 x double> %differeturn, [[PTR]] %ret, align 16
; CHECK-NEXT:   call void @cblas_zaxpy64_(i64 %len, [[PTR]] %ret, [[PTR]] %n, i64 %incn, [[PTR]] %"m'", i64 %incm)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @[[activeMod]](i64 %len, [[PTR]] noalias %m, [[PTR]] %"m'", i64 %incm, [[PTR]] noalias %n, [[PTR]] %"n'", i64 %incn, <2 x double> %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call { [[PTR]], [[PTR]] } @[[augMod:[^(]+]](i64 %len, [[PTR]] %m, [[PTR]] %"m'", i64 %incm, [[PTR]] %n, [[PTR]] %"n'", i64 %incn)
; CHECK:        call void @[[revMod:[^(]+]](i64 %len, [[PTR]] %m, [[PTR]] %"m'", i64 %incm, [[PTR]] %n, [[PTR]] %"n'", i64 %incn, <2 x double> %differeturn, { [[PTR]], [[PTR]] } %call_augmented)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; COM: x and y are overwritten later and must be cached (16 bytes per complex element)
; CHECK: define internal { [[PTR]], [[PTR]] } @[[augMod]](i64 %len, [[PTR]] noalias %m, [[PTR]] %"m'", i64 %incm, [[PTR]] noalias %n, [[PTR]] %"n'", i64 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %res = alloca <2 x double>, i64 1, align 8
; CHECK-NEXT:   %mallocsize = mul nuw nsw i64 %len, 16
; CHECK:        @malloc(i64 %mallocsize)
; CHECK:        call void @cblas_zcopy64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %cache.x, i64 1)
; CHECK-NEXT:   %mallocsize1 = mul nuw nsw i64 %len, 16
; CHECK:        @malloc(i64 %mallocsize1)
; CHECK:        call void @cblas_zcopy64_(i64 %len, [[PTR]] %n, i64 %incn, [[PTR]] %cache.y, i64 1)
; CHECK:        call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %res)
; CHECK-NEXT:   ret { [[PTR]], [[PTR]] }
; CHECK-NEXT: }

; CHECK: define internal void @[[revMod]](i64 %len, [[PTR]] noalias %m, [[PTR]] %"m'", i64 %incm, [[PTR]] noalias %n, [[PTR]] %"n'", i64 %incn, <2 x double> %differeturn, { [[PTR]], [[PTR]] }
; CHECK-NEXT: entry:
; CHECK-NEXT:   %ret = alloca <2 x double>, align 16
; CHECK-NEXT:   %"res'mi" = alloca <2 x double>, i64 1, align 8
; CHECK:        %1 = load <2 x double>, [[PTR]] %"res'mi", align 8
; CHECK-NEXT:   %2 = fadd fast <2 x double> %1, %differeturn
; CHECK-NEXT:   store <2 x double> %2, [[PTR]] %"res'mi", align 8
; CHECK-NEXT:   %tape.ext.x = extractvalue { [[PTR]], [[PTR]] } %0, 0
; CHECK-NEXT:   %tape.ext.y = extractvalue { [[PTR]], [[PTR]] } %0, 1
; CHECK-NEXT:   %3 = load <2 x double>, [[PTR]] %"res'mi", align 8
; CHECK-NEXT:   store <2 x double> zeroinitializer, [[PTR]] %"res'mi", align 8
; CHECK-NEXT:   store <2 x double> %3, [[PTR]] %ret, align 16
; CHECK-NEXT:   call void @cblas_zaxpy64_(i64 %len, [[PTR]] %ret, [[PTR]] %tape.ext.y, i64 1, [[PTR]] %"m'", i64 %incm)
; CHECK-NEXT:   call void @cblas_zaxpy64_(i64 %len, [[PTR]] %ret, [[PTR]] %tape.ext.x, i64 1, [[PTR]] %"n'", i64 %incn)
; CHECK:        @free(
; CHECK:        @free(
; CHECK:        ret void
; CHECK-NEXT: }

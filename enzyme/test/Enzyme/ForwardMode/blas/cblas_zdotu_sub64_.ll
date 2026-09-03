; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -enzyme-detect-readthrow=0 -S | FileCheck %s

; Complex unconjugated dot product, result returned through the last pointer arg.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare dso_local <2 x double> @__enzyme_fwddiff(...)

declare void @cblas_zdotu_sub64_(i64, <2 x double>*, i64, <2 x double>*, i64, <2 x double>*)

define <2 x double> @active(i64 %len, <2 x double>* noalias %m, <2 x double>* %dm, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn) {
entry:
  %r = call <2 x double> (...) @__enzyme_fwddiff(<2 x double> (i64, <2 x double>*, i64, <2 x double>*, i64)* @f, i64 %len, <2 x double>* noalias %m, <2 x double>* %dm, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn)
  ret <2 x double> %r
}

define <2 x double> @inactiveFirst(i64 %len, <2 x double>* noalias %m, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn) {
entry:
  %r = call <2 x double> (...) @__enzyme_fwddiff(<2 x double> (i64, <2 x double>*, i64, <2 x double>*, i64)* @f, i64 %len, metadata !"enzyme_const", <2 x double>* noalias %m, i64 %incm, <2 x double>* noalias %n, <2 x double>* %dn, i64 %incn)
  ret <2 x double> %r
}

define <2 x double> @inactiveSecond(i64 %len, <2 x double>* noalias %m, <2 x double>* noalias %dm, i64 %incm, <2 x double>* noalias %n, i64 %incn) {
entry:
  %r = call <2 x double> (...) @__enzyme_fwddiff(<2 x double> (i64, <2 x double>*, i64, <2 x double>*, i64)* @f, i64 %len, <2 x double>* noalias %m, <2 x double>* noalias %dm, i64 %incm, metadata !"enzyme_const", <2 x double>* noalias %n, i64 %incn)
  ret <2 x double> %r
}

define <2 x double> @f(i64 %len, <2 x double>* noalias %m, i64 %incm, <2 x double>* noalias %n, i64 %incn) {
entry:
  %res = alloca <2 x double>, align 8
  call void @cblas_zdotu_sub64_(i64 %len, <2 x double>* %m, i64 %incm, <2 x double>* %n, i64 %incn, <2 x double>* %res)
  %r = load <2 x double>, <2 x double>* %res, align 8
  ret <2 x double> %r
}

; COM: Pointer types and capture attributes are matched loosely since their
; COM: spelling depends on the llvm version.
; CHECK: declare void @cblas_zdotu_sub64_(i64 "enzyme_inactive", [[PTR:(<2 x double>\*|ptr)]] {{(nocapture readonly|readonly captures\(none\))}}, i64 "enzyme_inactive", [[PTR]] {{(nocapture readonly|readonly captures\(none\))}}, i64 "enzyme_inactive", [[PTR]] {{(nocapture|captures\(none\))}})

; CHECK: define <2 x double> @active
; CHECK-NEXT: entry
; CHECK-NEXT: call fast <2 x double> @[[active:[^(]+]](

; CHECK: define <2 x double> @inactiveFirst
; CHECK-NEXT: entry
; CHECK-NEXT: call fast <2 x double> @[[inactiveFirst:[^(]+]](

; CHECK: define <2 x double> @inactiveSecond
; CHECK-NEXT: entry
; CHECK-NEXT: call fast <2 x double> @[[inactiveSecond:[^(]+]](

; COM: dres = dotu(dx, y) + dotu(x, dy), written to the shadow of res (promoted by mem2reg)
; CHECK: define internal <2 x double> @[[active]](i64 %len, [[PTR]] noalias %m, [[PTR]] %"m'", i64 %incm, [[PTR]] noalias %n, [[PTR]] %"n'", i64 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %dotu_sub.ret = alloca <2 x double>, align 16
; CHECK-NEXT:   %dotu_sub.ret1 = alloca <2 x double>, align 16
; CHECK-NEXT:   %res = alloca <2 x double>, align 8
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %"m'", i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %dotu_sub.ret)
; CHECK-NEXT:   %0 = load <2 x double>, [[PTR]] %dotu_sub.ret, align 16
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %"n'", i64 %incn, [[PTR]] %dotu_sub.ret1)
; CHECK-NEXT:   %1 = load <2 x double>, [[PTR]] %dotu_sub.ret1, align 16
; CHECK-NEXT:   %2 = fadd fast <2 x double> %0, %1
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %res)
; CHECK-NEXT:   ret <2 x double> %2
; CHECK-NEXT: }

; CHECK: define internal <2 x double> @[[inactiveFirst]](i64 %len, [[PTR]] noalias %m, i64 %incm, [[PTR]] noalias %n, [[PTR]] %"n'", i64 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %dotu_sub.ret = alloca <2 x double>, align 16
; CHECK-NEXT:   %res = alloca <2 x double>, align 8
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %"n'", i64 %incn, [[PTR]] %dotu_sub.ret)
; CHECK-NEXT:   %0 = load <2 x double>, [[PTR]] %dotu_sub.ret, align 16
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %res)
; CHECK-NEXT:   ret <2 x double> %0
; CHECK-NEXT: }

; CHECK: define internal <2 x double> @[[inactiveSecond]](i64 %len, [[PTR]] noalias %m, [[PTR]] %"m'", i64 %incm, [[PTR]] noalias %n, i64 %incn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %dotu_sub.ret = alloca <2 x double>, align 16
; CHECK-NEXT:   %res = alloca <2 x double>, align 8
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %"m'", i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %dotu_sub.ret)
; CHECK-NEXT:   %0 = load <2 x double>, [[PTR]] %dotu_sub.ret, align 16
; CHECK-NEXT:   call void @cblas_zdotu_sub64_(i64 %len, [[PTR]] %m, i64 %incm, [[PTR]] %n, i64 %incn, [[PTR]] %res)
; CHECK-NEXT:   ret <2 x double> %0
; CHECK-NEXT: }

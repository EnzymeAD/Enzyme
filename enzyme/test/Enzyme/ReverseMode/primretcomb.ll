; RUN: if [ %llvmver -ge 11 ] && [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instcombine -simplifycfg -S | FileCheck %s; fi
; RUN: if [ %llvmver -ge 11 ]; then %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instcombine,%simplifycfg)" -S | FileCheck %s ; fi

source_filename = "primretcomb.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

declare { double, double } @__enzyme_autodiff(...)

; Function Attrs: noinline
define internal fastcc void @setup(double* noalias nocapture nofree noundef nonnull writeonly %arg, double %i8) {
bb:
  store double %i8, double* %arg, align 8
  ret void
}

; Function Attrs: argmemonly nofree noinline nosync readonly
define internal fastcc double @ldcall(double* nocapture nofree noundef nonnull readonly %arg, i8 %v) unnamed_addr #5 {
bb:
  %i1 = load double, double* %arg, align 8
  ret double %i1
}

define double @julia___2564_inner.1(double %ai8) {
bb:
  %i = alloca double, align 8 
  call fastcc void @setup(double* noalias nocapture nofree noundef nonnull writeonly %i, double %ai8)
  %i6 = call fastcc double @ldcall(double* nocapture nofree noundef nonnull readonly %i, i8 0)
  %i8 = call fastcc double @ldcall(double* nocapture nofree noundef nonnull readonly %i, i8 1)
  %i9 = fadd double %i6, %i8
  %i11 = fadd double %i9, %ai8
  ret double %i11
}

define { double, double } @wrap(double %x) {
bb:
  %res = call { double, double } (...) @__enzyme_autodiff(double (double)* @julia___2564_inner.1, metadata !"enzyme_primal_return", double %x)
  ret { double, double } %res
}

attributes #5 = { argmemonly nofree noinline nosync readonly }

; CHECK: define internal { double, double } @diffejulia___2564_inner.1(double %ai8, double %differeturn)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"i'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"i'ipa", align 8
; CHECK-NEXT:   %i = alloca double, align 8
; CHECK-NEXT:   call fastcc void @augmented_setup(double* nocapture nofree writeonly %i, double* nocapture nofree nonnull %"i'ipa", double %ai8)
; CHECK-NEXT:   %0 = call fastcc { double } @diffeldcall(double* nocapture nofree readonly %i, double* nocapture nofree nonnull %"i'ipa", i8 1, double %differeturn)
; CHECK-NEXT:   %1 = extractvalue { double } %0, 0
; CHECK-NEXT:   %2 = call fastcc { double } @diffeldcall.1(double* nocapture nofree readonly %i, double* nocapture nofree nonnull %"i'ipa", i8 0, double %differeturn)
; CHECK-NEXT:   %3 = extractvalue { double } %2, 0
; CHECK-NEXT:   %i9 = fadd double %3, %1
; CHECK-NEXT:   %i11 = fadd double %i9, %ai8
; CHECK-NEXT:   %4 = call fastcc { double } @diffesetup(double* nocapture nofree writeonly undef, double* nocapture nofree nonnull %"i'ipa", double %ai8)
; CHECK-NEXT:   %5 = extractvalue { double } %4, 0
; CHECK-NEXT:   %6 = fadd fast double %5, %differeturn
; CHECK-NEXT:   %7 = insertvalue { double, double } undef, double %i11, 0
; CHECK-NEXT:   %8 = insertvalue { double, double } %7, double %6, 1
; CHECK-NEXT:   ret { double, double } %8
; CHECK-NEXT: }

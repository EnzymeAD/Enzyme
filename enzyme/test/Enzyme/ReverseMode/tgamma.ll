; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -gvn -adce -instcombine -instsimplify -early-cse -simplifycfg -correlated-propagation -adce -jump-threading -instsimplify -enzyme-runtime-error -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,gvn,adce,instcombine,instsimplify,early-cse,%simplifycfg,correlated-propagation,adce,jump-threading,instsimplify)" -enzyme-runtime-error -S | FileCheck %s

define internal fastcc double @a5(double %arg) {
bb:
  %i8 = call double @tgamma(double %arg)
  %i22 = fmul double %i8, %arg
  ret double %i22
}

define double @julia_f_1997(double %arg) {
bb:
  %i7 = call fastcc double @a5(double %arg)
  %i8 = fmul double %i7, %i7
  ret double %i8
}

declare double @__enzyme_autodiff(...)

define double @dsquare(double %arg) local_unnamed_addr {
bb:
  %i = tail call double (...) @__enzyme_autodiff(double (double)* nonnull @julia_f_1997, metadata !"enzyme_out", double %arg)
  ret double %i
}

declare double @tgamma(double)

; CHECK: define internal fastcc double @augmented_a5(double %arg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %i8 = call double @tgamma(double %arg)
; CHECK-NEXT:   %i22 = fmul double %i8, %arg
; CHECK-NEXT:   ret double %i22
; CHECK-NEXT: }

; CHECK: define internal fastcc { double } @diffea5(double %arg, double %differeturn)

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -early-cse -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

define double* @f(double** %a0) "enzyme_ReadOnlyOrThrow"="" {
  %a3 = load double*, double** %a0, align 8
  %a5 = call double* @g(double* %a3)
  ret double* %a5
}

define double* @g(double* %0) "enzyme_ReadOnlyOrThrow"="" {
  ret double* %0
}

define double @square(double %x) {
entry:
  %a = alloca double
  store double %x, double* %a
  %ap = alloca double*
  store double* %a, double** %ap
  %z = call double* @f(double** %ap)
  %m = load double, double* %z
  %mul = fmul fast double %m, %m
  ret double %mul
}

declare double @__enzyme_autodiff(...)

define double @dsquare(double %x) {
entry:
  %0 = tail call double (...) @__enzyme_autodiff(double (double)* nonnull @square, double %x)
  ret double %0
}

; CHECK: define internal { { double*, double* }, double*, double* } @augmented_f(double** %a0, double** %"a0'")
; CHECK-NEXT:   %1 = alloca { { double*, double* }, double*, double* }, align 8
; CHECK-NEXT:   %2 = getelementptr inbounds { { double*, double* }, double*, double* }, { { double*, double* }, double*, double* }* %1, i32 0, i32 0
; CHECK-NEXT:   %"a3'ipl" = load double*, double** %"a0'", align 8, !alias.scope !15, !noalias !18
; CHECK-NEXT:   %a3 = load double*, double** %a0, align 8, !alias.scope !18, !noalias !15
; CHECK-NEXT:   %3 = getelementptr inbounds { double*, double* }, { double*, double* }* %2, i32 0, i32 1
; CHECK-NEXT:   store double* %a3, double** %3, align 8
; CHECK-NEXT:   %a5_augmented = call { double*, double* } @augmented_g(double* %a3, double* %"a3'ipl")
; CHECK-NEXT:   %a5 = extractvalue { double*, double* } %a5_augmented, 0
; CHECK-NEXT:   %"a5'ac" = extractvalue { double*, double* } %a5_augmented, 1
; CHECK-NEXT:   %4 = getelementptr inbounds { double*, double* }, { double*, double* }* %2, i32 0, i32 0
; CHECK-NEXT:   store double* %"a5'ac", double** %4, align 8
; CHECK-NEXT:   %5 = getelementptr inbounds { { double*, double* }, double*, double* }, { { double*, double* }, double*, double* }* %1, i32 0, i32 1
; CHECK-NEXT:   store double* %a5, double** %5, align 8
; CHECK-NEXT:   %6 = getelementptr inbounds { { double*, double* }, double*, double* }, { { double*, double* }, double*, double* }* %1, i32 0, i32 2
; CHECK-NEXT:   store double* %"a5'ac", double** %6, align 8
; CHECK-NEXT:   %7 = load { { double*, double* }, double*, double* }, { { double*, double* }, double*, double* }* %1, align 8
; CHECK-NEXT:   ret { { double*, double* }, double*, double* } %7
; CHECK-NEXT: }

; CHECK: define internal void @diffef(double** %a0, double** %"a0'", { double*, double* } %tapeArg)
; CHECK-NEXT: invert:
; CHECK-NEXT:   %a3 = extractvalue { double*, double* } %tapeArg, 1
; CHECK-NEXT:   call void @diffeg(double* %a3, double* {{(undef|poison)}})
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

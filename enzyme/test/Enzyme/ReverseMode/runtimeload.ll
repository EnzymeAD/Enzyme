; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-runtime-activity -enzyme-loose-types -enzyme -mem2reg -instsimplify -adce -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-runtime-activity -enzyme-loose-types -passes="enzyme,function(mem2reg,instsimplify,adce,%simplifycfg,early-cse)" -S | FileCheck %s

declare void @__enzyme_autodiff(...)

define void @f() local_unnamed_addr {
bb:
  call void (...) @__enzyme_autodiff(void (double**, double**)* nonnull @julia___2553_inner.1, double* null, double* null, double* null, double* null)
  ret void
}

define void @julia___2553_inner.1(double**%arg, double** %arg2) {
bb:
  %i36 = call fastcc double* @a1(double** %arg2)
  store double* %i36, double** %arg, align 8
  ret void
}

declare double* @malloc(i64)
define internal fastcc double* @a1(double** %i7) {
bb:
  %i8 = load double*, double** %i7, align 8
  %i17 = call noalias double* @malloc(i64 8)
  %i22 = load double, double* %i8, align 8
  store double %i22, double* %i17, align 8
  ret double* %i17
}

; CHECK: define internal fastcc void @diffea1(double** %i7, double** %"i7'", { double*, double*, double*, double* } %tapeArg)
; CHECK-NEXT: bb:
; CHECK-NEXT:   %"i8'il_phi" = extractvalue { double*, double*, double*, double* } %tapeArg, 2
; CHECK-NEXT:   %i8 = extractvalue { double*, double*, double*, double* } %tapeArg, 3
; CHECK-NEXT:   %"i17'mi" = extractvalue { double*, double*, double*, double* } %tapeArg, 0
; CHECK-NEXT:   %i17 = extractvalue { double*, double*, double*, double* } %tapeArg, 1
; CHECK-NEXT:   %0 = load double, double* %"i17'mi", align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"i17'mi", align 8
; CHECK-NEXT:   %1 = icmp ne double* %i8, %"i8'il_phi"
; CHECK-NEXT:   br i1 %1, label %invertbb_active, label %invertbb_amerge

; CHECK: invertbb_active:                                  ; preds = %bb
; CHECK-NEXT:   %2 = load double, double* %"i8'il_phi", align 8
; CHECK-NEXT:   %3 = fadd fast double %2, %0
; CHECK-NEXT:   store double %3, double* %"i8'il_phi", align 8
; CHECK-NEXT:   br label %invertbb_amerge

; CHECK: invertbb_amerge:                                  ; preds = %invertbb_active, %bb
; CHECK-NEXT:   %4 = bitcast double* %"i17'mi" to i8*
; CHECK-NEXT:   call void @free(i8* nonnull %4)
; CHECK-NEXT:   %5 = bitcast double* %i17 to i8*
; CHECK-NEXT:   call void @free(i8* %5)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

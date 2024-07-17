; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

define void @f(double** nonnull dereferenceable(8) %_M_current, double** nonnull align 8 dereferenceable(8) %__i) {
entry:
  %0 = load double*, double** %__i, align 8
  store double* %0, double** %_M_current, align 8
  ret void
}

define void @test_derivative(double* %x, double* %dx, double* %y, double* %dy) {
entry:
  call void (...) @__enzyme_augmentfwd(void (double**, double**)* nonnull @f, metadata !"enzyme_dup", i8* null, i8* null, metadata !"enzyme_dupnoneed", i8* null, i8* null)
  ret void
}

; Function Attrs: nounwind
declare void @__enzyme_augmentfwd(...)

; CHECK: define internal i8* @augmented_f(double** nonnull dereferenceable(8) %_M_current, double** %"_M_current'", double** align 8 dereferenceable(8) %__i, double** align 8 %"__i'") 
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"'ipl" = load double*, double** %"__i'"
; CHECK-NEXT:   %[[i0:.+]] = load double*, double** %__i
; CHECK-NEXT:   store double* %"'ipl", double** %"_M_current'"
; CHECK-NEXT:   store double* %[[i0]], double** %_M_current
; CHECK-NEXT:   ret i8* null
; CHECK-NEXT: }

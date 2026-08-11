; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -instsimplify -adce -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,instsimplify,adce)" -S | FileCheck %s

declare double @julia_det_5684(double* %0) #14

define double @julia_call_with_kwargs_5681(double* %0) {
top:
  %a4 = call double @julia_det_5684(double* %0)
  ret double %a4
}

declare void @__enzyme_autodiff(...)

define void @dsquare(double* %x, double* %dx) {
entry:
  call void (...) @__enzyme_autodiff(double (double*)* nonnull @julia_call_with_kwargs_5681, metadata !"enzyme_const_return", metadata !"enzyme_dup", double* %x, double* %dx)
  ret void
}

attributes #14 = { nofree "enzyme_LocalReadOnlyOrThrow" }

; CHECK: define internal void @diffejulia_call_with_kwargs_5681(double* %0, double* %"'")
; CHECK-NEXT: top:
; CHECK-NEXT:   %a4 = call double @julia_det_5684(double* %0)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


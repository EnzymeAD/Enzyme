; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare void @__enzyme_fwddiff(...)

define void @f(double* %arg, double* %arg2) {
bb:
  call void (...) @__enzyme_fwddiff(void (double*)* @julia__f_2997, metadata !"enzyme_dup", double* %arg , double* %arg)
  ret void
}

define void @julia__f_2997(double* %arg) {
top:
  %arrayptr60.i.fr.i = freeze double* %arg
  store double 0.0, double* %arrayptr60.i.fr.i
  ret void
}

; CHECK: define internal void @fwddiffejulia__f_2997(double* %arg, double* %"arg'")
; CHECK-NEXT: top:
; CHECK-NEXT:   %arrayptr60.i.fr.i = freeze double* %arg
; CHECK-NEXT:   %[[i0:.+]] = freeze double* %"arg'"
; CHECK-NEXT:   store double 0.000000e+00, double* %[[i0]]
; CHECK-NEXT:   store double 0.000000e+00, double* %arrayptr60.i.fr.i
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

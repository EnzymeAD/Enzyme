; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme" -S | FileCheck %s

declare void @__enzyme_autodiff(...) 

define void @sub(double* writeonly nocapture dereferenceable(8) nonnull noundef %x) {
entry:
  store double 0.0, double* %x
  ret void
}

define void @square(double* %x) {
entry:
  call void @sub(double* %x)
  ret void
}

define void @dsquare(double* %x, double* %dx) {
entry:
  call void (...) @__enzyme_autodiff(void (double*)* @square, metadata !"enzyme_dupnoneed", double* %x, double* %dx)
  ret void
}

; CHECK: define internal void @diffesquare(double* %x, double* %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertentry
; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   call void @diffesub(double* {{(undef|poison)}}, double* %"x'")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(double* nocapture writeonly %x, double* nocapture %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertentry
; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double 0.000000e+00, ptr %"x'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

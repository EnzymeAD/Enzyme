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

define void @dsquare(ptr %x, ptr %dx) {
entry:
  call void (...) @__enzyme_autodiff(void (ptr)* @square, metadata !"enzyme_dupnoneed", ptr %x, ptr %dx)
  ret void
}

; CHECK: define internal void @diffesquare(ptr %x, ptr %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertentry
; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   call void @diffesub(ptr undef, ptr %"x'")
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @diffesub(ptr nocapture writeonly %x, ptr nocapture dereferenceable(8) %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %invertentry
; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double 0.000000e+00, ptr %"x'", align 8
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

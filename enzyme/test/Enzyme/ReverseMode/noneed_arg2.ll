; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare void @__enzyme_autodiff(...) 

define double @sub(double *%x) {
  %ld = load double, double* %x
  %mul = fmul double %ld, %ld
  ret double %mul
}

define double @square(i8 *%p) {
entry:
  %x = bitcast i8* %p to double*
  %cl = call double @sub(double* %x)
  ret double %cl
}

define void @dsquare(double* %x, double* %dx) {
entry:
  call void (...) @__enzyme_autodiff(double (i8*)* @square, metadata !"enzyme_dupnoneed", double* %x, double* %dx)
  ret void
}

; CHECK: define internal void @diffesquare(i8* %p, i8* %"p'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"cl'de" = alloca double
; CHECK-NEXT:   store double 0.000000e+00, double* %"cl'de"
; CHECK-NEXT:   %"x'ipc" = bitcast i8* %"p'" to double*
; CHECK-NEXT:   %x = bitcast i8* %p to double*
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double %differeturn, double* %"cl'de"
; CHECK-NEXT:   %0 = load double, double* %"cl'de"
; CHECK-NEXT:   call void @diffesub(double* %x, double* %"x'ipc", double %0)
; CHECK-NEXT:   store double 0.000000e+00, double* %"cl'de"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


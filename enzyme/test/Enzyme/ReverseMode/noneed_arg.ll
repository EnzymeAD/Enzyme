; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme" -S | FileCheck %s

declare void @__enzyme_autodiff(...) 

define double @sub(double *%x) {
  %ld = load double, double* %x
  %mul = fmul double %ld, %ld
  ret double %mul
}

define double @square(double *%x) {
entry:
  %cl = call double @sub(double* %x)
  ret double %cl
}

define void @dsquare(double* %x, double* %dx) {
entry:
  call void (...) @__enzyme_autodiff(double (double*)* @square, metadata !"enzyme_dupnoneed", double* %x, double* %dx)
  ret void
}

; CHECK: define internal void @diffesquare(double* %x, double* %"x'", double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"cl'de" = alloca double
; CHECK-NEXT:   store double 0.000000e+00, double* %"cl'de"
; CHECK-NEXT:   br label %invertentry

; CHECK: invertentry:                                      ; preds = %entry
; CHECK-NEXT:   store double %differeturn, double* %"cl'de"
; CHECK-NEXT:   %0 = load double, double* %"cl'de"
; CHECK-NEXT:   call void @diffesub(double* %x, double* %"x'", double %0)
; CHECK-NEXT:   store double 0.000000e+00, double* %"cl'de"
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

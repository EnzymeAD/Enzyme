; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_fwddiff(...) 

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

define double @dsquare(double* %x, double* %dx) {
entry:
  %res = call double (...) @__enzyme_fwddiff(double (double*)* @square, metadata !"enzyme_dupnoneed", double* %x, double* %dx)
  ret double %res
}

; CHECK: define internal double @fwddiffesquare(double* %x, double* %"x'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call fast double @fwddiffesub(double* %x, double* %"x'")
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_fwddiff(...) 

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

define double @dsquare(double* %x, double* %dx) {
entry:
  %res = call double (...) @__enzyme_fwddiff(double (i8*)* @square, metadata !"enzyme_dupnoneed", double* %x, double* %dx)
  ret double %res
}

; CHECK: define internal double @fwddiffesquare(i8* %p, i8* %"p'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"x'ipc" = bitcast i8* %"p'" to double*
; CHECK-NEXT:   %x = bitcast i8* %p to double*
; CHECK-NEXT:   %0 = call fast double @fwddiffesub(double* %x, double* %"x'ipc")
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

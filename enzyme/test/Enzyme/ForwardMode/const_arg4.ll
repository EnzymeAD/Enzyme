; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_fwddiff(...) 

define double @dsquare(double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double)* @julia_g_6797_inner.1, metadata !"enzyme_dup", double %x, double 1.0)
  ret double %0
}

define dso_local double @julia_f_kw__kw_6800(double* nocapture %a0, double %a1) {
top:
  %a3 = fmul double %a1, %a1
  store double %a3, double* %a0, align 8
  ret double %a3
}

; Function Attrs: nosync readnone
define double @julia_g_6797_inner.1(double %a0) {
entry:
  %a2 = alloca double, align 8
  %a7 = call double @julia_f_kw__kw_6800(double* %a2, double %a0) writeonly
  ret double %a7
}

; CHECK: define internal double @fwddiffejulia_g_6797_inner.1(double %a0, double %"a0'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a2 = alloca double, align 8
; CHECK-NEXT:   %0 = call fast double @fwddiffejulia_f_kw__kw_6800(double* %a2, double %a0, double %"a0'")
; CHECK-NEXT:   ret double %0
; CHECK-NEXT: }

; CHECK: define internal double @fwddiffejulia_f_kw__kw_6800(double* nocapture %a0, double %a1, double %"a1'")
; CHECK-NEXT: top:
; CHECK-NEXT:   %a3 = fmul double %a1, %a1
; CHECK-NEXT:   %0 = fmul fast double %"a1'", %a1
; CHECK-NEXT:   %1 = fmul fast double %"a1'", %a1
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   store double %a3, double* %a0, align 8
; CHECK-NEXT:   ret double %2
; CHECK-NEXT: }

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
  %ld = load double, double* %a2, align 8
  %fm = fmul double %ld, %a7
  ret double %fm
}

; CHECK: define internal double @fwddiffejulia_g_6797_inner.1(double %a0, double %"a0'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"a2'ipa" = alloca double, align 8
; CHECK-NEXT:   store double 0.000000e+00, double* %"a2'ipa", align 8
; CHECK-NEXT:   %a2 = alloca double, align 8
; CHECK-NEXT:   %0 = call { double, double } @fwddiffejulia_f_kw__kw_6800(double* %a2, double* %"a2'ipa", double %a0, double %"a0'")
; CHECK-NEXT:   %1 = extractvalue { double, double } %0, 0
; CHECK-NEXT:   %2 = extractvalue { double, double } %0, 1
; CHECK-NEXT:   %"ld'ipl" = load double, double* %"a2'ipa"
; CHECK-NEXT:   %ld = load double, double* %a2, align 8
; CHECK-NEXT:   %3 = fmul fast double %"ld'ipl", %1
; CHECK-NEXT:   %4 = fmul fast double %2, %ld
; CHECK-NEXT:   %5 = fadd fast double %3, %4
; CHECK-NEXT:   ret double %5
; CHECK-NEXT: }

; CHECK: define internal { double, double } @fwddiffejulia_f_kw__kw_6800(double* nocapture %a0, double* nocapture %"a0'", double %a1, double %"a1'")
; CHECK-NEXT: top:
; CHECK-NEXT:   %a3 = fmul double %a1, %a1
; CHECK-NEXT:   %0 = fmul fast double %"a1'", %a1
; CHECK-NEXT:   %1 = fmul fast double %"a1'", %a1
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   store double %2, double* %"a0'", align 8
; CHECK-NEXT:   store double %a3, double* %a0, align 8
; CHECK-NEXT:   %3 = insertvalue { double, double } {{(undef|poison)?}}, double %a3, 0
; CHECK-NEXT:   %4 = insertvalue { double, double } %3, double %2, 1
; CHECK-NEXT:   ret { double, double } %4
; CHECK-NEXT: }

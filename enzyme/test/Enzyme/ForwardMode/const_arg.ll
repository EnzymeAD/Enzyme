; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

declare double @__enzyme_fwddiff(...) 

define double @dsquare(double %x) {
entry:
  %0 = tail call double (...) @__enzyme_fwddiff(double (double, i8*)* @julia_g_6797_inner.1, metadata !"enzyme_dup", double %x, double 1.0, metadata !"enzyme_const", i8* null)
  ret double %0
}

define dso_local double @julia_f_kw__kw_6800([1 x double]* nocapture %a0, double %a1) {
top:
  %a3 = fmul double %a1, %a1
  ret double %a3
}

; Function Attrs: nosync readnone
define double @julia_g_6797_inner.1(double %a0, i8* %a1) {
entry:
  %a2 = bitcast i8* %a1 to [1 x double]*
  %a7 = call double @julia_f_kw__kw_6800([1 x double]* %a2, double %a0) readnone
  ret double %a7
}

; CHECK: define internal double @fwddiffejulia_g_6797_inner.1(double %a0, double %"a0'", i8* %a1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[r:.+]] = call fast double @fwddiffejulia_f_kw__kw_6800([1 x double]* undef, double %a0, double %"a0'")
; CHECK-NEXT:   ret double %[[r]]
; CHECK-NEXT: }

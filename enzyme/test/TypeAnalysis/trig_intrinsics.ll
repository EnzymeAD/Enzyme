; RUN: if [ %llvmver -ge 20 ]; then %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=tester -S -o /dev/null | FileCheck %s; fi

; The trigonometric intrinsics take and return floating-point values. An
; untyped asin result used to make a phi merging it with a float untyped too
; ("Cannot deduce type of phi", MITgcm gchem_insolation).

declare double @llvm.tan.f64(double)
declare double @llvm.asin.f64(double)
declare double @llvm.acos.f64(double)
declare double @llvm.atan.f64(double)
declare double @llvm.atan2.f64(double, double)
declare { double, double } @llvm.sincos.f64(double)

define void @tester(double %a, double %b, double %c, double %d, double %e, double %f, double %g, i1 %cond) {
entry:
  %t = call double @llvm.tan.f64(double %a)
  %as = call double @llvm.asin.f64(double %b)
  %ac = call double @llvm.acos.f64(double %c)
  %at = call double @llvm.atan.f64(double %d)
  %at2 = call double @llvm.atan2.f64(double %e, double %f)
  %sc = call { double, double } @llvm.sincos.f64(double %g)
  br i1 %cond, label %left, label %join

left:
  %l = fmul double %a, %b
  br label %join

join:
  %p = phi double [ %as, %entry ], [ %l, %left ]
  ret void
}

; CHECK: tester
; CHECK: i1 %cond:
; CHECK-NEXT: {{^}}entry
; CHECK-NEXT:   %t = call double @llvm.tan.f64(double %a): {[-1]:Float@double}
; CHECK-NEXT:   %as = call double @llvm.asin.f64(double %b): {[-1]:Float@double}
; CHECK-NEXT:   %ac = call double @llvm.acos.f64(double %c): {[-1]:Float@double}
; CHECK-NEXT:   %at = call double @llvm.atan.f64(double %d): {[-1]:Float@double}
; CHECK-NEXT:   %at2 = call double @llvm.atan2.f64(double %e, double %f): {[-1]:Float@double}
; CHECK-NEXT:   %sc = call { double, double } @llvm.sincos.f64(double %g): {[-1]:Float@double}
; CHECK: {{^}}join
; CHECK-NEXT:   %p = phi double [ %as, %entry ], [ %l, %left ]: {[-1]:Float@double}

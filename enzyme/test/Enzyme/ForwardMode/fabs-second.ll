; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -enzyme-preopt=false -S | FileCheck %s; fi

; fabs(-(x*x)) == x*x, so its second derivative at 0 must be 2. At x = 0 the
; argument of fabs is -0.0, so fabs'(-0.0) must be -1, i.e. copysign(1, arg),
; not (arg < 0 ? -1 : 1).

define double @tester(double %x) {
entry:
  %sq = fmul double %x, %x
  %neg = fneg double %sq
  %abs = call double @llvm.fabs.f64(double %neg)
  ret double %abs
}

define double @dtester(double %x) {
entry:
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @tester, double %x, double 1.0)
  ret double %r
}

define double @ddtester(double %x) {
entry:
  %r = call double (ptr, ...) @__enzyme_fwddiff(ptr @dtester, double %x, double 1.0)
  ret double %r
}

declare double @llvm.fabs.f64(double)
declare double @__enzyme_fwddiff(ptr, ...)

; CHECK: define internal double @fwddiffefwddiffetester(double %x, double %"x'", double %"x'1")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %sq = fmul double %x, %x
; CHECK-NEXT:   %0 = fmul fast double %"x'", %"x'1"
; CHECK-NEXT:   %1 = fmul fast double %"x'", %"x'1"
; CHECK-NEXT:   %2 = fadd fast double %0, %1
; CHECK-NEXT:   %neg = fneg double %sq
; CHECK-NEXT:   %3 = fneg fast double %2
; CHECK-NEXT:   %4 = call fast double @llvm.copysign.f64(double 1.000000e+00, double %neg)
; CHECK-NEXT:   %5 = fmul fast double %3, %4
; CHECK-NEXT:   ret double %5
; CHECK-NEXT: }

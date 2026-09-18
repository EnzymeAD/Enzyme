; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; A generated adjoint carries the fast math flags of the primal instruction it
; derives from, and no others: a primal with no flags yields an adjoint with no
; flags, and a fast primal yields a fast adjoint.

define double @strict(double %x, double %y) {
entry:
  %cmp = fcmp ogt double %x, 0.000000e+00
  %mul = fmul double %x, %y
  %sel = select i1 %cmp, double %mul, double 0.000000e+00
  ret double %sel
}

define double @loose(double %x, double %y) {
entry:
  %cmp = fcmp ogt double %x, 0.000000e+00
  %mul = fmul fast double %x, %y
  %sel = select fast i1 %cmp, double %mul, double 0.000000e+00
  ret double %sel
}

define double @test_strict(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @strict, double %x, double %y)
  ret double %0
}

define double @test_loose(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__enzyme_autodiff(double (double, double)* nonnull @loose, double %x, double %y)
  ret double %0
}

declare double @__enzyme_autodiff(double (double, double)*, ...)

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffestrict(double %x, double %y, double %[[differet:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = fcmp ogt double %x, 0.000000e+00
; CHECK-NEXT:   %[[s0:.+]] = fadd double 0.000000e+00, %[[differet]]
; CHECK-NEXT:   %[[s1:.+]] = select i1 %cmp, double %[[s0]], double 0.000000e+00
; CHECK-NEXT:   %[[s2:.+]] = fmul double %[[s1]], %y
; CHECK-NEXT:   %[[s3:.+]] = fadd double 0.000000e+00, %[[s2]]
; CHECK-NEXT:   %[[s4:.+]] = fmul double %[[s1]], %x
; CHECK-NEXT:   %[[s5:.+]] = fadd double 0.000000e+00, %[[s4]]
; CHECK-NEXT:   %[[s6:.+]] = insertvalue { double, double } undef, double %[[s3]], 0
; CHECK-NEXT:   %[[s7:.+]] = insertvalue { double, double } %[[s6]], double %[[s5]], 1
; CHECK-NEXT:   ret { double, double } %[[s7]]
; CHECK-NEXT: }

; CHECK: define internal {{(dso_local )?}}{ double, double } @diffeloose(double %x, double %y, double %[[looseret:.+]])
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = fcmp ogt double %x, 0.000000e+00
; CHECK-NEXT:   %[[l0:.+]] = select fast i1 %cmp, double %[[looseret]], double 0.000000e+00
; CHECK-NEXT:   %[[l1:.+]] = fmul fast double %[[l0]], %y
; CHECK-NEXT:   %[[l2:.+]] = fmul fast double %[[l0]], %x
; CHECK-NEXT:   %[[l3:.+]] = insertvalue { double, double } undef, double %[[l1]], 0
; CHECK-NEXT:   %[[l4:.+]] = insertvalue { double, double } %[[l3]], double %[[l2]], 1
; CHECK-NEXT:   ret { double, double } %[[l4]]
; CHECK-NEXT: }

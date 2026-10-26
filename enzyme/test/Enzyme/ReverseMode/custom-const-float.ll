; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; @scale has a custom augmented forward pass and a custom reverse pass that
; treat %c as active. @caller passes a constant %c, so Enzyme massages the
; custom reverse pass to treat %c as active (OUT_DIFF). The massaged custom
; reverse pass must not return the adjoint of %c.

define internal { {}, double } @augment_scale(double %x, double %c) {
entry:
  %m = fmul double %x, %c
  %r = insertvalue { {}, double } undef, double %m, 1
  ret { {}, double } %r
}

define internal { double, double } @gradient_scale(double %x, double %c, double %differet, {} %tape) {
entry:
  %dx = fmul double %differet, %c
  %dc = fmul double %differet, %x
  %r0 = insertvalue { double, double } undef, double %dx, 0
  %r1 = insertvalue { double, double } %r0, double %dc, 1
  ret { double, double } %r1
}

declare !enzyme_augment !{ptr @augment_scale} !enzyme_gradient !{ptr @gradient_scale} double @scale(double %x, double %c)

define double @caller(double %x, double %c) {
entry:
  %r = call double @scale(double %x, double %c)
  %sq = fmul double %r, %r
  ret double %sq
}

define double @dcaller(double %x, double %c) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @caller, double %x, metadata !"enzyme_const", double %c)
  ret double %r
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffecaller(double %x, double %c, double %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r_augmented = call { {}, double } @fixaugment_scale(double %x, double %c)
; CHECK-NEXT:   %r = extractvalue { {}, double } %r_augmented, 1
; CHECK-NEXT:   %[[i0:.+]] = fmul fast double %differeturn, %r
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %differeturn, %r
; CHECK-NEXT:   %[[i2:.+]] = fadd fast double %[[i0]], %[[i1]]
; CHECK-NEXT:   %[[i3:.+]] = call { double } @fixgradient_scale(double %x, double %c, double %[[i2]], {} undef)
; CHECK:        ret { double }
; CHECK-NEXT: }

; CHECK: define internal { double } @fixgradient_scale(double %arg0, double %arg1, double %postarg0, {} %postarg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call { double, double } @gradient_scale(double %arg0, double %arg1, double %postarg0, {} %postarg1)
; CHECK-NEXT:   %[[i1:.+]] = extractvalue { double, double } %[[i0]], 0
; CHECK-NEXT:   %[[i2:.+]] = insertvalue { double } undef, double %[[i1]], 0
; CHECK-NEXT:   ret { double } %[[i2]]
; CHECK-NEXT: }

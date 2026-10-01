; RUN: if [ %llvmver -ge 15 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s; fi

; @scale has a custom augmented forward pass and a custom reverse pass that
; treat %c as active. @caller passes a constant %c, so Enzyme massages both
; custom passes to treat %c as active (OUT_DIFF). @top uses the result of
; @caller, so Enzyme differentiates @caller in split mode. The reverse pass of
; @caller must keep %c constant, and the massaged custom reverse pass must not
; return the adjoint of %c.

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
  ret double %r
}

define void @top(double %x, double %c, ptr %out) {
entry:
  %r = call double @caller(double %x, double %c)
  %sq = fmul double %r, %r
  store double %sq, ptr %out
  ret void
}

define double @dtop(double %x, double %c, ptr %out, ptr %dout) {
entry:
  %r = call double (...) @__enzyme_autodiff(ptr @top, double %x, metadata !"enzyme_const", double %c, ptr %out, ptr %dout)
  ret double %r
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffetop(double %x, double %c, ptr nocapture writeonly %out, ptr nocapture %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r_augmented = call { {}, double } @augmented_caller(double %x, double %c)
; CHECK:        %{{.+}} = call { double } @diffecaller(double %x, double %c, double %{{.+}}, {} undef)

; CHECK: define internal { {}, double } @fixaugment_scale(double %arg0, double %arg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call { {}, double } @augment_scale(double %arg0, double %arg1)
; CHECK-NEXT:   ret { {}, double } %[[i0]]
; CHECK-NEXT: }

; CHECK: define internal { {}, double } @augmented_caller(double %x, double %c)
; CHECK:   %r_augmented = call { {}, double } @fixaugment_scale(double %x, double %c)

; CHECK: define internal { double } @diffecaller(double %x, double %c, double %differeturn, {} %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call { double } @fixgradient_scale(double %x, double %c, double %differeturn, {} undef)
; CHECK-NEXT:   %[[i1:.+]] = extractvalue { double } %[[i0]], 0
; CHECK-NEXT:   %[[i2:.+]] = insertvalue { double } undef, double %[[i1]], 0
; CHECK-NEXT:   ret { double } %[[i2]]
; CHECK-NEXT: }

; CHECK: define internal { double } @fixgradient_scale(double %arg0, double %arg1, double %postarg0, {} %postarg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call { double, double } @gradient_scale(double %arg0, double %arg1, double %postarg0, {} %postarg1)
; CHECK-NEXT:   %[[i1:.+]] = extractvalue { double, double } %[[i0]], 0
; CHECK-NEXT:   %[[i2:.+]] = insertvalue { double } undef, double %[[i1]], 0
; CHECK-NEXT:   ret { double } %[[i2]]
; CHECK-NEXT: }

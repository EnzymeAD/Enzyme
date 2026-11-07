; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -instsimplify -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,instsimplify,%simplifycfg)" -S | FileCheck %s

; @scale has a custom augmented forward pass and a custom reverse pass that
; take a shadow for %p. @caller passes a constant %p, so Enzyme massages both
; custom passes to take a constant %p. @top uses the result of @caller, so
; Enzyme differentiates @caller in split mode (augmented forward pass +
; reverse pass). The reverse pass of @caller must call the massaged custom
; reverse pass with a constant %p.

define internal { {}, double } @augment_scale(double %x, double* %p, double* %dp) {
entry:
  %s = load double, double* %p
  %m = fmul double %x, %s
  %r = insertvalue { {}, double } undef, double %m, 1
  ret { {}, double } %r
}

define internal { double } @gradient_scale(double %x, double* %p, double* %dp, double %differet, {} %tape) {
entry:
  %s = load double, double* %p
  %dx = fmul double %differet, %s
  %r = insertvalue { double } undef, double %dx, 0
  ret { double } %r
}

declare !enzyme_augment !{{ {}, double } (double, double*, double*)* @augment_scale} !enzyme_gradient !{{ double } (double, double*, double*, double, {})* @gradient_scale} double @scale(double %x, double* %p)

define double @caller(double %x, double* %p) {
entry:
  %r = call double @scale(double %x, double* %p)
  ret double %r
}

define void @top(double %x, double* %p, double* %out) {
entry:
  %r = call double @caller(double %x, double* %p)
  %sq = fmul double %r, %r
  store double %sq, double* %out
  ret void
}

define double @dtop(double %x, double* %p, double* %out, double* %dout) {
entry:
  %r = call double (...) @__enzyme_autodiff(void (double, double*, double*)* @top, double %x, metadata !"enzyme_const", double* %p, double* %out, double* %dout)
  ret double %r
}

declare double @__enzyme_autodiff(...)

; CHECK: define internal { double } @diffetop(double %x, double* %p, double* nocapture writeonly %out, double* nocapture %"out'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %r_augmented = call { {}, double } @augmented_caller(double %x, double* %p)
; CHECK-NEXT:   %r = extractvalue { {}, double } %r_augmented, 1
; CHECK-NEXT:   %sq = fmul double %r, %r
; CHECK-NEXT:   store double %sq, double* %out
; CHECK-NEXT:   %[[i0:.+]] = load double, double* %"out'"
; CHECK-NEXT:   store double 0.000000e+00, double* %"out'"
; CHECK-NEXT:   %[[i1:.+]] = fmul fast double %[[i0]], %r
; CHECK-NEXT:   %[[i2:.+]] = fmul fast double %[[i0]], %r
; CHECK-NEXT:   %[[i3:.+]] = fadd fast double %[[i1]], %[[i2]]
; CHECK-NEXT:   %[[i4:.+]] = call { double } @diffecaller(double %x, double* %p, double %[[i3]], {} undef)
; CHECK-NEXT:   ret { double } %[[i4]]
; CHECK-NEXT: }

; CHECK: define internal { {}, double } @fixaugment_scale(double %arg0, double* %arg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call { {}, double } @augment_scale(double %arg0, double* %arg1, double* %arg1)
; CHECK-NEXT:   ret { {}, double } %[[i0]]
; CHECK-NEXT: }

; CHECK: define internal { {}, double } @augmented_caller(double %x, double* %p)
; CHECK:   %r_augmented = call { {}, double } @fixaugment_scale(double %x, double* %p)

; CHECK: define internal { double } @diffecaller(double %x, double* %p, double %differeturn, {} %tapeArg)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call { double } @fixgradient_scale(double %x, double* %p, double %differeturn, {} undef)
; CHECK-NEXT:   ret { double } %[[i0]]
; CHECK-NEXT: }

; CHECK: define internal { double } @fixgradient_scale(double %arg0, double* %arg1, double %postarg0, {} %postarg1)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[i0:.+]] = call { double } @gradient_scale(double %arg0, double* %arg1, double* %arg1, double %postarg0, {} %postarg1)
; CHECK-NEXT:   ret { double } %[[i0]]
; CHECK-NEXT: }

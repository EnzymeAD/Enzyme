; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -simplifycfg -early-cse -instsimplify -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,early-cse,instsimplify)" -S | FileCheck %s

declare { double, double } @__muldc3(double, double, double, double)
declare { double, double, double, double } @__enzyme_autodiff(...)


define { double, double } @square(double %xre, double %xim, double %yre, double %yim) {
entry:
  ; CHECK-LABEL: define internal { double, double, double, double } @diffesquare(double %xre, double %xim, double %yre, double %yim, { double, double } %differeturn)
  ; CHECK-NEXT: entry:
  ; CHECK-NEXT:   %[[DRE:.+]] = extractvalue { double, double } %differeturn, 0
  ; CHECK-NEXT:   %[[DXR0:.+]] = fmul fast double %[[DRE]], %yre
  ; CHECK-NEXT:   %[[DIM:.+]] = extractvalue { double, double } %differeturn, 1
  ; CHECK-NEXT:   %[[DXR1:.+]] = fmul fast double %[[DIM]], %yim
  ; CHECK-NEXT:   %[[DXR:.+]] = fadd fast double %[[DXR0]], %[[DXR1]]
  ; CHECK-NEXT:   %[[DXI0_POS:.+]] = fmul fast double %[[DRE]], %yim
  ; CHECK-NEXT:   %[[DXI0:.+]] = fneg fast double %[[DXI0_POS]]
  ; CHECK-NEXT:   %[[DXI1:.+]] = fmul fast double %[[DIM]], %yre
  ; CHECK-NEXT:   %[[DXI:.+]] = fadd fast double %[[DXI0]], %[[DXI1]]
  ; CHECK-NEXT:   %[[DYR0:.+]] = fmul fast double %[[DRE]], %xre
  ; CHECK-NEXT:   %[[DYR1:.+]] = fmul fast double %[[DIM]], %xim
  ; CHECK-NEXT:   %[[DYR:.+]] = fadd fast double %[[DYR0]], %[[DYR1]]
  ; CHECK-NEXT:   %[[DYI0_POS:.+]] = fmul fast double %[[DRE]], %xim
  ; CHECK-NEXT:   %[[DYI0:.+]] = fneg fast double %[[DYI0_POS]]
  ; CHECK-NEXT:   %[[DYI1:.+]] = fmul fast double %[[DIM]], %xre
  ; CHECK-NEXT:   %[[DYI:.+]] = fadd fast double %[[DYI0]], %[[DYI1]]
  ; CHECK-NEXT:   %[[RES0:.+]] = insertvalue { double, double, double, double } {{.+}}, double %[[DXR]], 0
  ; CHECK-NEXT:   %[[RES1:.+]] = insertvalue { double, double, double, double } %[[RES0]], double %[[DXI]], 1
  ; CHECK-NEXT:   %[[RES2:.+]] = insertvalue { double, double, double, double } %[[RES1]], double %[[DYR]], 2
  ; CHECK-NEXT:   %[[RES3:.+]] = insertvalue { double, double, double, double } %[[RES2]], double %[[DYI]], 3
  ; CHECK-NEXT:   ret { double, double, double, double } %[[RES3]]
  ; CHECK-NEXT: }
  %call = call { double, double } @__muldc3(double %xre, double %xim, double %yre, double %yim)
  ret { double, double } %call
}

define { double, double, double, double } @dsquare(double %x.coerce0, double %x.coerce1) {
entry:
  %call = call { double, double, double, double } (...) @__enzyme_autodiff({ double, double } (double, double, double, double)* @square, double %x.coerce0, double %x.coerce1, double 1.000000e+00, double 0.000000e+00)
  ret { double, double, double, double } %call
}

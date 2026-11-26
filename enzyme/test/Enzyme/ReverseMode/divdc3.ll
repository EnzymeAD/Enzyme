; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false  -enzyme -mem2reg -simplifycfg -early-cse -instsimplify  -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,%simplifycfg,early-cse,instsimplify)" -S | FileCheck %s

define { double, double } @test(double %xre, double %xim, double %yre, double %yim) {
entry:
  ; CHECK-LABEL: define internal { double, double, double, double } @diffetest(double %xre, double %xim, double %yre, double %yim, { double, double } %differeturn)
  ; CHECK-NEXT: entry:
  ; CHECK-NEXT:   %[[DRE:.+]] = extractvalue { double, double } %differeturn, 0
  ; CHECK-NEXT:   %[[YR2:.+]] = fmul fast double %yre, %yre
  ; CHECK-NEXT:   %[[YI2:.+]] = fmul fast double %yim, %yim
  ; CHECK-NEXT:   %[[YNORM2:.+]] = fadd fast double %[[YR2]], %[[YI2]]
  ; CHECK-NEXT:   %[[YR_OVER_D:.+]] = fdiv fast double %yre, %[[YNORM2]]
  ; CHECK-NEXT:   %[[DXR0:.+]] = fmul fast double %[[DRE]], %[[YR_OVER_D]]
  ; CHECK-NEXT:   %[[DIM:.+]] = extractvalue { double, double } %differeturn, 1
  ; CHECK-NEXT:   %[[NYIM:.+]] = fneg fast double %yim
  ; CHECK-NEXT:   %[[NYI_OVER_D:.+]] = fdiv fast double %[[NYIM]], %[[YNORM2]]
  ; CHECK-NEXT:   %[[DXR1:.+]] = fmul fast double %[[DIM]], %[[NYI_OVER_D]]
  ; CHECK-NEXT:   %[[DXR:.+]] = fadd fast double %[[DXR0]], %[[DXR1]]
  ; CHECK-NEXT:   %[[YI_OVER_D:.+]] = fdiv fast double %yim, %[[YNORM2]]
  ; CHECK-NEXT:   %[[DXI0:.+]] = fmul fast double %[[DRE]], %[[YI_OVER_D]]
  ; CHECK-NEXT:   %[[DXI1:.+]] = fmul fast double %[[DIM]], %[[YR_OVER_D]]
  ; CHECK-NEXT:   %[[DXI:.+]] = fadd fast double %[[DXI0]], %[[DXI1]]
  ; CHECK-NEXT:   %[[NXRE:.+]] = fneg fast double %xre
  ; CHECK-NEXT:   %[[NXIM:.+]] = fneg fast double %xim
  ; CHECK-NEXT:   %[[YSQ_RE:.+]] = fsub fast double %[[YR2]], %[[YI2]]
  ; CHECK-NEXT:   %[[YR_YI:.+]] = fmul fast double %yre, %yim
  ; CHECK-NEXT:   %[[YSQ_IM:.+]] = fadd fast double %[[YR_YI]], %[[YR_YI]]
  ; CHECK-NEXT:   %[[W_NUM_RE0:.+]] = fmul fast double %[[NXRE]], %[[YSQ_RE]]
  ; CHECK-NEXT:   %[[W_NUM_RE1:.+]] = fmul fast double %[[NXIM]], %[[YSQ_IM]]
  ; CHECK-NEXT:   %[[W_NUM_RE:.+]] = fadd fast double %[[W_NUM_RE0]], %[[W_NUM_RE1]]
  ; CHECK-NEXT:   %[[YSQ_RE2:.+]] = fmul fast double %[[YSQ_RE]], %[[YSQ_RE]]
  ; CHECK-NEXT:   %[[YSQ_IM2:.+]] = fmul fast double %[[YSQ_IM]], %[[YSQ_IM]]
  ; CHECK-NEXT:   %[[YNORM4:.+]] = fadd fast double %[[YSQ_RE2]], %[[YSQ_IM2]]
  ; CHECK-NEXT:   %[[W_RE:.+]] = fdiv fast double %[[W_NUM_RE]], %[[YNORM4]]
  ; CHECK-NEXT:   %[[W_NUM_IM0:.+]] = fmul fast double %[[YSQ_RE]], %[[NXIM]]
  ; CHECK-NEXT:   %[[W_NUM_IM1:.+]] = fmul fast double %[[NXRE]], %[[YSQ_IM]]
  ; CHECK-NEXT:   %[[W_NUM_IM:.+]] = fsub fast double %[[W_NUM_IM0]], %[[W_NUM_IM1]]
  ; CHECK-NEXT:   %[[W_IM:.+]] = fdiv fast double %[[W_NUM_IM]], %[[YNORM4]]
  ; CHECK-NEXT:   %[[DYR0:.+]] = fmul fast double %[[DRE]], %[[W_RE]]
  ; CHECK-NEXT:   %[[DYR1:.+]] = fmul fast double %[[DIM]], %[[W_IM]]
  ; CHECK-NEXT:   %[[DYR:.+]] = fadd fast double %[[DYR0]], %[[DYR1]]
  ; CHECK-NEXT:   %[[NW_IM:.+]] = fneg fast double %[[W_IM]]
  ; CHECK-NEXT:   %[[DYI0:.+]] = fmul fast double %[[DRE]], %[[NW_IM]]
  ; CHECK-NEXT:   %[[DYI1:.+]] = fmul fast double %[[DIM]], %[[W_RE]]
  ; CHECK-NEXT:   %[[DYI:.+]] = fadd fast double %[[DYI0]], %[[DYI1]]
  ; CHECK-NEXT:   %[[RES0:.+]] = insertvalue { double, double, double, double } {{.+}}, double %[[DXR]], 0
  ; CHECK-NEXT:   %[[RES1:.+]] = insertvalue { double, double, double, double } %[[RES0]], double %[[DXI]], 1
  ; CHECK-NEXT:   %[[RES2:.+]] = insertvalue { double, double, double, double } %[[RES1]], double %[[DYR]], 2
  ; CHECK-NEXT:   %[[RES3:.+]] = insertvalue { double, double, double, double } %[[RES2]], double %[[DYI]], 3
  ; CHECK-NEXT:   ret { double, double, double, double } %[[RES3]]
  ; CHECK-NEXT: }
  %call = call { double, double } @__divdc3(double %xre, double %xim, double %yre, double %yim)
  ret { double, double } %call
}

declare { double, double } @__divdc3(double, double, double, double)

define { double, double, double, double} @dtest(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double, double, double } (i8*, ...) @__enzyme_autodiff(i8* bitcast ({ double, double } (double, double, double, double)* @test to i8*), double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1)
  ret { double, double, double, double} %call
}

declare { double, double, double, double } @__enzyme_autodiff(i8*, ...)


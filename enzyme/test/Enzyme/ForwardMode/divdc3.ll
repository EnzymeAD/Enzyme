; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -early-cse -instsimplify -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(early-cse,instsimplify)" -enzyme-preopt=false -S | FileCheck %s

declare dso_local { double, double } @__divdc3(double, double, double, double)
declare { double, double } @__enzyme_fwddiff(i8*, ...)

define { double, double } @tester(double %xre, double %xim, double %yre, double %yim) {
entry:
  ; CHECK-LABEL: define internal { double, double } @fwddiffetester(double %xre, double %"xre'", double %xim, double %"xim'", double %yre, double %"yre'", double %yim, double %"yim'")
  ; CHECK-NEXT: entry:
  ; CHECK-NEXT:   %[[DXR_YR:.+]] = fmul fast double %"xre'", %yre
  ; CHECK-NEXT:   %[[DXI_YI:.+]] = fmul fast double %"xim'", %yim
  ; CHECK-NEXT:   %[[NUM_RE:.+]] = fadd fast double %[[DXR_YR]], %[[DXI_YI]]
  ; CHECK-NEXT:   %[[YR2:.+]] = fmul fast double %yre, %yre
  ; CHECK-NEXT:   %[[YI2:.+]] = fmul fast double %yim, %yim
  ; CHECK-NEXT:   %[[YNORM2:.+]] = fadd fast double %[[YR2]], %[[YI2]]
  ; CHECK-NEXT:   %[[DX_DIV_Y_RE:.+]] = fdiv fast double %[[NUM_RE]], %[[YNORM2]]
  ; CHECK-NEXT:   %[[YR_DXI:.+]] = fmul fast double %yre, %"xim'"
  ; CHECK-NEXT:   %[[DXR_YI:.+]] = fmul fast double %"xre'", %yim
  ; CHECK-NEXT:   %[[NUM_IM:.+]] = fsub fast double %[[YR_DXI]], %[[DXR_YI]]
  ; CHECK-NEXT:   %[[DX_DIV_Y_IM:.+]] = fdiv fast double %[[NUM_IM]], %[[YNORM2]]
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
  ; CHECK-NEXT:   %[[DY_W_RE0:.+]] = fmul fast double %"yre'", %[[W_RE]]
  ; CHECK-NEXT:   %[[DY_W_RE1:.+]] = fmul fast double %"yim'", %[[W_IM]]
  ; CHECK-NEXT:   %[[DY_TERM_RE:.+]] = fsub fast double %[[DY_W_RE0]], %[[DY_W_RE1]]
  ; CHECK-NEXT:   %[[DY_W_IM0:.+]] = fmul fast double %"yre'", %[[W_IM]]
  ; CHECK-NEXT:   %[[DY_W_IM1:.+]] = fmul fast double %[[W_RE]], %"yim'"
  ; CHECK-NEXT:   %[[DY_TERM_IM:.+]] = fadd fast double %[[DY_W_IM0]], %[[DY_W_IM1]]
  ; CHECK-NEXT:   %[[OUT_RE:.+]] = fadd fast double %[[DX_DIV_Y_RE]], %[[DY_TERM_RE]]
  ; CHECK-NEXT:   %[[OUT_IM:.+]] = fadd fast double %[[DX_DIV_Y_IM]], %[[DY_TERM_IM]]
  ; CHECK-NEXT:   %[[RES0:.+]] = insertvalue { double, double } {{.+}}, double %[[OUT_RE]], 0
  ; CHECK-NEXT:   %[[RES1:.+]] = insertvalue { double, double } %[[RES0]], double %[[OUT_IM]], 1
  ; CHECK-NEXT:   ret { double, double } %[[RES1]]
  ; CHECK-NEXT: }
  %call = call { double, double } @__divdc3(double %xre, double %xim, double %yre, double %yim)
  ret { double, double } %call
}

define dso_local { double, double } @test_derivative(double %x.coerce0, double %x.coerce1, double %y.coerce0, double %y.coerce1) {
entry:
  %call = call { double, double } (i8*, ...) @__enzyme_fwddiff(i8* bitcast ({ double, double } (double, double, double, double)* @tester to i8*), double %x.coerce0, double %x.coerce1, double 1.000000e+00, double 0.000000e+00, double %y.coerce0, double %y.coerce1, double 1.000000e+00, double 0.000000e+00)
  ret { double, double } %call
}

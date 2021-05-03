; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -early-cse -simplifycfg -S | FileCheck %s

; Function Attrs: noinline
define dso_local void @compute_sumabs(float* %a, float* %b, float* %ret) #1 {
entry:
  %al = load float, float* %a
  %call = call float @myabs(float %al)
  %bl = load float, float* %b
  %call1 = call float @myabs(float %bl)
  %add = fadd float %call, %call1
  store float %add, float* %ret
  ret void
}

; Function Attrs: noinline
define linkonce_odr dso_local float @myabs(float %x) #1 {
entry:
  %0 = call float @llvm.fabs.f32(float %x)
  ret float %0
}

define void @dsumabs(float* %a, float* %da, float* %b, float* %db, float* %ret, float* %dret) {
entry:
  %0 = call double (...) @__enzyme_autodiff.f64(void (float*, float*, float*)* @compute_sumabs, float* %a, float* %da, float* %b, float* %db, float* %ret, float* %dret)
  ret void
}

declare double @__enzyme_autodiff.f64(...)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.fabs.f32(float) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { noinline }

; CHECK: define internal {{(dso_local )?}}void @diffecompute_sumabs(float* %a, float* %"a'", float* %b, float* %"b'", float* %ret, float* %"ret'")
; CHECK-NEXT: entry:
; CHECK-NEXT:  %al = load float, float* %a
; CHECK-NEXT:  %[[myabs1:.+]] = call fast float @augmented_myabs(float %al)
; CHECK-NEXT:  %bl = load float, float* %b
; CHECK-NEXT:  %[[myabsret:.+]] = call fast float @augmented_myabs(float %bl)
; CHECK-NEXT:  %add = fadd float %[[myabs1]], %[[myabsret]]
; CHECK-NEXT:  store float %add, float* %ret
; CHECK-NEXT:  %[[ldret:.+]] = load float, float* %"ret'"
; CHECK-NEXT:  store float 0.000000e+00, float* %"ret'"
; CHECK-NEXT:  %[[dabsb:.+]] = call { float } @diffemyabs(float %bl, float %[[ldret]])
; CHECK-NEXT:  %[[extb:.+]] = extractvalue { float } %[[dabsb]], 0
; CHECK-NEXT:  %[[preb:.+]] = load float, float* %"b'"
; CHECK-NEXT:  %[[totalb:.+]] = fadd fast float %[[preb]], %[[extb]]
; CHECK-NEXT:  store float %[[totalb]], float* %"b'"
; CHECK-NEXT:  %[[dabsa:.+]] = call { float } @diffemyabs(float %al, float %[[ldret]])
; CHECK-NEXT:  %[[exta:.+]] = extractvalue { float } %[[dabsa]], 0
; CHECK-NEXT:  %[[prea:.+]] = load float, float* %"a'"
; CHECK-NEXT:  %[[totala:.+]] = fadd fast float %[[prea]], %[[exta]]
; CHECK-NEXT:  store float %[[totala]], float* %"a'"
; CHECK-NEXT:  ret void
; CHECK-NEXT: }

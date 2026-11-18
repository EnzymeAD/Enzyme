; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme" -enzyme-preopt=false -S | FileCheck %s

@enzyme_const = external global i32

define dso_local float @man_max(float* %a, float* %b) #0 {
entry:
  %0 = load float, float* %a, align 4
  %1 = load float, float* %b, align 4
  %cmp = fcmp ogt float %0, %1
  %a.b = select i1 %cmp, float* %a, float* %b
  %retval.0 = load float, float* %a.b, align 4
  ret float %retval.0
}

define void @dman_max(float* %a, float* %da, float* %b) {
entry:
  %ec = load i32, i32* @enzyme_const, align 4
  call float (...) @__enzyme_fwddiff.f64(float (float*, float*)* @man_max, float* %a, float* %da, i32 %ec, float* %b)
  ret void
}

declare float @__enzyme_fwddiff.f64(...)

attributes #0 = { noinline }

; CHECK: define internal float @fwddiffeman_max(float* %a, float* %"a'", float* %b)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load float, float* %a, align 4
; CHECK-NEXT:   %1 = load float, float* %b, align 4
; CHECK-NEXT:   %cmp = fcmp ogt float %0, %1
; CHECK-NEXT:   %"a.b'ipse" = select i1 %cmp, float* %"a'", float* %b
; CHECK-NEXT:   %[[ld:.+]] = load float, float* %"a.b'ipse", align 4
; CHECK-NEXT:   %[[res:.+]] = select{{( fast)?}} i1 %cmp, float %[[ld]], float 0.000000e+00
; CHECK-NEXT:   ret float %[[res]]
; CHECK-NEXT: }

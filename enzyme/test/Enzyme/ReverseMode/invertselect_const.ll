; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -enzyme -mem2reg -early-cse -simplifycfg -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,%simplifycfg)" -S | FileCheck %s

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
  call void (...) @__enzyme_autodiff.f64(float (float*, float*)* @man_max, float* %a, float* %da, i32 %ec, float* %b)
  ret void
}

declare void @__enzyme_autodiff.f64(...)

attributes #0 = { noinline }

; CHECK: define internal {{(dso_local )?}}void @diffeman_max(float* %a, float* %"a'", float* %b, float %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = load float, float* %a, align 4
; CHECK-NEXT:   %1 = load float, float* %b, align 4
; CHECK-NEXT:   %cmp = fcmp ogt float %0, %1
; CHECK-NEXT:   br i1 %cmp, label %invertentry_active, label %invertentry_amerge

; CHECK: invertentry_active:
; CHECK-NEXT:   %2 = load float, float* %"a'", align 4
; CHECK-NEXT:   %3 = fadd fast float %2, %differeturn
; CHECK-NEXT:   store float %3, float* %"a'", align 4
; CHECK-NEXT:   br label %invertentry_amerge

; CHECK: invertentry_amerge:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

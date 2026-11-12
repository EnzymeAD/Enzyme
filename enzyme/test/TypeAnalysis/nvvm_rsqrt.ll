; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -print-type-analysis -type-analysis-func=tester -S -o /dev/null | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="print-type-analysis" -type-analysis-func=tester -S -o /dev/null | FileCheck %s

declare double @llvm.nvvm.rsqrt.approx.d(double)
declare float @llvm.nvvm.rsqrt.approx.f(float)
declare float @llvm.nvvm.rsqrt.approx.ftz.f(float)

define void @tester(double %a, float %b, float %c) {
entry:
  %r1 = call double @llvm.nvvm.rsqrt.approx.d(double %a)
  %r2 = call float @llvm.nvvm.rsqrt.approx.f(float %b)
  %r3 = call float @llvm.nvvm.rsqrt.approx.ftz.f(float %c)
  ret void
}

; CHECK: tester
; CHECK-NEXT: double %a: {[-1]:Float@double}
; CHECK-NEXT: float %b: {[-1]:Float@float}
; CHECK-NEXT: float %c: {[-1]:Float@float}
; CHECK-NEXT: entry
; CHECK-NEXT:   %r1 = call double @llvm.nvvm.rsqrt.approx.d(double %a): {[-1]:Float@double}
; CHECK-NEXT:   %r2 = call float @llvm.nvvm.rsqrt.approx.f(float %b): {[-1]:Float@float}
; CHECK-NEXT:   %r3 = call float @llvm.nvvm.rsqrt.approx.ftz.f(float %c): {[-1]:Float@float}

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -fp-opt -enzyme-print-fpopt -enzyme-print-herbie -enzyme-preopt=false -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="fp-opt" -enzyme-preopt=false -S | FileCheck %s

; Function Attrs: noinline nounwind readnone uwtable
define float @tester(float %x) {
entry:
  %0 = call fast float @llvm.cos.f32(float %x)
  %1 = fmul fast float %0, %0
  %2 = fsub fast float 1.000000e+00, %1
  ret float %2
}

; Function Attrs: nounwind readnone speculatable
declare float @llvm.cos.f32(float)

; Function Attrs: nounwind readnone speculatable
declare float @llvm.sin.f32(float)

; CHECK: define float @tester(float %x)
; CHECK: entry:
; CHECK-NEXT:   %[[i0:.+]] = call fast float @llvm.sin.f32(float %x)
; CHECK-NEXT:   %[[i1:.+]] = call fast float @llvm.pow.f32(float %[[i0]], float 2.000000e+00)
; CHECK-NEXT:   ret float %[[i1]]

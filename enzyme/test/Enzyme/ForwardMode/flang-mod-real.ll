; RUN: if [ %llvmver -ge 16 ]; then %opt < %s %OPnewLoadEnzyme -enzyme-preopt=false -passes="enzyme,function(mem2reg,early-cse,instsimplify,%simplifycfg,adce)" -S | FileCheck %s; fi

; Forward mode of LLVM flang's runtime MOD of REAL(4):
; d(MOD(a, p)) = da - trunc(a/p) dp.

@file = private constant [6 x i8] c"m.f90\00"

define float @mod(float %a, float %p) {
entry:
  %r = call float @_FortranAModReal4(float %a, float %p, ptr @file, i32 3)
  ret float %r
}

define float @test_derivative(float %a, float %p) {
entry:
  %0 = call float (...) @__enzyme_fwddiff(ptr @mod, float %a, float 1.0, float %p, float 1.0)
  ret float %0
}

declare float @_FortranAModReal4(float, float, ptr, i32)
declare float @__enzyme_fwddiff(...)

; CHECK: define internal float @fwddiffemod(float %a, float %"a'", float %p, float %"p'")
; CHECK-NEXT: entry:
; CHECK-NEXT:   %[[q:.+]] = fdiv fast float %a, %p
; CHECK-NEXT:   %[[t:.+]] = call fast float @llvm.trunc.f32(float %[[q]])
; CHECK-NEXT:   %[[n:.+]] = fneg fast float %[[t]]
; CHECK-NEXT:   %[[dp:.+]] = fmul fast float %"p'", %[[n]]
; CHECK-NEXT:   %[[d:.+]] = fadd fast float %"a'", %[[dp]]
; CHECK-NEXT:   ret float %[[d]]

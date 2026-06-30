; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s
; Regression test: partial-window accumulation into a fixed vector (<2 x float>).
; Previously asserted: "unhandled accumulate with partial sizes".

source_filename = "partial_vec_window"
target triple = "x86_64-pc-linux-gnu"

%ret2v = type { <2 x float>, <2 x float> }

define %ret2v @make(float %x) {
entry:
  %v0 = insertelement <2 x float> zeroinitializer, float %x, i32 0
  %r0 = insertvalue %ret2v undef, <2 x float> %v0, 0
  %r1 = insertvalue %ret2v %r0, <2 x float> zeroinitializer, 1
  ret %ret2v %r1
}

define float @tester(float %x) {
entry:
  %call = call %ret2v @make(float %x)
  %vec  = extractvalue %ret2v %call, 0

  ; Force "partial" use: only the first 4 bytes of the <2 x float>
  %tmp = alloca <2 x float>, align 8
  store <2 x float> %vec, <2 x float>* %tmp, align 8
  %fp  = bitcast <2 x float>* %tmp to float*
  %a   = load float, float* %fp, align 4

  ret float %a
}

define float @test_derivative(float %x) {
entry:
  %d = call float (float (float)*, ...) @__enzyme_autodiff(float (float)* @tester, float %x)
  ret float %d
}

declare float @__enzyme_autodiff(float (float)*, ...)
; CHECK: @diffetester

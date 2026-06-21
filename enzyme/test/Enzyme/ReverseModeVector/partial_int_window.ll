; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,early-cse,sroa,instsimplify,%simplifycfg,adce)" -enzyme-preopt=false -S | FileCheck %s

source_filename = "partial_int_window"
target triple = "x86_64-pc-linux-gnu"

%struct.Gradients = type { [2 x float] }
%ret2v = type { <2 x float>, <2 x float> }

declare %struct.Gradients @__enzyme_autodiff(float (float)*, ...)

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
  %val  = bitcast <2 x float> %vec to i64

  ; Force "partial" use: only the first 4 bytes of the i64
  %tmp = alloca i64, align 8
  store i64 %val, i64* %tmp, align 8
  %fp  = bitcast i64* %tmp to float*
  %a   = load float, float* %fp, align 4

  ret float %a
}

define %struct.Gradients @test_derivative(float %x) {
entry:
  %d = call %struct.Gradients (float (float)*, ...) @__enzyme_autodiff(float (float)* @tester, metadata !"enzyme_width", i64 2, float %x)
  ret %struct.Gradients %d
}

; CHECK: define internal { [2 x float] } @diffe2tester(float %x, [2 x float] %differeturn)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %call_augmented = call [2 x %ret2v] @augmented_make(float %x)
; CHECK-NEXT:   %0 = call { [2 x float] } @diffe2make(float %x)
; CHECK-NEXT:   %1 = extractvalue { [2 x float] } %0, 0, 0
; CHECK-NEXT:   %2 = extractvalue { [2 x float] } %0, 0, 1
; CHECK-NEXT:   %.fca.0.insert = insertvalue [2 x float] poison, float %1, 0
; CHECK-NEXT:   %.fca.1.insert = insertvalue [2 x float] %.fca.0.insert, float %2, 1
; CHECK-NEXT:   %3 = insertvalue { [2 x float] } undef, [2 x float] %.fca.1.insert, 0
; CHECK-NEXT:   ret { [2 x float] } %3

; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -mem2reg -simplifycfg -early-cse -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -passes="enzyme,function(mem2reg,%simplifycfg,early-cse)" -enzyme-preopt=false -S | FileCheck %s

; Regression test: reverse vector mode handles partial-window accumulation into a fixed vector.

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
  %vec = extractvalue %ret2v %call, 0
  %tmp = alloca <2 x float>, align 8
  store <2 x float> %vec, <2 x float>* %tmp, align 8
  %fp = bitcast <2 x float>* %tmp to float*
  %a = load float, float* %fp, align 4
  ret float %a
}

define %struct.Gradients @test_derivative(float %x) {
entry:
  %d = call %struct.Gradients (float (float)*, ...) @__enzyme_autodiff(float (float)* @tester, metadata !"enzyme_width", i64 2, float %x)
  ret %struct.Gradients %d
}

; CHECK-LABEL: define internal { [2 x float] } @diffe2tester(float %x, [2 x float] %differeturn)
; CHECK: entry:
; CHECK:   %"vec'de" = alloca [2 x <2 x float>]
; CHECK:   %"call'de" = alloca [2 x %ret2v]
; CHECK:   %"x'de" = alloca [2 x float]
; CHECK:   %call_augmented = call [2 x %ret2v] @augmented_make(float %x)
; CHECK:   %"tmp'ipa" = alloca <2 x float>
; CHECK:   %"tmp'ipa1" = alloca <2 x float>
; CHECK:   %[[D0:.+]] = extractvalue [2 x float] %differeturn, 0
; CHECK:   %[[L0:.+]] = load float, {{.*}}align 4{{.*}}
; CHECK:   %[[A0:.+]] = fadd fast float %[[L0]], %[[D0]]
; CHECK:   store float %[[A0]], {{.*}}align 4{{.*}}
; CHECK:   %[[D1:.+]] = extractvalue [2 x float] %differeturn, 1
; CHECK:   %[[L1:.+]] = load float, {{.*}}align 4{{.*}}
; CHECK:   %[[A1:.+]] = fadd fast float %[[L1]], %[[D1]]
; CHECK:   store float %[[A1]], {{.*}}align 4{{.*}}
; CHECK:   %[[V0:.+]] = load <2 x float>, {{.*}}align 8{{.*}}
; CHECK:   %[[V1:.+]] = load <2 x float>, {{.*}}align 8{{.*}}
; CHECK:   %[[PACK:.+]] = load [2 x <2 x float>], {{.*}}align 8
; CHECK:   %[[LANE0V:.+]] = extractvalue [2 x <2 x float>] %[[PACK]], 0
; CHECK:   %[[LANE0:.+]] = extractelement <2 x float> %[[LANE0V]], i32 0
; CHECK:   %[[LANE1V:.+]] = extractvalue [2 x <2 x float>] %[[PACK]], 1
; CHECK:   %[[LANE1:.+]] = extractelement <2 x float> %[[LANE1V]], i32 0
; CHECK:   %[[MAKE:.+]] = call { [2 x float] } @diffe2make(float %x)
; CHECK:   ret { [2 x float] }

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
  store <2 x float> %vec, ptr %tmp, align 8
  %a = load float, ptr %tmp, align 4
  ret float %a
}

define %struct.Gradients @test_derivative(float %x) {
entry:
  %d = call %struct.Gradients (float (float)*, ...) @__enzyme_autodiff(float (float)* @tester, metadata !"enzyme_width", i64 2, float %x)
  ret %struct.Gradients %d
}

; CHECK: define internal { [2 x float] } @diffe2tester(float %x, [2 x float] %differeturn) #{{.+}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %"vec'de" = alloca [2 x <2 x float>], align 8
; CHECK-NEXT:   store [2 x <2 x float>] zeroinitializer, ptr %"vec'de", align 8
; CHECK-NEXT:   %"call'de" = alloca [2 x %ret2v], align 8
; CHECK-NEXT:   store [2 x %ret2v] zeroinitializer, ptr %"call'de", align 8
; CHECK-NEXT:   %"x'de" = alloca [2 x float], align 4
; CHECK-NEXT:   store [2 x float] zeroinitializer, ptr %"x'de", align 4
; CHECK-NEXT:   %call_augmented = call [2 x %ret2v] @augmented_make(float %x)
; CHECK-NEXT:   %"tmp'ipa" = alloca <2 x float>, align 8
; CHECK-NEXT:   %"tmp'ipa1" = alloca <2 x float>, align 8
; CHECK-NEXT:   store <2 x float> zeroinitializer, ptr %"tmp'ipa", align 8
; CHECK-NEXT:   store <2 x float> zeroinitializer, ptr %"tmp'ipa1", align 8
; CHECK-NEXT:   %0 = extractvalue [2 x float] %differeturn, 0
; CHECK-NEXT:   %1 = load float, ptr %"tmp'ipa", align 4{{.*}}
; CHECK-NEXT:   %2 = fadd fast float %1, %0
; CHECK-NEXT:   store float %2, ptr %"tmp'ipa", align 4{{.*}}
; CHECK-NEXT:   %3 = extractvalue [2 x float] %differeturn, 1
; CHECK-NEXT:   %4 = load float, ptr %"tmp'ipa1", align 4{{.*}}
; CHECK-NEXT:   %5 = fadd fast float %4, %3
; CHECK-NEXT:   store float %5, ptr %"tmp'ipa1", align 4{{.*}}
; CHECK-NEXT:   %6 = load <2 x float>, ptr %"tmp'ipa", align 8{{.*}}
; CHECK-NEXT:   %7 = load <2 x float>, ptr %"tmp'ipa1", align 8{{.*}}
; CHECK-NEXT:   store <2 x float> zeroinitializer, ptr %"tmp'ipa", align 8{{.*}}
; CHECK-NEXT:   store <2 x float> zeroinitializer, ptr %"tmp'ipa1", align 8{{.*}}
; CHECK-NEXT:   %8 = load <2 x float>, ptr %"vec'de", align 8
; CHECK-NEXT:   %9 = fadd fast <2 x float> %8, %6
; CHECK-NEXT:   store <2 x float> %9, ptr %"vec'de", align 8
; CHECK-NEXT:   %10 = getelementptr inbounds [2 x <2 x float>], ptr %"vec'de", i32 0, i32 1
; CHECK-NEXT:   %11 = load <2 x float>, ptr %10, align 8
; CHECK-NEXT:   %12 = fadd fast <2 x float> %11, %7
; CHECK-NEXT:   store <2 x float> %12, ptr %10, align 8
; CHECK-NEXT:   %13 = load [2 x <2 x float>], ptr %"vec'de", align 8
; CHECK-NEXT:   %14 = extractvalue [2 x <2 x float>] %13, 0
; CHECK-NEXT:   %15 = extractelement <2 x float> %14, i32 0
; CHECK-NEXT:   %16 = insertelement <2 x float> zeroinitializer, float %15, i32 0
; CHECK-NEXT:   %17 = load <2 x float>, ptr %"call'de", align 8
; CHECK-NEXT:   %18 = fadd fast <2 x float> %17, %16
; CHECK-NEXT:   store <2 x float> %18, ptr %"call'de", align 8
; CHECK-NEXT:   %19 = extractvalue [2 x <2 x float>] %13, 1
; CHECK-NEXT:   %20 = extractelement <2 x float> %19, i32 0
; CHECK-NEXT:   %21 = insertelement <2 x float> zeroinitializer, float %20, i32 0
; CHECK-NEXT:   %22 = getelementptr inbounds [2 x %ret2v], ptr %"call'de", i32 0, i32 1, i32 0
; CHECK-NEXT:   %23 = load <2 x float>, ptr %22, align 8
; CHECK-NEXT:   %24 = fadd fast <2 x float> %23, %21
; CHECK-NEXT:   store <2 x float> %24, ptr %22, align 8
; CHECK-NEXT:   store [2 x <2 x float>] zeroinitializer, ptr %"vec'de", align 8
; CHECK-NEXT:   %25 = call { [2 x float] } @diffe2make(float %x)
; CHECK-NEXT:   %26 = extractvalue { [2 x float] } %25, 0, 0
; CHECK-NEXT:   %27 = load float, ptr %"x'de", align 4
; CHECK-NEXT:   %28 = fadd fast float %27, %26
; CHECK-NEXT:   store float %28, ptr %"x'de", align 4
; CHECK-NEXT:   %29 = extractvalue { [2 x float] } %25, 0, 1
; CHECK-NEXT:   %30 = getelementptr inbounds [2 x float], ptr %"x'de", i32 0, i32 1
; CHECK-NEXT:   %31 = load float, ptr %30, align 4
; CHECK-NEXT:   %32 = fadd fast float %31, %29
; CHECK-NEXT:   store float %32, ptr %30, align 4
; CHECK-NEXT:   %33 = load [2 x float], ptr %"x'de", align 4
; CHECK-NEXT:   %34 = insertvalue { [2 x float] } undef, [2 x float] %33, 0
; CHECK-NEXT:   ret { [2 x float] } %34
; CHECK-NEXT: }

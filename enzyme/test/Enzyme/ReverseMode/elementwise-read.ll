; RUN: if [ %llvmver -lt 16 ]; then %opt < %s %loadEnzyme -enzyme-preopt=false -preserve-nvvm -enzyme -enzyme-detect-readthrow=0 -S | FileCheck %s; fi
; RUN: %opt < %s %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="preserve-nvvm,enzyme" -S | FileCheck %s

; ModuleID = 'elementwise-read.ll'
source_filename = "elementwise-read.ll"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-ni:10:11:12:13"
target triple = "nvptx64-nvidia-cuda"

@.str.enzyme_elementwise_read = private unnamed_addr constant [24 x i8] c"enzyme_elementwise_read\00", section "llvm.metadata"
@.str.file = private unnamed_addr constant [17 x i8] c"elementwise-read\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { i8*, i8*, i8*, i32, i8* }] [{ i8*, i8*, i8*, i32, i8* } { i8* bitcast (float (float addrspace(1)*)* @vmul to i8*), i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.enzyme_elementwise_read, i32 0, i32 0), i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.file, i32 0, i32 0), i32 1, i8* null }], section "llvm.metadata"

declare float @llvm.nvvm.ldg.global.f.f32.p1f32(float addrspace(1)* nocapture, i32)

define float @vmul(float addrspace(1)* %inp) {
top:
  %ld = call float @llvm.nvvm.ldg.global.f.f32.p1f32(float addrspace(1)* %inp, i32 4)
  ret float %ld
}

define float @test_derivative(float addrspace(1)* %inp, float addrspace(1)* %dinp) {
entry:
  %0 = tail call float (float (float addrspace(1)*)*, ...) @__enzyme_autodiff(float (float addrspace(1)*)* nonnull @vmul, float addrspace(1)* %inp, float addrspace(1)* %dinp)
  ret float %0
}

declare float @__enzyme_autodiff(float (float addrspace(1)*)*, ...)

; CHECK: @llvm.global.annotations = appending global [1 x { i8*, i8*, i8*, i32, i8* }] zeroinitializer
; CHECK-LABEL: define float @vmul(float addrspace(1)* %inp)

; CHECK-LABEL: define internal void @diffevmul(float addrspace(1)* %inp, float addrspace(1)* %"inp'", float %differeturn)
; CHECK-NOT: atomicrmw
; CHECK-NEXT: top:
; CHECK-NEXT:   %[[OLD:[^ ]+]] = load float, float addrspace(1)* %"inp'"
; CHECK-NEXT:   %[[NEW:[^ ]+]] = fadd fast float %[[OLD]], %differeturn
; CHECK-NEXT:   store float %[[NEW]], float addrspace(1)* %"inp'"
; CHECK-NEXT:   ret void

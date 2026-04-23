; RUN: split-file %s %t
; RUN: if [ %llvmver -le 20 ]; then %opt < %t/llvm20.ll %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme" -S | FileCheck %t/llvm20.ll; fi
; RUN: if [ %llvmver -gt 20 ]; then %opt < %t/llvm21plus.ll %newLoadEnzyme -enzyme-preopt=false -enzyme-detect-readthrow=0 -passes="enzyme" -S | FileCheck %t/llvm21plus.ll; fi

;--- llvm20.ll
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.barrier0()
declare float @__enzyme_autodiff(float (float)*, ...)

define float @f_sync(float %x) {
entry:
  call void @llvm.nvvm.barrier0()
  %res = fadd float %x, 1.000000e+00
  ret float %res
}

define float @test(float %x) {
entry:
  %r = call float (float (float)*, ...) @__enzyme_autodiff(float (float)* @f_sync, float %x)
  ret float %r
}

; CHECK: define internal { float } @diffef_sync(float %x, float %differeturn)
; CHECK: call void @llvm.nvvm.barrier0()
; CHECK: call void @llvm.nvvm.barrier0()
; CHECK: ret { float }

;--- llvm21plus.ll
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32)
declare void @llvm.nvvm.barrier.cta.sync.aligned.count(i32, i32)
declare float @__enzyme_autodiff(float (float)*, ...)

define float @f_sync_all(float %x) {
entry:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 7)
  %res = fadd float %x, 1.000000e+00
  ret float %res
}

define float @test_all(float %x) {
entry:
  %r = call float (float (float)*, ...) @__enzyme_autodiff(float (float)* @f_sync_all, float %x)
  ret float %r
}

define float @f_sync_count(float %x) {
entry:
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 7, i32 16)
  %res = fadd float %x, 1.000000e+00
  ret float %res
}

define float @test_count(float %x) {
entry:
  %r = call float (float (float)*, ...) @__enzyme_autodiff(float (float)* @f_sync_count, float %x)
  ret float %r
}

; CHECK: define internal { float } @diffef_sync_all(float %x, float %differeturn)
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 7)
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 7)
; CHECK: ret { float }

; CHECK: define internal { float } @diffef_sync_count(float %x, float %differeturn)
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 7, i32 16)
; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 7, i32 16)
; CHECK: ret { float }

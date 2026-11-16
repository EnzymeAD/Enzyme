; RUN: rm -rf %t && mkdir -p %t && cp %S/../Inputs/lambda_stub/launches_*.json %t/
; RUN: not %clang -O0 -x ir %s %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t -mllvm -poseidon-unit=fixture -S -emit-llvm -o /dev/null 2>&1 | FileCheck %s

; The launch stub of a profiled kernel (Inputs/lambda_stub) that does not end
; in the launch API the profiling run rewrites, here the cudaLaunch of a host
; half compiled without CUDA's headers, cannot pass the kernel its shadows: an
; error, not a profiled kernel launched without them.

; CHECK: error: {{.*}}Poseidon: the launch stub _ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_ does not end in __cuda/__hipPopCallConfiguration and cuda/hipLaunchKernel, so the profiling run cannot pass its kernel the shadows it is profiled with. Compile against the CUDA or ROCm headers, which declare that launch API.

source_filename = "lambda_stub.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.DeviceTensor = type { i32, ptr, [2 x i32] }
%class.anon = type { i32, i32, %struct.DeviceTensor, %struct.DeviceTensor, %struct.DeviceTensor }

define void @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(i32 %0, ptr byval(%class.anon) align 8 %1) {
  %3 = call i32 @cudaLaunch(ptr @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_)
  ret void
}

declare i32 @cudaLaunch(ptr)

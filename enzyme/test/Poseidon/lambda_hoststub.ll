; RUN: %clang -O0 -x ir %s %clangLoadPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%S/Inputs/lambda_stub -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=GEN %s
; RUN: %clang -O0 -x ir %s %clangLoadPoseidon -mllvm -poseidon-profile-use=%S/Inputs/lambda_stub -mllvm -poseidon-cache=%S/Inputs/lambda_stub -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=USE %s
; REQUIRES: poseidon
; The launch stub of a kernel that takes a lambda by value
; (Inputs/lambda_stub/lambda_stub.cu, clang --cuda-host-only -O0): the
; descriptors name pointers and sizes as fields of the lambda, which only the
; launch's own argument array holds in every host ABI. The profiling launch
; hands the runtime that array and each pointer's offset; the kernel's
; device-side name carries the -fgpu-rdc postfix of an internal kernel. The
; dispatch reads its operands and extents from the same array and runs only
; when the extents the recognition assumed equal are, launching the kernel as
; written otherwise.

; GEN: [poseidon-profgen] launch of _ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd now allocates and seeds 3 shadow buffer(s)
; GEN: @poseidon.off.{{.*}} = private unnamed_addr constant [3 x i32] [i32 16, i32 40, i32 64]
; GEN-LABEL: define void @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(
; GEN: %[[ARGS:[0-9]+]] = alloca ptr, i64 2
; GEN: call i32 @__cudaPopCallConfiguration(ptr %[[GRID:[0-9]+]], ptr %[[BLOCK:[0-9]+]], ptr %[[SHMEM:[0-9]+]], ptr %[[STREAM:[0-9]+]])
; GEN: call void @__poseidon_launch_profiled(ptr @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_, i32 0, ptr %[[GRID]], ptr %[[BLOCK]], i64 %{{[0-9]+}}, ptr %{{[0-9]+}}, i32 2, ptr %[[ARGS]], i32 3, ptr @poseidon.ptr.{{.*}}, ptr @poseidon.off.{{.*}}, ptr @poseidon.seed.{{.*}})
; GEN-NOT: call {{.*}}@cudaLaunchKernel(

; USE: [ozaki-host-dispatch] replaced the launch in stub _ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_ with __poseidon_ozaki_dgemm_ex (C=arg1[+64] A=arg1[+40] B=arg1[+16]), if arg1[+4] == arg1[+24], if arg1[+4] == arg1[+48]
; USE-LABEL: define void @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(
; USE: %[[ARGS:[0-9]+]] = alloca ptr, i64 2
; USE: %[[S:[0-9]+]] = getelementptr inbounds ptr, ptr %[[ARGS]], i32 1
; USE-NEXT: %[[L:[0-9]+]] = load ptr, ptr %[[S]]
; USE-NEXT: %[[F:[0-9]+]] = getelementptr inbounds i8, ptr %[[L]], i64 64
; USE-NEXT: %[[C:[0-9]+]] = load ptr, ptr %[[F]]
; USE: %[[K:[0-9]+]] = mul i32 %{{[0-9]+}}, 3
; USE: %[[EQ0:[0-9]+]] = icmp eq i32
; USE: %[[EQ1:[0-9]+]] = icmp eq i32
; USE: %[[OK:[0-9]+]] = and i1 %[[EQ0]], %[[EQ1]]
; USE-NEXT: br i1 %[[OK]], label %[[DISPATCH:[0-9]+]], label %[[LAUNCH:[0-9]+]]
; USE: [[DISPATCH]]:
; USE-NEXT: call void @__poseidon_ozaki_dgemm_ex(ptr %[[C]], ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %[[K]], i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 0, i32 1, i32 1, double 1.000000e+00, double 1.000000e+00, ptr null, i32 13)
; USE: [[LAUNCH]]:
; USE-NEXT: call noundef i32 @cudaLaunchKernel(

; ModuleID = 'lambda_stub.cu'
source_filename = "lambda_stub.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.DeviceTensor = type { i32, ptr, [2 x i32] }
%class.anon = type { i32, i32, %struct.DeviceTensor, %struct.DeviceTensor, %struct.DeviceTensor }
%struct.dim3 = type { i32, i32, i32 }

define void @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(i32 noundef %0, ptr noundef byval(%class.anon) align 8 %1) #4 {
  %3 = alloca i32, align 4
  %4 = alloca %struct.dim3, align 8
  %5 = alloca %struct.dim3, align 8
  %6 = alloca i64, align 8
  %7 = alloca ptr, align 8
  %8 = alloca { i64, i32 }, align 8
  %9 = alloca { i64, i32 }, align 8
  store i32 %0, ptr %3, align 4
  %10 = alloca ptr, i64 2, align 16
  %11 = getelementptr ptr, ptr %10, i32 0
  store ptr %3, ptr %11, align 8
  %12 = getelementptr ptr, ptr %10, i32 1
  store ptr %1, ptr %12, align 8
  %13 = call i32 @__cudaPopCallConfiguration(ptr %4, ptr %5, ptr %6, ptr %7)
  %14 = load i64, ptr %6, align 8
  %15 = load ptr, ptr %7, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %8, ptr align 8 %4, i64 12, i1 false)
  %16 = getelementptr inbounds nuw { i64, i32 }, ptr %8, i32 0, i32 0
  %17 = load i64, ptr %16, align 8
  %18 = getelementptr inbounds nuw { i64, i32 }, ptr %8, i32 0, i32 1
  %19 = load i32, ptr %18, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %9, ptr align 8 %5, i64 12, i1 false)
  %20 = getelementptr inbounds nuw { i64, i32 }, ptr %9, i32 0, i32 0
  %21 = load i64, ptr %20, align 8
  %22 = getelementptr inbounds nuw { i64, i32 }, ptr %9, i32 0, i32 1
  %23 = load i32, ptr %22, align 8
  %24 = call noundef i32 @cudaLaunchKernel(ptr noundef @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_, i64 %17, i32 %19, i64 %21, i32 %23, ptr noundef %10, i64 noundef %14, ptr noundef %15)
  br label %25

25:                                               ; preds = %2
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)
declare i32 @__cudaPopCallConfiguration(ptr, ptr, ptr, ptr)
declare i32 @cudaLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

attributes #4 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }

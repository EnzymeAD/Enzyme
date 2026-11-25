; RUN: rm -rf %t && mkdir -p %t && cp %S/Inputs/launch_stub_cache/launches_*.json %t/
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %s %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/launch_stub_cache -mllvm -poseidon-cache=%t -mllvm -poseidon-compilation-id=fixture -S -emit-llvm -mllvm -poseidon-print -o - 2>&1 | FileCheck --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s
; Host-side launches replaced by library dispatches, which read their operands from the launch's argument array, from the descriptors in the manifest of hoststub.cu, unit fixture, in Inputs/launch_stub_cache (a square scheme-2 Direct FP16 one and a non-square scheme-1 TCEC one); the pass runs at PipelineStart of the clang pipeline, which is why this drives clang on IR. Host IR from clang -x cuda --cuda-host-only -O0 -S -emit-llvm of
;   __global__ void gemm_kernel(const double *A, const double *B, double *C) {}
;   __global__ void gemm_tall_kernel(const double *A, const double *B, double *C,
;                                    int ncols) {}
;   void launch(const double *A, const double *B, double *C, int ncols) {
;     gemm_kernel<<<dim3(32, 32), dim3(16, 16)>>>(A, B, C);
;     gemm_tall_kernel<<<dim3(8, 32), dim3(16, 16)>>>(A, B, C, ncols);
;   }

; ModuleID = 'hoststub.cu'
source_filename = "hoststub.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }

$_ZN4dim3C2Ejjj = comdat any

define dso_local void @_Z26__device_stub__gemm_kernelPKdS0_Pd(ptr noundef %0, ptr noundef %1, ptr noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca %struct.dim3, align 8
  %8 = alloca %struct.dim3, align 8
  %9 = alloca i64, align 8
  %10 = alloca ptr, align 8
  %11 = alloca { i64, i32 }, align 8
  %12 = alloca { i64, i32 }, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %13 = alloca ptr, i64 3, align 16
  %14 = getelementptr ptr, ptr %13, i32 0
  store ptr %4, ptr %14, align 8
  %15 = getelementptr ptr, ptr %13, i32 1
  store ptr %5, ptr %15, align 8
  %16 = getelementptr ptr, ptr %13, i32 2
  store ptr %6, ptr %16, align 8
  %17 = call i32 @__cudaPopCallConfiguration(ptr %7, ptr %8, ptr %9, ptr %10)
  %18 = load i64, ptr %9, align 8
  %19 = load ptr, ptr %10, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %11, ptr align 8 %7, i64 12, i1 false)
  %20 = getelementptr inbounds nuw { i64, i32 }, ptr %11, i32 0, i32 0
  %21 = load i64, ptr %20, align 8
  %22 = getelementptr inbounds nuw { i64, i32 }, ptr %11, i32 0, i32 1
  %23 = load i32, ptr %22, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %12, ptr align 8 %8, i64 12, i1 false)
  %24 = getelementptr inbounds nuw { i64, i32 }, ptr %12, i32 0, i32 0
  %25 = load i64, ptr %24, align 8
  %26 = getelementptr inbounds nuw { i64, i32 }, ptr %12, i32 0, i32 1
  %27 = load i32, ptr %26, align 8
  %28 = call noundef i32 @cudaLaunchKernel(ptr noundef @_Z26__device_stub__gemm_kernelPKdS0_Pd, i64 %21, i32 %23, i64 %25, i32 %27, ptr noundef %13, i64 noundef %18, ptr noundef %19)
  br label %29

29:                                               ; preds = %3
  ret void
}

declare i32 @__cudaPopCallConfiguration(ptr, ptr, ptr, ptr)

declare i32 @cudaLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr)

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1

define dso_local void @_Z31__device_stub__gemm_tall_kernelPKdS0_Pdi(ptr noundef %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca %struct.dim3, align 8
  %10 = alloca %struct.dim3, align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  %13 = alloca { i64, i32 }, align 8
  %14 = alloca { i64, i32 }, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  store i32 %3, ptr %8, align 4
  %15 = alloca ptr, i64 4, align 16
  %16 = getelementptr ptr, ptr %15, i32 0
  store ptr %5, ptr %16, align 8
  %17 = getelementptr ptr, ptr %15, i32 1
  store ptr %6, ptr %17, align 8
  %18 = getelementptr ptr, ptr %15, i32 2
  store ptr %7, ptr %18, align 8
  %19 = getelementptr ptr, ptr %15, i32 3
  store ptr %8, ptr %19, align 8
  %20 = call i32 @__cudaPopCallConfiguration(ptr %9, ptr %10, ptr %11, ptr %12)
  %21 = load i64, ptr %11, align 8
  %22 = load ptr, ptr %12, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %13, ptr align 8 %9, i64 12, i1 false)
  %23 = getelementptr inbounds nuw { i64, i32 }, ptr %13, i32 0, i32 0
  %24 = load i64, ptr %23, align 8
  %25 = getelementptr inbounds nuw { i64, i32 }, ptr %13, i32 0, i32 1
  %26 = load i32, ptr %25, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %14, ptr align 8 %10, i64 12, i1 false)
  %27 = getelementptr inbounds nuw { i64, i32 }, ptr %14, i32 0, i32 0
  %28 = load i64, ptr %27, align 8
  %29 = getelementptr inbounds nuw { i64, i32 }, ptr %14, i32 0, i32 1
  %30 = load i32, ptr %29, align 8
  %31 = call noundef i32 @cudaLaunchKernel(ptr noundef @_Z31__device_stub__gemm_tall_kernelPKdS0_Pdi, i64 %24, i32 %26, i64 %28, i32 %30, ptr noundef %15, i64 noundef %21, ptr noundef %22)
  br label %32

32:                                               ; preds = %4
  ret void
}

define dso_local void @_Z6launchPKdS0_Pdi(ptr noundef %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) #2 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca %struct.dim3, align 4
  %10 = alloca %struct.dim3, align 4
  %11 = alloca { i64, i32 }, align 4
  %12 = alloca { i64, i32 }, align 4
  %13 = alloca %struct.dim3, align 4
  %14 = alloca %struct.dim3, align 4
  %15 = alloca { i64, i32 }, align 4
  %16 = alloca { i64, i32 }, align 4
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  store i32 %3, ptr %8, align 4
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %9, i32 noundef 32, i32 noundef 32, i32 noundef 1) #5
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %10, i32 noundef 16, i32 noundef 16, i32 noundef 1) #5
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %11, ptr align 4 %9, i64 12, i1 false)
  %17 = getelementptr inbounds nuw { i64, i32 }, ptr %11, i32 0, i32 0
  %18 = load i64, ptr %17, align 4
  %19 = getelementptr inbounds nuw { i64, i32 }, ptr %11, i32 0, i32 1
  %20 = load i32, ptr %19, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %12, ptr align 4 %10, i64 12, i1 false)
  %21 = getelementptr inbounds nuw { i64, i32 }, ptr %12, i32 0, i32 0
  %22 = load i64, ptr %21, align 4
  %23 = getelementptr inbounds nuw { i64, i32 }, ptr %12, i32 0, i32 1
  %24 = load i32, ptr %23, align 4
  %25 = call i32 @__cudaPushCallConfiguration(i64 %18, i32 %20, i64 %22, i32 %24, i64 noundef 0, ptr noundef null) #5
  %26 = icmp ne i32 %25, 0
  br i1 %26, label %31, label %27

27:                                               ; preds = %4
  %28 = load ptr, ptr %5, align 8
  %29 = load ptr, ptr %6, align 8
  %30 = load ptr, ptr %7, align 8
  call void @_Z26__device_stub__gemm_kernelPKdS0_Pd(ptr noundef %28, ptr noundef %29, ptr noundef %30) #5
  br label %31

31:                                               ; preds = %27, %4
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %13, i32 noundef 8, i32 noundef 32, i32 noundef 1) #5
  call void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %14, i32 noundef 16, i32 noundef 16, i32 noundef 1) #5
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %15, ptr align 4 %13, i64 12, i1 false)
  %32 = getelementptr inbounds nuw { i64, i32 }, ptr %15, i32 0, i32 0
  %33 = load i64, ptr %32, align 4
  %34 = getelementptr inbounds nuw { i64, i32 }, ptr %15, i32 0, i32 1
  %35 = load i32, ptr %34, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %16, ptr align 4 %14, i64 12, i1 false)
  %36 = getelementptr inbounds nuw { i64, i32 }, ptr %16, i32 0, i32 0
  %37 = load i64, ptr %36, align 4
  %38 = getelementptr inbounds nuw { i64, i32 }, ptr %16, i32 0, i32 1
  %39 = load i32, ptr %38, align 4
  %40 = call i32 @__cudaPushCallConfiguration(i64 %33, i32 %35, i64 %37, i32 %39, i64 noundef 0, ptr noundef null) #5
  %41 = icmp ne i32 %40, 0
  br i1 %41, label %47, label %42

42:                                               ; preds = %31
  %43 = load ptr, ptr %5, align 8
  %44 = load ptr, ptr %6, align 8
  %45 = load ptr, ptr %7, align 8
  %46 = load i32, ptr %8, align 4
  call void @_Z31__device_stub__gemm_tall_kernelPKdS0_Pdi(ptr noundef %43, ptr noundef %44, ptr noundef %45, i32 noundef %46) #5
  br label %47

47:                                               ; preds = %42, %31
  ret void
}

declare i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64 noundef, ptr noundef) #3

define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(ptr noundef nonnull align 4 dereferenceable(12) %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) unnamed_addr #4 comdat align 2 {
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store i32 %1, ptr %6, align 4
  store i32 %2, ptr %7, align 4
  store i32 %3, ptr %8, align 4
  %9 = load ptr, ptr %5, align 8
  %10 = getelementptr inbounds nuw %struct.dim3, ptr %9, i32 0, i32 0
  %11 = load i32, ptr %6, align 4
  store i32 %11, ptr %10, align 4
  %12 = getelementptr inbounds nuw %struct.dim3, ptr %9, i32 0, i32 1
  %13 = load i32, ptr %7, align 4
  store i32 %13, ptr %12, align 4
  %14 = getelementptr inbounds nuw %struct.dim3, ptr %9, i32 0, i32 2
  %15 = load i32, ptr %8, align 4
  store i32 %15, ptr %14, align 4
  ret void
}

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #4 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #5 = { "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}

; CHECK: [poseidon] replaced the launch in stub _Z26__device_stub__gemm_kernelPKdS0_Pd with __poseidon_direct_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; CHECK: [poseidon] replaced the launch in stub _Z31__device_stub__gemm_tall_kernelPKdS0_Pdi with __poseidon_tcec_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; CHECK-LABEL: define {{.*}} @_Z26__device_stub__gemm_kernelPKdS0_Pd(
; CHECK: %[[ARGS:[0-9]+]] = alloca ptr, i64 3
; CHECK: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; CHECK: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; CHECK: %[[CS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 2|i64 16}}{{$}}
; CHECK-NEXT: %[[CL:[0-9]+]] = load ptr, ptr %[[CS]], align 8
; CHECK-NEXT: %[[C:[0-9]+]] = load ptr, ptr %[[CL]], align 8
; CHECK-NEXT: %[[AS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 0|i64 0}}{{$}}
; CHECK-NEXT: %[[AL:[0-9]+]] = load ptr, ptr %[[AS]], align 8
; CHECK-NEXT: %[[A:[0-9]+]] = load ptr, ptr %[[AL]], align 8
; CHECK-NEXT: %[[BS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; CHECK-NEXT: %[[BL:[0-9]+]] = load ptr, ptr %[[BS]], align 8
; CHECK-NEXT: %[[B:[0-9]+]] = load ptr, ptr %[[BL]], align 8
; CHECK-NEXT: call void @__poseidon_direct_dgemm(ptr %[[C]], ptr %[[A]], ptr %[[B]], i32 512, i32 512, i32 512, i32 512, i32 512, i32 512, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]], i32 1)
; CHECK-NEXT: br label
; CHECK-LABEL: define {{.*}} @_Z31__device_stub__gemm_tall_kernelPKdS0_Pdi(
; CHECK: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; CHECK: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; CHECK: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; CHECK: %[[CS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 2|i64 16}}{{$}}
; CHECK-NEXT: %[[CL:[0-9]+]] = load ptr, ptr %[[CS]], align 8
; CHECK-NEXT: %[[C:[0-9]+]] = load ptr, ptr %[[CL]], align 8
; CHECK-NEXT: %[[AS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 0|i64 0}}{{$}}
; CHECK-NEXT: %[[AL:[0-9]+]] = load ptr, ptr %[[AS]], align 8
; CHECK-NEXT: %[[A:[0-9]+]] = load ptr, ptr %[[AL]], align 8
; CHECK-NEXT: %[[BS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; CHECK-NEXT: %[[BL:[0-9]+]] = load ptr, ptr %[[BS]], align 8
; CHECK-NEXT: %[[B:[0-9]+]] = load ptr, ptr %[[BL]], align 8
; CHECK-NEXT: call void @__poseidon_tcec_dgemm(ptr %[[C]], ptr %[[A]], ptr %[[B]], i32 128, i32 512, i32 256, i32 256, i32 256, i32 512, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]], i32 3)
; CHECK-NEXT: br label

; DEFINE: %{bad} = %clang -O0 -x ir -mllvm -poseidon-cache=%t/bad -mllvm -poseidon-compilation-id=fixture -S -emit-llvm -o /dev/null %loadClangPoseidon
; RUN: rm -rf %t && split-file %s %t && mkdir %t/c %t/bad && cp %S/Inputs/launch_stub_cache/launches_*.json %t/c/
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %t/hoststub.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/launch_stub_cache -mllvm -poseidon-cache=%t/c -mllvm -poseidon-compilation-id=fixture -S -emit-llvm -mllvm -poseidon-print -o - 2>&1 | FileCheck --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s

; Host launches replaced by library calls reading their operands from the
; launch's argument array, from the descriptors of Inputs/launch_stub_cache (a
; square scheme-2 Direct FP16 one and a non-square scheme-1 TCEC one). Host IR
; of clang -x cuda --cuda-host-only -O0 for gemm_kernel<<<dim3(32, 32),
; dim3(16, 16)>>>(A, B, C) and gemm_tall_kernel<<<dim3(8, 32), dim3(16, 16)>>>(A, B, C, ncols).
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

; A descriptor of the manifest that does not parse stops the host compile,
; naming the manifest and the descriptor.
; RUN: cp %S/Inputs/launch_stub_cache/launches_*.json %S/Inputs/lambda_stub/launches_*.json %t/bad/
; RUN: sed -i 's/ 0 0 0 0 rd [^"]*"/ 0 0 0"/' %t/bad/launches_hoststub.cu_*.json
; RUN: not %{bad} %t/empty.ll -mllvm -poseidon-profile-use=%S/Inputs/launch_stub_cache 2>&1 | FileCheck --check-prefix=SHORT -DCACHE=%t/bad %s
; RUN: cp %S/Inputs/launch_stub_cache/launches_*.json %t/bad/
; RUN: sed -i 's/2 0 1 1 3 0 0 0 1 rd/2 0 1 1 99 0 0 0 1 rd/' %t/bad/launches_hoststub.cu_*.json
; RUN: not %{bad} %t/empty.ll -mllvm -poseidon-profile-use=%S/Inputs/launch_stub_cache 2>&1 | FileCheck --check-prefix=MODULI -DCACHE=%t/bad %s
; RUN: sed 's/hoststub\.cu/lambda_stub.cu/' %t/empty.ll > %t/lambda.ll
; RUN: sed -i 's/ off 16 40 64"/ off 16 40"/' %t/bad/launches_lambda_stub.cu_*.json
; RUN: not %{bad} %t/lambda.ll -mllvm -poseidon-profile-generate 2>&1 | FileCheck --check-prefix=OFFSETS -DCACHE=%t/bad %s
; RUN: sed -i 's/"_Z11gemm_kernelPKdS0_Pd [^"]*"/7/' %t/bad/launches_hoststub.cu_*.json
; RUN: not %{bad} %t/empty.ll -mllvm -poseidon-profile-use=%S/Inputs/launch_stub_cache 2>&1 | FileCheck --check-prefix=NOTSTRING -DCACHE=%t/bad %s

; SHORT: {{^}}error: Poseidon: the launch manifest [[CACHE]]/launches_hoststub.cu_{{[0-9A-F]+}}.json holds the library calls descriptor '_Z11gemm_kernelPKdS0_Pd 2 0 1 1 1 0 0 0', which does not parse (it ends before its scheme); remove the -poseidon-cache directory and compile the unit again{{$}}
; MODULI: {{^}}error: Poseidon: the launch manifest [[CACHE]]/launches_hoststub.cu_{{[0-9A-F]+}}.json holds the library calls descriptor '_Z16gemm_tall_kernelPKdS0_Pdi 2 0 1 1 99 0 0 0 1 rd 0 1 128 0 1 512 0 1 256 0 1 256 0 1 256 0 1 512 0.000000e+00', which does not parse (its modulus field 99 is above 14); remove the -poseidon-cache directory and compile the unit again{{$}}
; OFFSETS: {{^}}error: Poseidon: the launch manifest [[CACHE]]/launches_lambda_stub.cu_{{[0-9A-F]+}}.json holds the profile generation descriptor '_ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd 0 2 3 1 1 1 0 0 8 off 16 40', which does not parse (it ends before its pointer offset); remove the -poseidon-cache directory and compile the unit again{{$}}
; NOTSTRING: {{^}}error: Poseidon: the launch manifest [[CACHE]]/launches_hoststub.cu_{{[0-9A-F]+}}.json holds a library calls entry that is not a string; remove the -poseidon-cache directory and compile the unit again{{$}}

;--- empty.ll
source_filename = "hoststub.cu"
target triple = "x86_64-unknown-linux-gnu"

define void @f() {
  ret void
}

;--- hoststub.ll
source_filename = "hoststub.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }

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

define dso_local void @_Z6launchPKdS0_Pdi(ptr %0, ptr %1, ptr %2, i32 %3) {
  call void @_Z26__device_stub__gemm_kernelPKdS0_Pd(ptr %0, ptr %1, ptr %2)
  call void @_Z31__device_stub__gemm_tall_kernelPKdS0_Pdi(ptr %0, ptr %1, ptr %2, i32 %3)
  ret void
}

attributes #0 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}

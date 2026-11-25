; RUN: rm -rf %t && mkdir -p %t && cp %S/Inputs/lambda_stub/launches_*.json %t/
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu %s %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t -mllvm -poseidon-print -mllvm -poseidon-compilation-id=fixture -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=GEN --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu %s %loadClangPoseidon -mllvm -poseidon-profile-use=%S/Inputs/lambda_stub -mllvm -poseidon-cache=%t -mllvm -poseidon-compilation-id=fixture -S -emit-llvm -mllvm -poseidon-print -o - 2>&1 | FileCheck --check-prefix=USE --implicit-check-not='call {{.*}}@cudaLaunchKernel(' --implicit-check-not='call void @__poseidon_ozakiii_dgemm' %s
; The launch stub of a kernel that takes a lambda by value, host IR from
; clang -x cuda --cuda-host-only -O0 -S -emit-llvm of
;   template <int D> struct DeviceTensor {
;     int capacity;
;     double *data;
;     int sizes[D];
;   };
;   template <typename BODY> __global__ static void CuKernel2D(int N, BODY body) {
;     int k = blockIdx.x * blockDim.z + threadIdx.z;
;     if (k < N)
;       body(k);
;   }
;   void reduce(int nq, int ne, int nd, double *Q, double *G, double *y) {
;     DeviceTensor<2> QR{nq * ne, Q, {nq, ne}};
;     DeviceTensor<2> GR{nq * nd, G, {nq, nd}};
;     DeviceTensor<2> Y{nd * ne, y, {nd, ne}};
;     auto body = [=] __host__ __device__(int e) {
;       for (int i = threadIdx.y; i < nd; i += blockDim.y) {
;         double sum = 0.0;
;         for (int p = 0; p < nq; ++p)
;           sum += QR.data[p + QR.sizes[0] * e] * GR.data[p + GR.sizes[0] * i];
;         Y.data[i + Y.sizes[0] * e] += sum;
;       }
;     };
;     CuKernel2D<<<ne, dim3(1, nd, 1)>>>(ne, body);
;   }
; The descriptors name pointers and sizes as fields of the lambda, which only
; the launch's own argument array holds in every host ABI. The profiling launch
; passes the runtime that array and each pointer's offset; the kernel's
; device-side name carries the -fgpu-rdc postfix of an internal kernel. The
; dispatch reads its operands and extents from the same array and runs only
; when the extents the recognition assumed equal are equal and every extent
; fits the dispatch's int parameters; otherwise it launches the kernel as
; written.

; GEN: [poseidon] launch of _ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd now allocates and seeds 3 shadow buffer(s)
; GEN: @poseidon.ptr._ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd = private unnamed_addr constant [3 x i32] [i32 1, i32 1, i32 1]
; GEN-NEXT: @poseidon.off._ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd = private unnamed_addr constant [3 x i32] [i32 16, i32 40, i32 64]
; GEN-NEXT: @poseidon.seed._ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd = private unnamed_addr constant [3 x i32] [i32 0, i32 0, i32 8]
; GEN-LABEL: define void @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(
; GEN: %[[ARGS:[0-9]+]] = alloca ptr, i64 2
; GEN: call i32 @__cudaPopCallConfiguration(ptr %[[GRID:[0-9]+]], ptr %[[BLOCK:[0-9]+]], ptr %[[SHMEMP:[0-9]+]], ptr %[[STREAMP:[0-9]+]])
; GEN-NEXT: %[[SHMEM:[0-9]+]] = load i64, ptr %[[SHMEMP]], align 8
; GEN-NEXT: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; GEN: call void @__poseidon_launch_profiled(ptr @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_, i32 0, ptr %[[GRID]], ptr %[[BLOCK]], i64 %[[SHMEM]], ptr %[[STREAM]], i32 2, ptr %[[ARGS]], i32 3, ptr @poseidon.ptr._ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd, ptr @poseidon.off._ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd, ptr @poseidon.seed._ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd)
; GEN-NOT: call {{.*}}@cudaLaunchKernel(

; USE: [poseidon] replaced the launch in stub _ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_ with __poseidon_ozakiii_dgemm (C=arg1[+64] A=arg1[+40] B=arg1[+16]), if arg1[+4] == arg1[+24], if arg1[+4] == arg1[+48]
; USE-LABEL: define void @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_(
; USE: %[[ARGS:[0-9]+]] = alloca ptr, i64 2
; USE: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[SHMEMP:[0-9]+]], ptr %[[STREAMP:[0-9]+]])
; USE-NEXT: %[[SHMEM:[0-9]+]] = load i64, ptr %[[SHMEMP]], align 8
; USE-NEXT: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; USE: %[[S:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[L:[0-9]+]] = load ptr, ptr %[[S]], align 8
; USE-NEXT: %[[F:[0-9]+]] = getelementptr inbounds i8, ptr %[[L]], i64 64{{$}}
; USE-NEXT: %[[C:[0-9]+]] = load ptr, ptr %[[F]], align 8
; USE-NEXT: %[[AS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[AL:[0-9]+]] = load ptr, ptr %[[AS]], align 8
; USE-NEXT: %[[AF:[0-9]+]] = getelementptr inbounds i8, ptr %[[AL]], i64 40{{$}}
; USE-NEXT: %[[A:[0-9]+]] = load ptr, ptr %[[AF]], align 8
; USE-NEXT: %[[BS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[BL:[0-9]+]] = load ptr, ptr %[[BS]], align 8
; USE-NEXT: %[[BF:[0-9]+]] = getelementptr inbounds i8, ptr %[[BL]], i64 16{{$}}
; USE-NEXT: %[[B:[0-9]+]] = load ptr, ptr %[[BF]], align 8
; USE-NEXT: %[[MS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[ML:[0-9]+]] = load ptr, ptr %[[MS]], align 8
; USE-NEXT: %[[M32:[0-9]+]] = load i32, ptr %[[ML]], align 4
; USE-NEXT: %[[M64:[0-9]+]] = sext i32 %[[M32]] to i64
; USE-NEXT: %[[MOK:[0-9]+]] = icmp ule i64 %[[M64]], 2147483647
; USE-NEXT: %[[M:[0-9]+]] = trunc i64 %[[M64]] to i32
; USE-NEXT: %[[NS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 0|i64 0}}{{$}}
; USE-NEXT: %[[NL:[0-9]+]] = load ptr, ptr %[[NS]], align 8
; USE-NEXT: %[[N32:[0-9]+]] = load i32, ptr %[[NL]], align 4
; USE-NEXT: %[[N64:[0-9]+]] = sext i32 %[[N32]] to i64
; USE-NEXT: %[[NOK:[0-9]+]] = icmp ule i64 %[[N64]], 2147483647
; USE-NEXT: %[[R1:[0-9]+]] = and i1 %[[MOK]], %[[NOK]]
; USE-NEXT: %[[N:[0-9]+]] = trunc i64 %[[N64]] to i32
; USE-NEXT: %[[KS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[KL:[0-9]+]] = load ptr, ptr %[[KS]], align 8
; USE-NEXT: %[[KF:[0-9]+]] = getelementptr inbounds i8, ptr %[[KL]], i64 4{{$}}
; USE-NEXT: %[[K4:[0-9]+]] = load i32, ptr %[[KF]], align 4
; USE-NEXT: %[[K464:[0-9]+]] = sext i32 %[[K4]] to i64
; USE-NEXT: %[[K4OK:[0-9]+]] = icmp ule i64 %[[K464]], 2147483647
; USE-NEXT: %[[R2:[0-9]+]] = and i1 %[[R1]], %[[K4OK]]
; USE-NEXT: %[[K64:[0-9]+]] = mul i64 %[[K464]], 3
; USE-NEXT: %[[KOK:[0-9]+]] = icmp ule i64 %[[K64]], 2147483647
; USE-NEXT: %[[R3:[0-9]+]] = and i1 %[[R2]], %[[KOK]]
; USE-NEXT: %[[K:[0-9]+]] = trunc i64 %[[K64]] to i32
; USE-NEXT: %[[LDAS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[LDAL:[0-9]+]] = load ptr, ptr %[[LDAS]], align 8
; USE-NEXT: %[[LDAF:[0-9]+]] = getelementptr inbounds i8, ptr %[[LDAL]], i64 48{{$}}
; USE-NEXT: %[[LDA32:[0-9]+]] = load i32, ptr %[[LDAF]], align 4
; USE-NEXT: %[[LDA64:[0-9]+]] = sext i32 %[[LDA32]] to i64
; USE-NEXT: %[[LDAOK:[0-9]+]] = icmp ule i64 %[[LDA64]], 2147483647
; USE-NEXT: %[[R4:[0-9]+]] = and i1 %[[R3]], %[[LDAOK]]
; USE-NEXT: %[[LDA:[0-9]+]] = trunc i64 %[[LDA64]] to i32
; USE-NEXT: %[[LDBS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[LDBL:[0-9]+]] = load ptr, ptr %[[LDBS]], align 8
; USE-NEXT: %[[LDBF:[0-9]+]] = getelementptr inbounds i8, ptr %[[LDBL]], i64 24{{$}}
; USE-NEXT: %[[LDB32:[0-9]+]] = load i32, ptr %[[LDBF]], align 4
; USE-NEXT: %[[LDB64:[0-9]+]] = sext i32 %[[LDB32]] to i64
; USE-NEXT: %[[LDBOK:[0-9]+]] = icmp ule i64 %[[LDB64]], 2147483647
; USE-NEXT: %[[R5:[0-9]+]] = and i1 %[[R4]], %[[LDBOK]]
; USE-NEXT: %[[LDB:[0-9]+]] = trunc i64 %[[LDB64]] to i32
; USE-NEXT: %[[LDCS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[LDCL:[0-9]+]] = load ptr, ptr %[[LDCS]], align 8
; USE-NEXT: %[[LDCF:[0-9]+]] = getelementptr inbounds i8, ptr %[[LDCL]], i64 72{{$}}
; USE-NEXT: %[[LDC32:[0-9]+]] = load i32, ptr %[[LDCF]], align 4
; USE-NEXT: %[[LDC64:[0-9]+]] = sext i32 %[[LDC32]] to i64
; USE-NEXT: %[[LDCOK:[0-9]+]] = icmp ule i64 %[[LDC64]], 2147483647
; USE-NEXT: %[[R6:[0-9]+]] = and i1 %[[R5]], %[[LDCOK]]
; USE-NEXT: %[[LDC:[0-9]+]] = trunc i64 %[[LDC64]] to i32
; Each equality loads its two operands in the order the descriptor names
; them, whichever compiler built the plugin.
; USE-NEXT: %[[X0S:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[X0L:[0-9]+]] = load ptr, ptr %[[X0S]], align 8
; USE-NEXT: %[[X0F:[0-9]+]] = getelementptr inbounds i8, ptr %[[X0L]], i64 4{{$}}
; USE-NEXT: %[[X0:[0-9]+]] = load i32, ptr %[[X0F]], align 4
; USE-NEXT: %[[Y0S:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[Y0L:[0-9]+]] = load ptr, ptr %[[Y0S]], align 8
; USE-NEXT: %[[Y0F:[0-9]+]] = getelementptr inbounds i8, ptr %[[Y0L]], i64 24{{$}}
; USE-NEXT: %[[Y0:[0-9]+]] = load i32, ptr %[[Y0F]], align 4
; USE-NEXT: %[[EQ0:[0-9]+]] = icmp eq i32 %[[X0]], %[[Y0]]
; USE-NEXT: %[[X1S:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[X1L:[0-9]+]] = load ptr, ptr %[[X1S]], align 8
; USE-NEXT: %[[X1F:[0-9]+]] = getelementptr inbounds i8, ptr %[[X1L]], i64 4{{$}}
; USE-NEXT: %[[X1:[0-9]+]] = load i32, ptr %[[X1F]], align 4
; USE-NEXT: %[[Y1S:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; USE-NEXT: %[[Y1L:[0-9]+]] = load ptr, ptr %[[Y1S]], align 8
; USE-NEXT: %[[Y1F:[0-9]+]] = getelementptr inbounds i8, ptr %[[Y1L]], i64 48{{$}}
; USE-NEXT: %[[Y1:[0-9]+]] = load i32, ptr %[[Y1F]], align 4
; USE-NEXT: %[[EQ1:[0-9]+]] = icmp eq i32 %[[X1]], %[[Y1]]
; USE-NEXT: %[[EQ:[0-9]+]] = and i1 %[[EQ0]], %[[EQ1]]
; USE-NEXT: %[[OK:[0-9]+]] = and i1 %[[EQ]], %[[R6]]
; USE-NEXT: br i1 %[[OK]], label %[[DISPATCH:[0-9]+]], label %[[LAUNCH:[0-9]+]]
; USE: [[DISPATCH]]:
; USE-NEXT: call void @__poseidon_ozakiii_dgemm(ptr %[[C]], ptr %[[A]], ptr %[[B]], i32 %[[M]], i32 %[[N]], i32 %[[K]], i32 %[[LDA]], i32 %[[LDB]], i32 %[[LDC]], i32 0, i32 1, i32 1, double 1.000000e+00, double 1.000000e+00, ptr %[[STREAM]], i32 13)
; USE-NEXT: br label %[[JOIN:[0-9]+]]
; USE: [[LAUNCH]]:
; USE-NEXT: %{{[0-9]+}} = call noundef i32 @cudaLaunchKernel(ptr noundef @_ZL25__device_stub__CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT_, {{.*}}, ptr noundef %[[ARGS]], i64 noundef %[[SHMEM]], ptr noundef %[[STREAM]])
; USE-NEXT: br label %[[JOIN]]
; USE: [[JOIN]]:
; USE: ret void

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

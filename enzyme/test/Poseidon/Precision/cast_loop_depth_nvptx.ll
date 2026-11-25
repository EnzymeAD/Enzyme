; RUN: rm -rf %t && mkdir %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; RUN:   -poseidon-profile-use=%S/../Inputs/cast_loop_depth -poseidon-cache=%t/c \
; RUN:   -poseidon-enable-herbie=false -poseidon-enable-pt=true \
; RUN:   -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_gpu_synthetic.csv \
; RUN:   -poseidon-print -S -o /dev/null 2>&1 | FileCheck %s

; A precision candidate's conversion is priced as running once per thread
; only when it is in no loop. Narrowed to FP32, a[i*n+j] is converted in the
; j loop, which runs 8 times per thread, and b[j*m+k] in the k loop; both count
; 64 as the static walk counts the multiply-add once. Only the widening of the
; result after both loops is scaled to once per thread (64 * 64 / 8192). A
; conversion at any depth shallower than the deepest loop of the candidate
; was scaled alike, here the conversion of a[i*n+j] by 1/128.
; The IR is the kernel
;   POSEIDON_OPTIMIZE __global__ void cast_depth(int n, int m, const double *a,
;                                                const double *b, double *out) {
;     const int i = blockIdx.x * blockDim.x + threadIdx.x;
;     double acc = 0.0;
;     int j = 0;
;     do {
;       const double s = a[i * n + j] * a[i * n + j];
;       int k = 0;
;       do
;         acc += s * b[j * m + k];
;       while (++k < m);
;     } while (++j < n);
;     out[i] = acc;
;   }
; via clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -O2
; -ffp-contract=on -fno-unroll-loops -fno-vectorize -fno-slp-vectorize
; -I<enzyme>/include -S -emit-llvm, then llvm-extract of the kernel and its
; annotation. Profile: the same source through poseidon-clang++
; -poseidon-profile-generate, launched once as cast_depth<<<1, 64>>>(8, 16, a,
; b, o) with a[i] = 0.5 + 0.001 * (i % 17) and b[i] = 1.0 + 0.01 * (i % 7).

; CHECK: CS candidate 'All FP64(0%) + FP32(100%)': initial=1.024000e+03 cand=1.305000e+02

source_filename = "cast_depth.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = hidden unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = hidden unnamed_addr constant [14 x i8] c"cast_depth.cu\00", section "llvm.metadata"
@.args = hidden unnamed_addr constant { i32 } zeroinitializer, section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z10cast_depthiiPKdS0_Pd, ptr @.str, ptr @.str1, i32 8, ptr @.args }], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local ptx_kernel void @_Z10cast_depthiiPKdS0_Pd(i32 noundef %0, i32 noundef %1, ptr nofree noundef readonly captures(none) %2, ptr nofree noundef readonly captures(none) %3, ptr nofree noundef writeonly captures(none) %4) #0 {
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %8 = mul i32 %6, %7
  %9 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %10 = add i32 %8, %9
  %11 = mul nsw i32 %10, %0
  %12 = tail call i32 @llvm.smax.i32(i32 %1, i32 1)
  %13 = tail call i32 @llvm.smax.i32(i32 %0, i32 1)
  br label %14

14:                                               ; preds = %33, %5
  %15 = phi double [ 0.000000e+00, %5 ], [ %30, %33 ]
  %16 = phi i32 [ 0, %5 ], [ %34, %33 ]
  %17 = add nsw i32 %16, %11
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds [8 x i8], ptr %2, i64 %18
  %20 = load double, ptr %19, align 8, !tbaa !11
  %21 = fmul double %20, %20
  %22 = mul nsw i32 %16, %1
  br label %23

23:                                               ; preds = %23, %14
  %24 = phi double [ %15, %14 ], [ %30, %23 ]
  %25 = phi i32 [ 0, %14 ], [ %31, %23 ]
  %26 = add nsw i32 %25, %22
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds [8 x i8], ptr %3, i64 %27
  %29 = load double, ptr %28, align 8, !tbaa !11
  %30 = tail call double @llvm.fmuladd.f64(double %21, double %29, double %24)
  %31 = add nuw nsw i32 %25, 1
  %32 = icmp eq i32 %31, %12
  br i1 %32, label %33, label %23, !llvm.loop !13

33:                                               ; preds = %23
  %34 = add nuw nsw i32 %16, 1
  %35 = icmp eq i32 %34, %13
  br i1 %35, label %36, label %14, !llvm.loop !16

36:                                               ; preds = %33
  %37 = sext i32 %10 to i64
  %38 = getelementptr inbounds [8 x i8], ptr %4, i64 %37
  store double %30, ptr %38, align 8, !tbaa !11
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 2, !"debugger-tuning", !"gdb"}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !9, i64 0}
!13 = distinct !{!13, !14, !15}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.unroll.disable"}
!16 = distinct !{!16, !14, !15}

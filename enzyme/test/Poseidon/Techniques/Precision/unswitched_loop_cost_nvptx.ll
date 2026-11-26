; REQUIRES: nvptx-registered-target
; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/Inputs/unswitched_loop -poseidon-early-prune \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-precision-expansions=true -poseidon-tau=1e-6 \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cache=%t/c -o %t/out.ll 2> %t/log
; RUN: FileCheck %s < %t/log
; RUN: FileCheck --check-prefix=IR %s < %t/out.ll
; RUN: %{opt} -poseidon-cache=%t/k -unswitch-threshold=0 -o /dev/null 2> %t/kept.log
; RUN: grep '^\[poseidon\] precision candidate' %t/log | sed 's/ [^ ]*acc=.*//' > %t/priced
; RUN: grep '^\[poseidon\] precision candidate' %t/kept.log | sed 's/ [^ ]*acc=.*//' > %t/priced-kept
; RUN: diff %t/priced %t/priced-kept
; RUN: %{opt} -poseidon-cache=%t/f -enable-nontrivial-unswitch -o /dev/null 2> %t/forced.log
; RUN: grep '^\[poseidon\] precision candidate' %t/forced.log | sed 's/ [^ ]*acc=.*//' > %t/priced-forced
; RUN: diff %t/priced %t/priced-forced
; RUN: %opt %s -passes='default<O3>' -S \
; RUN:   | FileCheck --check-prefix=KEPT --implicit-check-not=.us %s

; CHECK-NOT: cand=0.000000e+00
; CHECK: [poseidon] precision candidate 'All FP64(0%) + FP32(100%)': initial=6.658000e+03 cand=3.980000e+02 exec=1024
; CHECK-NOT: cand=0.000000e+00
; CHECK: [poseidon] precision candidate 'All FP64(0%) + Expansion2(100%)': initial=6.658000e+03 cand={{[1-9]}}
; CHECK-NOT: cand=0.000000e+00
; CHECK: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; IR-LABEL: define {{.*}} @_Z12setup_kerneliPKdS0_Pd(
; IR: call void @preprocess__Z12setup_kerneliPKdS0_Pd_poseidon_body(i32 %0, ptr %1, ptr %2, ptr %3)
; IR-LABEL: define internal void @preprocess__Z12setup_kerneliPKdS0_Pd_poseidon_body(
; IR: [[TIDX:%[0-9]+]] = tail call {{.*}}i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; IR: [[GUARD:%[0-9]+]] = icmp {{.*}}ult i32 [[TIDX]], %0
; IR: br i1 [[GUARD]], label %[[US:[^ ,]+]], label %[[EXIT:[^ ,]+]]
; IR-EMPTY:
; IR-NEXT: {{^}}[[US]]:
; IR-NOT: [[GUARD]]{{[^0-9]}}
; IR: [[DET:%[0-9]+]] = {{.*}}call float @llvm.fmuladd.f32(
; IR: [[Q:%[0-9]+]] = fdiv float %{{[0-9]+}}, [[DET]]
; IR-NEXT: fpext float [[Q]] to double
; IR-NOT: [[GUARD]]{{[^0-9]}}
; IR: {{^}}[[EXIT]]:
; IR-NEXT: ret void
; KEPT-LABEL: define {{.*}} @_Z12setup_kerneliPKdS0_Pd(
; KEPT: [[TIDX:%[0-9]+]] = tail call {{.*}}i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; KEPT: [[GUARD:%[0-9]+]] = icmp {{.*}}ult i32 [[TIDX]], %0
; KEPT: br label %[[OUTER:[0-9]+]]
; KEPT: {{^}}[[OUTER]]:
; KEPT-NEXT: phi i32
; KEPT-NEXT: br i1 [[GUARD]], label %[[INNER:[0-9]+]], label
; KEPT: fdiv double

; A thread-strided loop nest whose inner guard (threadIdx.x < Q) is invariant
; in the outer loop: the emitted site is unswitched by the target-free O3
; cleanup (IR), but each candidate is priced on a clone the compile's own
; pipeline, with divergent branches, does not unswitch (KEPT); forcing or
; disabling unswitching leaves every price unchanged, and FP32 is applied.
; Module of unswitched_loop.cu (setup_kernel) via clang -x cuda
; --cuda-device-only --cuda-gpu-arch=sm_120 -O2 -ffp-contract=on; profile from
; setup_kernel<<<64, dim3(4, 4)>>>(4, ...) on an sm_120 GPU.

source_filename = "unswitched_loop/unswitched_loop.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [35 x i8] c"unswitched_loop/unswitched_loop.cu\00", section "llvm.metadata"
@.args = private unnamed_addr constant { i32 } zeroinitializer, section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z12setup_kerneliPKdS0_Pd, ptr @.str, ptr @.str1, i32 8, ptr @.args }], section "llvm.metadata"

define dso_local ptx_kernel void @_Z12setup_kerneliPKdS0_Pd(i32 noundef %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef readonly captures(none) %2, ptr nofree noundef writeonly captures(none) %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = icmp slt i32 %5, %0
  br i1 %6, label %7, label %15

7:                                                ; preds = %4
  %8 = mul i32 %0, %0
  %9 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %11 = mul i32 %8, %9
  %12 = icmp samesign ult i32 %10, %0
  %13 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %14 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  br label %16

15:                                               ; preds = %20, %4
  ret void

16:                                               ; preds = %20, %7
  %17 = phi i32 [ %5, %7 ], [ %21, %20 ]
  br i1 %12, label %18, label %20

18:                                               ; preds = %16
  %19 = mul nsw i32 %17, %0
  br label %23

20:                                               ; preds = %23, %16
  %21 = add i32 %17, %14
  %22 = icmp slt i32 %21, %0
  br i1 %22, label %16, label %15, !llvm.loop !11

23:                                               ; preds = %23, %18
  %24 = phi i32 [ %10, %18 ], [ %49, %23 ]
  %25 = add nsw i32 %24, %19
  %26 = add nsw i32 %25, %11
  %27 = shl nsw i32 %26, 2
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds [8 x i8], ptr %2, i64 %28
  %30 = load double, ptr %29, align 8, !tbaa !13
  %31 = getelementptr inbounds nuw i8, ptr %29, i64 24
  %32 = load double, ptr %31, align 8, !tbaa !13
  %33 = getelementptr inbounds nuw i8, ptr %29, i64 8
  %34 = load double, ptr %33, align 8, !tbaa !13
  %35 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %36 = load double, ptr %35, align 8, !tbaa !13
  %37 = fneg double %36
  %38 = fmul double %34, %37
  %39 = tail call double @llvm.fmuladd.f64(double %30, double %32, double %38)
  %40 = sext i32 %25 to i64
  %41 = getelementptr inbounds [8 x i8], ptr %1, i64 %40
  %42 = load double, ptr %41, align 8, !tbaa !13
  %43 = fmul double %34, %34
  %44 = tail call double @llvm.fmuladd.f64(double %30, double %30, double %43)
  %45 = fmul double %42, %44
  %46 = fdiv double %45, %39
  %47 = sext i32 %26 to i64
  %48 = getelementptr inbounds [8 x i8], ptr %3, i64 %47
  store double %46, ptr %48, align 8, !tbaa !13
  %49 = add i32 %24, %13
  %50 = icmp slt i32 %49, %0
  br i1 %50, label %23, label %20, !llvm.loop !15
}

declare double @llvm.fmuladd.f64(double, double, double) #1
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2
attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!14, !14, i64 0}
!14 = !{!"double", !9, i64 0}
!15 = distinct !{!15, !12}

; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: for f in ptr index gather; do cp %S/../../Inputs/matmul_site_profile/matmul_body.fpprofile %t/prof/${f}_body.fpprofile; done
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; RUN:   -poseidon-profile-use=%t/prof -poseidon-cache=%t/c \
; RUN:   -poseidon-enable-rewrite=false -poseidon-enable-precision=false \
; RUN:   -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-tau=1e-6 \
; RUN:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; RUN:   -poseidon-print -S -o %t/out.ll 2>&1 | FileCheck %s
; RUN: FileCheck --check-prefix=IR %s < %t/out.ll

; Two guarded products as clang emits them, from
;   extern "C" __device__ __attribute__((noinline)) void
;   ptr_body(const double *A, const double *B, double *C) {
;     int row = blockIdx.y * blockDim.y + threadIdx.y;
;     int col = blockIdx.x * blockDim.x + threadIdx.x;
;     if (row >= 40 || col >= 32)
;       return;
;     const double *arow = A + (size_t)row * 16;
;     const double *bcol = B + col;
;     double acc = 0.0;
;   #pragma unroll 1
;     for (int k = 0; k < 16; ++k)
;       acc += arow[k] * bcol[(size_t)k * 32];
;     C[(size_t)row * 32 + col] = acc;
;   }
; index_body, the same product indexed as A[row * 16 + k] * B[k * 32 + col]
; and stored to C[row * 32 + col], and gather_body, which reads
; B[perm[k] * 32 + col] for a perm parameter, each called from a kernel
; through __poseidon_fp_optimize, via clang -x cuda --cuda-device-only
; --cuda-gpu-arch=sm_120 -O2 -ffp-contract=on -S -emit-llvm. Each raises in a
; 16 x 16 block whose rows are all below 40 and columns below 32, which for
; that shape is blockIdx.y < 2 and blockIdx.x < 2. index_body's B address
; sign-extends the nsw sum of the loop counter's multiple and the column, which
; ScalarEvolution does not see as an affine function of the counter; the
; recognizer distributes the extension, since a wrap would make the loaded
; address poison. gather_body's B address depends on a load in the loop and is
; not a product.

; CHECK: Found 1 matrix product(s) in preprocess_ptr_body
; CHECK: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; CHECK: Found 1 matrix product(s) in preprocess_index_body
; CHECK: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; CHECK: [poseidon] reject loop in preprocess_gather_body: an operand's address is not an affine function of the loop's counter
; CHECK-NOT: Applying solution for matmul

; IR-LABEL: define {{.*}} @preprocess_ptr_body(
; IR: [[S1:%.+]] = icmp eq i32 %{{.+}}, 16
; IR-NEXT: [[S2:%.+]] = icmp eq i32 %{{.+}}, 16
; IR-NEXT: [[SXY:%.+]] = and i1 [[S1]], [[S2]]
; IR-NEXT: [[NZ:%.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; IR-NEXT: [[SZ:%.+]] = icmp eq i32 [[NZ]], 1
; IR-NEXT: [[SHAPE:%.+]] = and i1 [[SZ]], [[SXY]]
; IR-NEXT: [[BXY:%.+]] = or i32 %{{.+}}, %{{.+}}
; IR-NEXT: [[IN:%.+]] = icmp samesign ult i32 [[BXY]], 2
; IR-NEXT: [[RAISE:%.+]] = and i1 [[IN]], [[SHAPE]]
; IR-NEXT: br i1 [[RAISE]], label %raise.entry, label %{{.+}}
; IR: %raise.result = phi double

; IR-LABEL: define {{.*}} @preprocess_index_body(
; IR: [[S1:%.+]] = icmp eq i32 %{{.+}}, 16
; IR-NEXT: [[S2:%.+]] = icmp eq i32 %{{.+}}, 16
; IR-NEXT: [[SXY:%.+]] = and i1 [[S1]], [[S2]]
; IR-NEXT: [[NZ:%.+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; IR-NEXT: [[SZ:%.+]] = icmp eq i32 [[NZ]], 1
; IR-NEXT: [[SHAPE:%.+]] = and i1 [[SZ]], [[SXY]]
; IR-NEXT: [[BXY:%.+]] = or i32 %{{.+}}, %{{.+}}
; IR-NEXT: [[IN:%.+]] = icmp samesign ult i32 [[BXY]], 2
; IR-NEXT: [[RAISE:%.+]] = and i1 [[IN]], [[SHAPE]]
; IR-NEXT: br i1 [[RAISE]], label %raise.entry, label %{{.+}}
; IR: %raise.result = phi double
; B, read through the column less the block's thread index, then by the
; column the fill gives each thread.
; IR: [[COL8:%.+]] = shl nuw nsw i32 %13, 3
; IR-NEXT: [[COL8W:%.+]] = zext nneg i32 [[COL8]] to i64
; IR-NEXT: [[BCOL:%.+]] = getelementptr i8, ptr %1, i64 [[COL8W]]
; IR-NEXT: [[TX8:%.+]] = shl nuw nsw i32 %12, 3
; IR-NEXT: [[TX8W:%.+]] = zext nneg i32 [[TX8]] to i64
; IR-NEXT: [[BACK:%.+]] = sub nsw i64 0, [[TX8W]]
; IR-NEXT: {{%.+}} = getelementptr i8, ptr [[BCOL]], i64 [[BACK]]

source_filename = "real.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @ptr_body(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %6 = mul nuw nsw i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %8 = add nuw nsw i32 %6, %7
  %9 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %11 = mul i32 %9, %10
  %12 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %13 = add i32 %11, %12
  %14 = icmp samesign ugt i32 %8, 39
  %15 = icmp sgt i32 %13, 31
  %16 = or i1 %14, %15
  br i1 %16, label %39, label %17

17:                                               ; preds = %3
  %18 = zext nneg i32 %8 to i64
  %19 = shl nuw nsw i64 %18, 7
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 %19
  %21 = sext i32 %13 to i64
  %22 = getelementptr inbounds [8 x i8], ptr %1, i64 %21
  br label %27

23:                                               ; preds = %27
  %24 = shl nuw nsw i64 %18, 8
  %25 = getelementptr i8, ptr %2, i64 %24
  %26 = getelementptr [8 x i8], ptr %25, i64 %21
  store double %36, ptr %26, align 8, !tbaa !11
  br label %39

27:                                               ; preds = %17, %27
  %28 = phi i32 [ 0, %17 ], [ %37, %27 ]
  %29 = phi double [ 0.000000e+00, %17 ], [ %36, %27 ]
  %30 = zext nneg i32 %28 to i64
  %31 = getelementptr inbounds nuw [8 x i8], ptr %20, i64 %30
  %32 = load double, ptr %31, align 8, !tbaa !11
  %33 = shl nuw nsw i64 %30, 8
  %34 = getelementptr inbounds nuw i8, ptr %22, i64 %33
  %35 = load double, ptr %34, align 8, !tbaa !11
  %36 = tail call double @llvm.fmuladd.f64(double %32, double %35, double %29)
  %37 = add nuw nsw i32 %28, 1
  %38 = icmp eq i32 %37, 16
  br i1 %38, label %23, label %27, !llvm.loop !13

39:                                               ; preds = %3, %23
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @index_body(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %6 = mul nuw nsw i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %8 = add nuw nsw i32 %6, %7
  %9 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %11 = mul i32 %9, %10
  %12 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %13 = add i32 %11, %12
  %14 = icmp samesign ugt i32 %8, 39
  %15 = icmp sgt i32 %13, 31
  %16 = or i1 %14, %15
  br i1 %16, label %39, label %17

17:                                               ; preds = %3
  %18 = shl nuw nsw i32 %8, 4
  br label %24

19:                                               ; preds = %24
  %20 = shl nuw nsw i32 %8, 5
  %21 = add nsw i32 %20, %13
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds [8 x i8], ptr %2, i64 %22
  store double %36, ptr %23, align 8, !tbaa !11
  br label %39

24:                                               ; preds = %17, %24
  %25 = phi i32 [ 0, %17 ], [ %37, %24 ]
  %26 = phi double [ 0.000000e+00, %17 ], [ %36, %24 ]
  %27 = add nuw nsw i32 %25, %18
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %28
  %30 = load double, ptr %29, align 8, !tbaa !11
  %31 = shl nuw nsw i32 %25, 5
  %32 = add nsw i32 %31, %13
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds [8 x i8], ptr %1, i64 %33
  %35 = load double, ptr %34, align 8, !tbaa !11
  %36 = tail call double @llvm.fmuladd.f64(double %30, double %35, double %26)
  %37 = add nuw nsw i32 %25, 1
  %38 = icmp eq i32 %37, 16
  br i1 %38, label %19, label %24, !llvm.loop !16

39:                                               ; preds = %3, %19
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @gather_body(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2, ptr nofree noundef readonly captures(none) %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %7 = mul nuw nsw i32 %5, %6
  %8 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %9 = add nuw nsw i32 %7, %8
  %10 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %11 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %12 = mul i32 %10, %11
  %13 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %14 = add i32 %12, %13
  %15 = icmp samesign ugt i32 %9, 39
  %16 = icmp sgt i32 %14, 31
  %17 = or i1 %15, %16
  br i1 %17, label %43, label %18

18:                                               ; preds = %4
  %19 = shl nuw nsw i32 %9, 4
  br label %25

20:                                               ; preds = %25
  %21 = shl nuw nsw i32 %9, 5
  %22 = add nsw i32 %21, %14
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds [8 x i8], ptr %2, i64 %23
  store double %40, ptr %24, align 8, !tbaa !11
  br label %43

25:                                               ; preds = %18, %25
  %26 = phi i32 [ 0, %18 ], [ %41, %25 ]
  %27 = phi double [ 0.000000e+00, %18 ], [ %40, %25 ]
  %28 = add nuw nsw i32 %26, %19
  %29 = zext nneg i32 %28 to i64
  %30 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %29
  %31 = load double, ptr %30, align 8, !tbaa !11
  %32 = zext nneg i32 %26 to i64
  %33 = getelementptr inbounds nuw [4 x i8], ptr %3, i64 %32
  %34 = load i32, ptr %33, align 4, !tbaa !17
  %35 = shl nsw i32 %34, 5
  %36 = add nsw i32 %35, %14
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds [8 x i8], ptr %1, i64 %37
  %39 = load double, ptr %38, align 8, !tbaa !11
  %40 = tail call double @llvm.fmuladd.f64(double %31, double %39, double %27)
  %41 = add nuw nsw i32 %26, 1
  %42 = icmp eq i32 %41, 16
  br i1 %42, label %20, label %25, !llvm.loop !18

43:                                               ; preds = %4, %20
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z10ptr_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !17
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !17
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @ptr_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z12index_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !17
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !17
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @index_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z13gather_kernelPKdS0_PdS1_PKi(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4) local_unnamed_addr #2 {
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4, !tbaa !17
  %7 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4, !tbaa !17
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iPKiEET_PvDpT0_(ptr noundef nonnull @gather_body, i32 noundef %6, ptr noundef %0, i32 noundef %6, ptr noundef %1, i32 noundef %7, ptr noundef %2, ptr noundef %3, i32 noundef %6, ptr noundef %4) #5
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iPKiEET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4, !5}
!llvm.errno.tbaa = !{!6}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 2, !"debugger-tuning", !"gdb"}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 24.0.0git (https://github.com/llvm/llvm-project.git 8823cdce1b565e01cb47f89c30f51a0e0ec32d6a)"}
!5 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"__libc_errno", !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !9, i64 0}
!13 = distinct !{!13, !14, !15}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.unroll.disable"}
!16 = distinct !{!16, !14, !15}
!17 = !{!8, !8, i64 0}
!18 = distinct !{!18, !14, !15}

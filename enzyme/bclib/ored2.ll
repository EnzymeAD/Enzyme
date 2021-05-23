; ModuleID = 'red2.ll'
source_filename = "text"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@0 = internal unnamed_addr addrspace(3) global [32 x float] zeroinitializer, align 32

define float @julia_reduce_block_2997(float %0) {
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !2
  %3 = add nuw nsw i32 %2, 1
  %4 = zext i32 %3 to i64
  %5 = lshr i64 %4, 5
  %6 = and i64 %4, 2016
  %7 = icmp ne i64 %6, %4
  %8 = zext i1 %7 to i64
  %9 = add nuw nsw i64 %5, %8
  %10 = and i64 %4, 31
  %11 = icmp eq i64 %10, 0
  %12 = select i1 %11, i64 32, i64 %10
  %13 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %0, i32 1, i32 31)
  %14 = fadd fast float %13, %0
  %15 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %14, i32 2, i32 31)
  %16 = fadd fast float %15, %14
  %17 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %16, i32 4, i32 31)
  %18 = fadd fast float %17, %16
  %19 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %18, i32 8, i32 31)
  %20 = fadd fast float %19, %18
  %21 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %20, i32 16, i32 31)
  %22 = icmp eq i64 %12, 1
  br i1 %22, label %23, label %27

23:                                               ; preds = %1
  %24 = fadd fast float %21, %20
  %25 = add nsw i64 %9, -1
  %26 = getelementptr inbounds [32 x float], [32 x float] addrspace(3)* @0, i64 0, i64 %25
  store float %24, float addrspace(3)* %26, align 4, !tbaa !3
  br label %27

27:                                               ; preds = %23, %1
  call void @llvm.nvvm.barrier0()
  %28 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !range !6
  %29 = zext i32 %28 to i64
  %30 = lshr i64 %29, 5
  %31 = and i64 %29, 992
  %32 = icmp ne i64 %31, %29
  %33 = zext i1 %32 to i64
  %34 = add nuw nsw i64 %30, %33
  %35 = icmp ult i64 %34, %4
  br i1 %35, label %40, label %36

36:                                               ; preds = %27
  %37 = add nsw i64 %12, -1
  %38 = getelementptr inbounds [32 x float], [32 x float] addrspace(3)* @0, i64 0, i64 %37
  %39 = load float, float addrspace(3)* %38, align 4, !tbaa !3
  br label %40

40:                                               ; preds = %36, %27
  %41 = phi float [ %39, %36 ], [ 0.000000e+00, %27 ]
  %42 = icmp eq i64 %9, 1
  br i1 %42, label %43, label %54

43:                                               ; preds = %40
  %44 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %41, i32 1, i32 31)
  %45 = fadd fast float %44, %41
  %46 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %45, i32 2, i32 31)
  %47 = fadd fast float %46, %45
  %48 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %47, i32 4, i32 31)
  %49 = fadd fast float %48, %47
  %50 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %49, i32 8, i32 31)
  %51 = fadd fast float %50, %49
  %52 = call float @llvm.nvvm.shfl.sync.down.f32(i32 -1, float %51, i32 16, i32 31)
  %53 = fadd fast float %52, %51
  br label %54

54:                                               ; preds = %43, %40
  %55 = phi float [ %41, %40 ], [ %53, %43 ]
  ret float %55
}

; Function Attrs: convergent inaccessiblememonly nounwind
declare float @llvm.nvvm.shfl.sync.down.f32(i32, float, i32, i32) #1

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { convergent inaccessiblememonly nounwind }
attributes #2 = { convergent nounwind }
attributes #3 = { nofree nosync nounwind willreturn }

!2 = !{i32 0, i32 1023}
!3 = !{!4, !4, i64 0, i64 0}
!4 = !{!"custom_tbaa_addrspace(3)", !5, i64 0}
!5 = !{!"custom_tbaa"}
!6 = !{i32 1, i32 1024}

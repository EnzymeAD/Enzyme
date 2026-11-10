; The product of matmul_device.ll written as an annotated kernel: the
; reduction loop is the kernel's own body and POSEIDON_OPTIMIZE is the
; "poseidon" entry of llvm.global.annotations. Same kernel name and launch
; arguments as the stub in Inputs/ozaki_dispatch/ozaki_stub_host.ll.

; ModuleID = 'tcec_matmul.cu'
source_filename = "tcec_matmul.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [15 x i8] c"tcec_matmul.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z13matmul_kernelPKdS0_PdS1_, ptr @.str, ptr @.str.1, i32 1, ptr null }], section "llvm.metadata"

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2, ptr noundef readnone captures(none) %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %7 = shl nuw nsw i32 %6, 4
  br label %12

8:
  %9 = add nuw nsw i32 %7, %5
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %10
  store double %24, ptr %11, align 8
  ret void

12:
  %13 = phi i32 [ 0, %4 ], [ %25, %12 ]
  %14 = phi double [ 0.000000e+00, %4 ], [ %24, %12 ]
  %15 = add nuw nsw i32 %13, %7
  %16 = zext nneg i32 %15 to i64
  %17 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %16
  %18 = load double, ptr %17, align 8
  %19 = shl nuw nsw i32 %13, 4
  %20 = add nuw nsw i32 %19, %5
  %21 = zext nneg i32 %20 to i64
  %22 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %21
  %23 = load double, ptr %22, align 8
  %24 = tail call double @llvm.fmuladd.f64(double %18, double %23, double %14)
  %25 = add nuw nsw i32 %13, 1
  %26 = icmp eq i32 %25, 16
  br i1 %26, label %8, label %12
}

declare double @llvm.fmuladd.f64(double, double, double) #1
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

attributes #0 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

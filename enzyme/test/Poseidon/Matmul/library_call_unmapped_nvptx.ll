; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/raise_tcec/matmul_body.fpprofile %t/prof/two_store_body.fpprofile
; RUN: %opt %s %loadPoseidon -passes=poseidon,poseidon-finalize \
; RUN:   -poseidon-profile-use=%t/prof \
; RUN:   -poseidon-cache=%t/cache -poseidon-enable-herbie=false \
; RUN:   -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-tau=1e-12 -poseidon-print \
; RUN:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; RUN:   -S -o /dev/null 2>&1 | FileCheck --implicit-check-not='wrote descriptor' --implicit-check-not=fissioned %s

; The fused body of ozaki_dispatch_fused_nvptx.ll, its product's output a
; pointer the kernel loads from a global rather than one of its arguments. The
; host cannot be told where a library call would write, so none is proposed
; and the body is not fissioned.

; CHECK: [poseidon] preprocess_two_store_body: an operand or dimension of the product is not an argument of the kernel _Z9two_storePKdS0_PdS1_, from which the host reads it
; CHECK: [poseidon] error budget: matmul -> baseline F64

source_filename = "unmapped.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@out = dso_local addrspace(1) externally_initialized global ptr null, align 8
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr)], section "llvm.metadata"

define dso_local void @two_store_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2, ptr noundef writeonly captures(none) %3) #0 {
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %7 = shl nuw nsw i32 %6, 4
  br label %13

8:
  %9 = add nuw nsw i32 %7, %5
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %10
  store double %25, ptr %11, align 8
  %12 = getelementptr inbounds nuw [8 x i8], ptr %3, i64 %10
  store double %25, ptr %12, align 8
  ret void

13:
  %14 = phi i32 [ 0, %4 ], [ %26, %13 ]
  %15 = phi double [ 0.000000e+00, %4 ], [ %25, %13 ]
  %16 = add nuw nsw i32 %14, %7
  %17 = zext nneg i32 %16 to i64
  %18 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %17
  %19 = load double, ptr %18, align 8
  %20 = shl nuw nsw i32 %14, 4
  %21 = add nuw nsw i32 %20, %5
  %22 = zext nneg i32 %21 to i64
  %23 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %22
  %24 = load double, ptr %23, align 8
  %25 = tail call double @llvm.fmuladd.f64(double %19, double %24, double %15)
  %26 = add nuw nsw i32 %14, 1
  %27 = icmp eq i32 %26, 16
  br i1 %27, label %8, label %13
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z9two_storePKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %c = load ptr, ptr addrspacecast (ptr addrspace(1) @out to ptr), align 8
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr noundef nonnull @two_store_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %5, ptr noundef %c, i32 noundef %5, ptr noundef %3)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}

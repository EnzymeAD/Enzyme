; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/raise_tcec/preprocess_matmul_body.fpprofile %t/prof/preprocess_tri_body.fpprofile
; RUN: not %opt %s %loadPoseidon \
; RUN:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; RUN:   -poseidon-profile-use=%t/prof -poseidon-cache=%t/c \
; RUN:   -poseidon-enable-herbie=false -poseidon-enable-pt=false \
; RUN:   -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-tau=1e-6 \
; RUN:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; RUN:   -poseidon-print -S -o /dev/null 2>&1 | FileCheck %s

; The in-kernel raise synchronizes the thread block and runs warp-wide
; tensor-core calls in the reduction loop's preheader. In tri_body only the
; threads with threadIdx.x <= threadIdx.y reach the loop, so recognition marks
; the product, and the raise the solve selects for it is refused.

; CHECK: [raise-wmma] preprocess_tri_body: not every thread of the block reaches the reduction loop once
; CHECK: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32
; CHECK-NEXT: error: {{.*}}the in-kernel raise selected for this matrix product synchronizes the thread block, and not every thread of the block reaches the product's loop once.

source_filename = "raise_guard.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

; if (threadIdx.x <= threadIdx.y) { the product of matmul_device.ll }
define dso_local void @tri_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %tri = icmp ule i32 %tx, %ty
  br i1 %tri, label %pre, label %ret

pre:
  %row = shl nuw nsw i32 %ty, 4
  br label %loop

loop:
  %k = phi i32 [ 0, %pre ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %pre ], [ %fma, %loop ]
  %aidx = add nuw nsw i32 %k, %row
  %aidx64 = zext nneg i32 %aidx to i64
  %aaddr = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %aidx64
  %a = load double, ptr %aaddr, align 8
  %krow = shl nuw nsw i32 %k, 4
  %bidx = add nuw nsw i32 %krow, %tx
  %bidx64 = zext nneg i32 %bidx to i64
  %baddr = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %bidx64
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %cidx = add nuw nsw i32 %row, %tx
  %cidx64 = zext nneg i32 %cidx to i64
  %caddr = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %cidx64
  store double %fma, ptr %caddr, align 8
  br label %ret

ret:
  ret void
}

define dso_local ptx_kernel void @_Z10tri_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @tri_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #2
declare double @llvm.fmuladd.f64(double, double, double)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y()

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #2 = { convergent nounwind "uniform-work-group-size" }

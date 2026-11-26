; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: for f in tri gemm staged; do cp %S/../../Inputs/matmul_site_profile/matmul_body.fpprofile %t/prof/${f}_body.fpprofile; done
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-cache=%t/c \
; RUN:   -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-tau=1e-6 \
; RUN:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-print -S -o %t/out.ll 2>&1 | FileCheck --implicit-check-not='Applying solution for matmul' %s
; RUN: FileCheck --check-prefix=IR --implicit-check-not=wmma --implicit-check-not=barrier %s < %t/out.ll

; No in-kernel raise where not every thread of the block reaches the loop or the
; site, or where the kernel's static shared memory leaves no room for the tiles.

; CHECK: [poseidon] preprocess_tri_body: not every thread of the block reaches the reduction loop once, so it has no in-kernel raise
; CHECK: [poseidon] matmul[0] candidates
; CHECK-NEXT: (no candidates emitted)
; CHECK: [poseidon] preprocess_tri_body: tau={{[^ ]+}} S={{[^ ]+}} A0=0.000000e+00 selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00
; CHECK: [poseidon] preprocess_gemm_body: not every thread of the block reaches the site in _Z13guard_kernelPKdS0_PdS1_ once, so it has no in-kernel raise
; CHECK: [poseidon] preprocess_gemm_body: tau={{[^ ]+}} S={{[^ ]+}} A0=0.000000e+00 selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00
; CHECK-DAG: [poseidon] matmul[0] 16x16x16: in-kernel direct raise m16n16k16 f16/f32 not proposed: it needs 50176 B of static shared memory against a 49152 B cap (48128 B already live in the enclosing kernel + 2048 B for the raise: 1024 B D tile (16x16 f32) + 512 B A tile + 512 B B tile)
; CHECK-DAG: [poseidon] matmul[0] 16x16x16: in-kernel TCEC n=2 m16n16k16 f16/f32 not proposed: it needs 52224 B of static shared memory
; CHECK: [poseidon] preprocess_staged_body: tau={{[^ ]+}} S={{[^ ]+}} A0=0.000000e+00 selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00

; IR-LABEL: define {{.*}} @_Z10tri_kernelPKdS0_PdS1_(
; IR-NEXT: call void @tri_body(ptr %A, ptr %B, ptr %C)
; IR-LABEL: define {{.*}} @_Z13guard_kernelPKdS0_PdS1_(
; IR: call void @gemm_body(ptr %A, ptr %B, ptr %C)

source_filename = "raise_refused.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"
@stage = internal addrspace(3) global [48128 x i8] undef, align 16

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

define dso_local void @gemm_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %fma, %loop ]
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
  ret void
}

define dso_local ptx_kernel void @_Z13guard_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
entry:
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %some = icmp ult i32 %tx, 8
  br i1 %some, label %site, label %done

site:
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @gemm_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  br label %done

done:
  ret void
}

define dso_local void @staged_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  store double 0.000000e+00, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @stage, i64 40), align 8
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %fma, %loop ]
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
  ret void
}

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @staged_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #2
declare double @llvm.fmuladd.f64(double, double, double)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y()

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #2 = { convergent nounwind "uniform-work-group-size" }

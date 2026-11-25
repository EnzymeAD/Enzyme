; DEFINE: %{opt} = %opt %s %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-enable-pt=false -poseidon-raise-wmma \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; DEFINE:   -poseidon-print -S
; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/raise_tcec/matmul_body.fpprofile %t/prof/seed_body.fpprofile
; RUN: %{opt} -poseidon-cache=%t/lib -poseidon-tau=1e-12 -o %t/lib.ll 2>&1 \
; RUN:   | FileCheck --implicit-check-not='Matmul[' --implicit-check-not='wrote descriptor' %s
; RUN: %{opt} -poseidon-cache=%t/raise -poseidon-tau=1e-6 -poseidon-ozaki-host-dispatch=false -o %t/raise.ll 2>&1 \
; RUN:   | FileCheck --implicit-check-not='Matmul[' %s
; RUN: FileCheck --check-prefix=IR --implicit-check-not=wmma %s < %t/raise.ll

; The product of Inputs/stale_cache/matmul_device.ll accumulated onto the
; value C already holds, `double acc = C[ty * 16 + tx]` before the loop. Both
; realizations compute the product alone, so the reduction is not recognized
; and the kernel calls the body as written.

; CHECK: [raise-wmma] reject loop in preprocess_seed_body: the accumulator does not start at +0.0

; IR-LABEL: define {{.*}} @_Z13matmul_kernelPKdS0_PdS1_(
; IR-NEXT: call void @seed_body(ptr %A, ptr %B, ptr %C)

source_filename = "seed.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @seed_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  %cidx = add nuw nsw i32 %row, %tx
  %cidx64 = zext nneg i32 %cidx to i64
  %caddr = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %cidx64
  %cinit = load double, ptr %caddr, align 8
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ %cinit, %entry ], [ %fma, %loop ]
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
  store double %fma, ptr %caddr, align 8
  ret void
}

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @seed_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #2
declare double @llvm.fmuladd.f64(double, double, double)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y()

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #2 = { convergent nounwind "uniform-work-group-size" }

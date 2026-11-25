; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/raise_tcec/matmul_body.fpprofile %t/prof/trip_body.fpprofile
; RUN: cp %S/../Inputs/raise_tcec/matmul_body.fpprofile %t/prof/exit_body.fpprofile
; RUN: %opt %s %loadPoseidon \
; RUN:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; RUN:   -poseidon-profile-use=%t/prof -poseidon-cache=%t/c \
; RUN:   -poseidon-enable-herbie=false -poseidon-enable-pt=false \
; RUN:   -poseidon-raise-wmma -poseidon-tau=1e-6 \
; RUN:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; RUN:   -poseidon-print -S -o %t/out.ll 2>&1 \
; RUN:   | FileCheck --implicit-check-not='Matmul[' --implicit-check-not='wrote descriptor' %s
; RUN: FileCheck --check-prefix=IR --implicit-check-not=wmma %s < %t/out.ll

; Reduction loops whose trip count is not a compile-time constant are not
; products of constant shape. trip_body runs k over [0, n & 31), whose largest
; trip count, 31, is a constant; exit_body leaves its 16-step loop early when
; k reaches n. Neither is recognized, and both kernels call their bodies as
; written.

; CHECK-DAG: [poseidon] reject loop in preprocess_trip_body: trip count is not a known constant
; CHECK-DAG: [poseidon] reject loop in preprocess_exit_body: the loop exits from a block other than its latch
; CHECK-DAG: [poseidon]   reject phi in loop: the loop exits from a block other than its latch

; IR-LABEL: define {{.*}} @trip_kernel(
; IR-NEXT: call void @trip_body(ptr %A, ptr %B, ptr %C, i32 %n)
; IR-LABEL: define {{.*}} @exit_kernel(
; IR-NEXT: call void @exit_body(ptr %A, ptr %B, ptr %C, i32 %n)

source_filename = "shape.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

; for (int64_t k = 0; k < (int64_t)(n & 31); ++k)
define dso_local void @trip_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C, i32 noundef %n) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %tx64 = zext nneg i32 %tx to i64
  %ty64 = zext nneg i32 %ty to i64
  %row = shl nuw nsw i64 %ty64, 4
  %nk = and i32 %n, 31
  %nk64 = zext nneg i32 %nk to i64
  %any = icmp ne i32 %nk, 0
  br i1 %any, label %loop, label %exit

loop:
  %k = phi i64 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %fma, %loop ]
  %aidx = add nuw nsw i64 %k, %row
  %aaddr = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %aidx
  %a = load double, ptr %aaddr, align 8
  %krow = shl nuw nsw i64 %k, 4
  %bidx = add nuw nsw i64 %krow, %tx64
  %baddr = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %bidx
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, %nk64
  br i1 %done, label %exit, label %loop

exit:
  %sum = phi double [ 0.000000e+00, %entry ], [ %fma, %loop ]
  %cidx = add nuw nsw i64 %row, %tx64
  %caddr = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %cidx
  store double %sum, ptr %caddr, align 8
  ret void
}

; for (int k = 0; k < 16; ++k) { if (k == n) break; acc += ...; }
define dso_local void @exit_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C, i32 noundef %n) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %latch ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %fma, %latch ]
  %stop = icmp eq i32 %k, %n
  br i1 %stop, label %exit, label %latch

latch:
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
  %sum = phi double [ %acc, %loop ], [ %fma, %latch ]
  %cidx = add nuw nsw i32 %row, %tx
  %cidx64 = zext nneg i32 %cidx to i64
  %caddr = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %cidx64
  store double %sum, ptr %caddr, align 8
  ret void
}

define dso_local ptx_kernel void @trip_kernel(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC, i32 noundef %n) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iEET_PvDpT0_(ptr noundef nonnull @trip_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC, i32 noundef %n) #2
  ret void
}

define dso_local ptx_kernel void @exit_kernel(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC, i32 noundef %n) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iEET_PvDpT0_(ptr noundef nonnull @exit_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC, i32 noundef %n) #2
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_iEET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef, i32 noundef) local_unnamed_addr #2
declare double @llvm.fmuladd.f64(double, double, double)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y()

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #2 = { convergent nounwind "uniform-work-group-size" }

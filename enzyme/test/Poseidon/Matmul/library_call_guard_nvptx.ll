; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: cp %S/../Inputs/raise_tcec/preprocess_matmul_body.fpprofile %t/prof/preprocess_tri_body.fpprofile
; RUN: cp %S/../Inputs/raise_tcec/preprocess_matmul_body.fpprofile %t/prof/preprocess_short_body.fpprofile
; RUN: cp %S/../Inputs/raise_tcec/preprocess_matmul_body.fpprofile %t/prof/preprocess_bound_body.fpprofile
; RUN: cp %S/../Inputs/raise_tcec/preprocess_matmul_body.fpprofile %t/prof/preprocess_wrap_body.fpprofile
; RUN: %opt %s %loadPoseidon \
; RUN:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; RUN:   -poseidon-profile-use=%t/prof -poseidon-cache=%t/c \
; RUN:   -poseidon-enable-herbie=false -poseidon-enable-pt=false \
; RUN:   -poseidon-raise-wmma -poseidon-tau=1e-12 \
; RUN:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; RUN:   -poseidon-print -S -o /dev/null 2>&1 | FileCheck %s
; RUN: cat %t/c/launches_*.json | FileCheck --check-prefix=DESC --implicit-check-not=_kernel %s

; A library call writes every element of the M x Ncols product, so the store it
; replaces may be gated only by bounds on the product's indices that keep every
; element the call writes. tri_body stores only where threadIdx.x <=
; threadIdx.y, short_body only the first 8 of its 16 rows; neither is
; proposed. bound_body and wrap_body return early unless row < 16 && col < 16
; on the global thread indices, which keeps the whole 16 x 16 product; bound_body
; reaches the pass with the two bounds folded into (row | col) u< 16. Both
; library calls are written.

; CHECK-DAG: [ozaki-host-dispatch] preprocess_tri_body: a condition gating the GEMM store is not a thread-index bound
; CHECK-DAG: [ozaki-host-dispatch] preprocess_short_body: the GEMM store is gated by row < 8, which drops rows of the 16 x 16 product the library call writes
; CHECK-DAG: [ozaki-host-dispatch] wrote descriptor for _Z12bound_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)
; CHECK-DAG: [ozaki-host-dispatch] wrote descriptor for _Z11wrap_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; DESC-DAG: "_Z12bound_kernelPKdS0_PdS1_ 2 0 1 1 12 0 0 0 0 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"
; DESC-DAG: "_Z11wrap_kernelPKdS0_PdS1_ 2 0 1 1 12 0 0 0 0 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"

source_filename = "guard.cu"
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

; int row = blockIdx.y * blockDim.y + threadIdx.y, col likewise along x;
; the product of matmul_device.ll at (row, col), stored only if row < 8.
define dso_local void @short_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %bx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %by = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %nx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %ny = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %by.ny = mul nuw nsw i32 %by, %ny
  %grow = add nuw nsw i32 %by.ny, %ty
  %bx.nx = mul nuw nsw i32 %bx, %nx
  %gcol = add nuw nsw i32 %bx.nx, %tx
  %roff = shl nuw nsw i32 %grow, 7
  %roff64 = zext nneg i32 %roff to i64
  %arow = getelementptr inbounds nuw i8, ptr %A, i64 %roff64
  %gcol64 = sext i32 %gcol to i64
  %bcol = getelementptr [8 x i8], ptr %B, i64 %gcol64
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %fma, %loop ]
  %k64 = zext nneg i32 %k to i64
  %aaddr = getelementptr inbounds nuw [8 x i8], ptr %arow, i64 %k64
  %a = load double, ptr %aaddr, align 8
  %kb = shl nuw nsw i64 %k64, 7
  %baddr = getelementptr i8, ptr %bcol, i64 %kb
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %keep = icmp slt i32 %grow, 8
  br i1 %keep, label %store, label %ret

store:
  %crow = getelementptr i8, ptr %C, i64 %roff64
  %caddr = getelementptr [8 x i8], ptr %crow, i64 %gcol64
  store double %fma, ptr %caddr, align 8
  br label %ret

ret:
  ret void
}

; As short_body, with if (row >= 16 || col >= 16) return; ahead of the loop
; and no guard on the store.
define dso_local void @bound_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %bx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %by = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %nx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %ny = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %by.ny = mul nuw nsw i32 %by, %ny
  %grow = add nuw nsw i32 %by.ny, %ty
  %bx.nx = mul nuw nsw i32 %bx, %nx
  %gcol = add nuw nsw i32 %bx.nx, %tx
  %rowout = icmp sgt i32 %grow, 15
  %colout = icmp sgt i32 %gcol, 15
  %out = or i1 %rowout, %colout
  br i1 %out, label %ret, label %pre

pre:
  %roff = shl nuw nsw i32 %grow, 7
  %roff64 = zext nneg i32 %roff to i64
  %arow = getelementptr inbounds nuw i8, ptr %A, i64 %roff64
  %gcol64 = sext i32 %gcol to i64
  %bcol = getelementptr [8 x i8], ptr %B, i64 %gcol64
  br label %loop

loop:
  %k = phi i32 [ 0, %pre ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %pre ], [ %fma, %loop ]
  %k64 = zext nneg i32 %k to i64
  %aaddr = getelementptr inbounds nuw [8 x i8], ptr %arow, i64 %k64
  %a = load double, ptr %aaddr, align 8
  %kb = shl nuw nsw i64 %k64, 7
  %baddr = getelementptr i8, ptr %bcol, i64 %kb
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %crow = getelementptr i8, ptr %C, i64 %roff64
  %caddr = getelementptr [8 x i8], ptr %crow, i64 %gcol64
  store double %fma, ptr %caddr, align 8
  br label %ret

ret:
  ret void
}

; bound_body with col computed as clang emits it from a blockIdx.x that may
; reach 2^31, so the guard stays two comparisons.
define dso_local void @wrap_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %bx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %by = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %nx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %ny = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %by.ny = mul nuw nsw i32 %by, %ny
  %grow = add nuw nsw i32 %by.ny, %ty
  %bx.nx = mul i32 %bx, %nx
  %gcol = add i32 %bx.nx, %tx
  %rowout = icmp sgt i32 %grow, 15
  %colout = icmp sgt i32 %gcol, 15
  %out = or i1 %rowout, %colout
  br i1 %out, label %ret, label %pre

pre:
  %roff = shl nuw nsw i32 %grow, 7
  %roff64 = zext nneg i32 %roff to i64
  %arow = getelementptr inbounds nuw i8, ptr %A, i64 %roff64
  %gcol64 = sext i32 %gcol to i64
  %bcol = getelementptr [8 x i8], ptr %B, i64 %gcol64
  br label %loop

loop:
  %k = phi i32 [ 0, %pre ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %pre ], [ %fma, %loop ]
  %k64 = zext nneg i32 %k to i64
  %aaddr = getelementptr inbounds nuw [8 x i8], ptr %arow, i64 %k64
  %a = load double, ptr %aaddr, align 8
  %kb = shl nuw nsw i64 %k64, 7
  %baddr = getelementptr i8, ptr %bcol, i64 %kb
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %crow = getelementptr i8, ptr %C, i64 %roff64
  %caddr = getelementptr [8 x i8], ptr %crow, i64 %gcol64
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

define dso_local ptx_kernel void @_Z12short_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @short_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z12bound_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @bound_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z11wrap_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @wrap_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
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

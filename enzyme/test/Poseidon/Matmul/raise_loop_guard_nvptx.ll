; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: for f in or one and fold intra; do cp %S/../Inputs/raise_tcec/matmul_body.fpprofile %t/prof/${f}_body.fpprofile; done
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; RUN:   -poseidon-profile-use=%t/prof -poseidon-cache=%t/c \
; RUN:   -poseidon-enable-herbie=false -poseidon-enable-pt=false \
; RUN:   -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-tau=1e-6 \
; RUN:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; RUN:   -poseidon-print -S -o %t/out.ll 2>&1 | FileCheck %s
; RUN: FileCheck --check-prefix=IR %s < %t/out.ll

; Each body computes its row and column as global thread indices, row =
; blockIdx.y * blockDim.y + threadIdx.y and col = blockIdx.x * blockDim.x +
; threadIdx.x, and runs its reduction loop only for the threads that pass a
; guard on them. A block raises only when every one of its threads passes the
; guard, blockIdx < bound / 16 on each bounded axis, besides having the
; profiled 16 x 16 x 1 shape; any other block runs the guard and the loop as
; written, and both paths store the product. The guard forms:
;   or_body:   if (row >= 48 || col >= 32) return;   (an or of two compares)
;   one_body:  if (row < 48) { ... }                 (a single bound)
;   and_body:  if (row < 48 && col < 32) { ... }     (an and of two compares)
;   fold_body: if ((row | col) < 64) { ... }         (the folded form)
; intra_body runs the loop only for threadIdx.x < 8, a bound on no global
; index, so not every thread of a raising block would reach its barriers.

; CHECK: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; CHECK: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; CHECK: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; CHECK: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
; CHECK: [poseidon] preprocess_intra_body: not every thread of the block reaches the reduction loop once, so it has no in-kernel raise
; CHECK: [poseidon] error budget: matmul -> baseline F64
; CHECK-NOT: Applying solution for matmul

; IR-LABEL: define {{.*}} @preprocess_or_body(
; IR: %oob = or i1 %r.oob, %c.oob
; IR-NEXT: br i1 %oob, label %out, label %body
; IR: [[SX:%[0-9]+]] = icmp eq i32 %nx, 16
; IR-NEXT: [[SY:%[0-9]+]] = icmp eq i32 %ny, 16
; IR-NEXT: [[SXY:%[0-9]+]] = and i1 [[SX]], [[SY]]
; IR-NEXT: [[NZ:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; IR-NEXT: [[SZ:%[0-9]+]] = icmp eq i32 [[NZ]], 1
; IR-NEXT: [[SHAPE:%[0-9]+]] = and i1 [[SZ]], [[SXY]]
; IR-NEXT: %raise.inside = icmp samesign ult i32 %bx, 2
; IR-NEXT: [[X:%[0-9]+]] = and i1 %raise.inside, [[SHAPE]]
; IR-NEXT: %raise.inside1 = icmp samesign ult i32 %by, 3
; IR-NEXT: [[XY:%[0-9]+]] = and i1 %raise.inside1, [[X]]
; IR-NEXT: br i1 [[XY]], label %raise.entry, label %loop.preheader
; IR: {{^}}exit:
; IR-NEXT: %raise.result = phi double [ %{{[0-9]+}}, %ozaki_store_gate.merge ], [ %fma.15, %loop.preheader ]
; IR: store double %raise.result, ptr %caddr, align 8

; IR-LABEL: define {{.*}} @preprocess_one_body(
; IR: br i1 %r.in, label %body, label %out
; IR: [[SX:%[0-9]+]] = icmp eq i32 %nx, 16
; IR-NEXT: [[SY:%[0-9]+]] = icmp eq i32 %ny, 16
; IR-NEXT: [[SXY:%[0-9]+]] = and i1 [[SX]], [[SY]]
; IR-NEXT: [[NZ:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; IR-NEXT: [[SZ:%[0-9]+]] = icmp eq i32 [[NZ]], 1
; IR-NEXT: [[SHAPE:%[0-9]+]] = and i1 [[SZ]], [[SXY]]
; IR-NEXT: %raise.inside = icmp samesign ult i32 %by, 3
; IR-NEXT: [[Y:%[0-9]+]] = and i1 %raise.inside, [[SHAPE]]
; IR-NEXT: br i1 [[Y]], label %raise.entry, label %loop.preheader
; IR: %raise.result = phi double

; IR-LABEL: define {{.*}} @preprocess_and_body(
; IR: %in = and i1 %r.in, %c.in
; IR-NEXT: br i1 %in, label %body, label %out
; IR: [[SX:%[0-9]+]] = icmp eq i32 %nx, 16
; IR-NEXT: [[SY:%[0-9]+]] = icmp eq i32 %ny, 16
; IR-NEXT: [[SXY:%[0-9]+]] = and i1 [[SX]], [[SY]]
; IR-NEXT: [[NZ:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; IR-NEXT: [[SZ:%[0-9]+]] = icmp eq i32 [[NZ]], 1
; IR-NEXT: [[SHAPE:%[0-9]+]] = and i1 [[SZ]], [[SXY]]
; IR-NEXT: %raise.inside = icmp samesign ult i32 %bx, 2
; IR-NEXT: [[X:%[0-9]+]] = and i1 %raise.inside, [[SHAPE]]
; IR-NEXT: %raise.inside1 = icmp samesign ult i32 %by, 3
; IR-NEXT: [[XY:%[0-9]+]] = and i1 %raise.inside1, [[X]]
; IR-NEXT: br i1 [[XY]], label %raise.entry, label %loop.preheader
; IR: %raise.result = phi double

; IR-LABEL: define {{.*}} @preprocess_fold_body(
; IR: %in = icmp ult i32 %rc, 64
; IR-NEXT: br i1 %in, label %body, label %out
; IR: [[SX:%[0-9]+]] = icmp eq i32 %nx, 16
; IR-NEXT: [[SY:%[0-9]+]] = icmp eq i32 %ny, 16
; IR-NEXT: [[SXY:%[0-9]+]] = and i1 [[SX]], [[SY]]
; IR-NEXT: [[NZ:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; IR-NEXT: [[SZ:%[0-9]+]] = icmp eq i32 [[NZ]], 1
; IR-NEXT: [[SHAPE:%[0-9]+]] = and i1 [[SZ]], [[SXY]]
; IR-NEXT: [[BXY:%[0-9]+]] = or i32 %by, %bx
; IR-NEXT: [[IN:%[0-9]+]] = icmp samesign ult i32 [[BXY]], 4
; IR-NEXT: [[RAISE:%[0-9]+]] = and i1 [[IN]], [[SHAPE]]
; IR-NEXT: br i1 [[RAISE]], label %raise.entry, label %loop.preheader
; IR: %raise.result = phi double

source_filename = "loop_guard.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @or_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %bx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %nx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bxn = mul i32 %bx, %nx
  %col = add i32 %bxn, %tx
  %by = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %ny = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %byn = mul i32 %by, %ny
  %row = add i32 %byn, %ty
  %r.oob = icmp sgt i32 %row, 47
  %c.oob = icmp ugt i32 %col, 31
  %oob = or i1 %r.oob, %c.oob
  br i1 %oob, label %out, label %body

body:
  %row64 = sext i32 %row to i64
  %col64 = sext i32 %col to i64
  %aoff = shl nsw i64 %row64, 4
  %arow = getelementptr inbounds double, ptr %A, i64 %aoff
  %bcol = getelementptr inbounds double, ptr %B, i64 %col64
  br label %loop

loop:
  %k = phi i64 [ 0, %body ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %body ], [ %fma, %loop ]
  %aaddr = getelementptr inbounds double, ptr %arow, i64 %k
  %a = load double, ptr %aaddr, align 8
  %koff = shl nsw i64 %k, 5
  %baddr = getelementptr inbounds double, ptr %bcol, i64 %koff
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %coff = shl nsw i64 %row64, 5
  %crow = getelementptr inbounds double, ptr %C, i64 %coff
  %caddr = getelementptr inbounds double, ptr %crow, i64 %col64
  store double %fma, ptr %caddr, align 8
  br label %out

out:
  ret void
}

define dso_local void @one_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %bx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %nx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bxn = mul i32 %bx, %nx
  %col = add i32 %bxn, %tx
  %by = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %ny = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %byn = mul i32 %by, %ny
  %row = add i32 %byn, %ty
  %r.in = icmp slt i32 %row, 48
  br i1 %r.in, label %body, label %out

body:
  %row64 = sext i32 %row to i64
  %col64 = sext i32 %col to i64
  %aoff = shl nsw i64 %row64, 4
  %arow = getelementptr inbounds double, ptr %A, i64 %aoff
  %bcol = getelementptr inbounds double, ptr %B, i64 %col64
  br label %loop

loop:
  %k = phi i64 [ 0, %body ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %body ], [ %fma, %loop ]
  %aaddr = getelementptr inbounds double, ptr %arow, i64 %k
  %a = load double, ptr %aaddr, align 8
  %koff = shl nsw i64 %k, 5
  %baddr = getelementptr inbounds double, ptr %bcol, i64 %koff
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %coff = shl nsw i64 %row64, 5
  %crow = getelementptr inbounds double, ptr %C, i64 %coff
  %caddr = getelementptr inbounds double, ptr %crow, i64 %col64
  store double %fma, ptr %caddr, align 8
  br label %out

out:
  ret void
}

define dso_local void @and_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %bx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %nx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bxn = mul i32 %bx, %nx
  %col = add i32 %bxn, %tx
  %by = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %ny = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %byn = mul i32 %by, %ny
  %row = add i32 %byn, %ty
  %r.in = icmp ult i32 %row, 48
  %c.in = icmp slt i32 %col, 32
  %in = and i1 %r.in, %c.in
  br i1 %in, label %body, label %out

body:
  %row64 = sext i32 %row to i64
  %col64 = sext i32 %col to i64
  %aoff = shl nsw i64 %row64, 4
  %arow = getelementptr inbounds double, ptr %A, i64 %aoff
  %bcol = getelementptr inbounds double, ptr %B, i64 %col64
  br label %loop

loop:
  %k = phi i64 [ 0, %body ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %body ], [ %fma, %loop ]
  %aaddr = getelementptr inbounds double, ptr %arow, i64 %k
  %a = load double, ptr %aaddr, align 8
  %koff = shl nsw i64 %k, 5
  %baddr = getelementptr inbounds double, ptr %bcol, i64 %koff
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %coff = shl nsw i64 %row64, 5
  %crow = getelementptr inbounds double, ptr %C, i64 %coff
  %caddr = getelementptr inbounds double, ptr %crow, i64 %col64
  store double %fma, ptr %caddr, align 8
  br label %out

out:
  ret void
}

define dso_local void @fold_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %bx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %nx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bxn = mul i32 %bx, %nx
  %col = add i32 %bxn, %tx
  %by = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %ny = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %byn = mul i32 %by, %ny
  %row = add i32 %byn, %ty
  %rc = or i32 %row, %col
  %in = icmp ult i32 %rc, 64
  br i1 %in, label %body, label %out

body:
  %row64 = sext i32 %row to i64
  %col64 = sext i32 %col to i64
  %aoff = shl nsw i64 %row64, 4
  %arow = getelementptr inbounds double, ptr %A, i64 %aoff
  %bcol = getelementptr inbounds double, ptr %B, i64 %col64
  br label %loop

loop:
  %k = phi i64 [ 0, %body ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %body ], [ %fma, %loop ]
  %aaddr = getelementptr inbounds double, ptr %arow, i64 %k
  %a = load double, ptr %aaddr, align 8
  %koff = shl nsw i64 %k, 5
  %baddr = getelementptr inbounds double, ptr %bcol, i64 %koff
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %coff = shl nsw i64 %row64, 5
  %crow = getelementptr inbounds double, ptr %C, i64 %coff
  %caddr = getelementptr inbounds double, ptr %crow, i64 %col64
  store double %fma, ptr %caddr, align 8
  br label %out

out:
  ret void
}

define dso_local void @intra_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %bx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %nx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bxn = mul i32 %bx, %nx
  %col = add i32 %bxn, %tx
  %by = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %ny = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %byn = mul i32 %by, %ny
  %row = add i32 %byn, %ty
  %some = icmp ult i32 %tx, 8
  br i1 %some, label %body, label %out

body:
  %row64 = sext i32 %row to i64
  %col64 = sext i32 %col to i64
  %aoff = shl nsw i64 %row64, 4
  %arow = getelementptr inbounds double, ptr %A, i64 %aoff
  %bcol = getelementptr inbounds double, ptr %B, i64 %col64
  br label %loop

loop:
  %k = phi i64 [ 0, %body ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %body ], [ %fma, %loop ]
  %aaddr = getelementptr inbounds double, ptr %arow, i64 %k
  %a = load double, ptr %aaddr, align 8
  %koff = shl nsw i64 %k, 5
  %baddr = getelementptr inbounds double, ptr %bcol, i64 %koff
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i64 %k, 1
  %done = icmp eq i64 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %coff = shl nsw i64 %row64, 5
  %crow = getelementptr inbounds double, ptr %C, i64 %coff
  %caddr = getelementptr inbounds double, ptr %crow, i64 %col64
  store double %fma, ptr %caddr, align 8
  br label %out

out:
  ret void
}

define dso_local ptx_kernel void @or_body_kernel(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
entry:
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @or_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @one_body_kernel(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
entry:
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @one_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @and_body_kernel(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
entry:
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @and_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @fold_body_kernel(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
entry:
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @fold_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @intra_body_kernel(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
entry:
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @intra_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
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

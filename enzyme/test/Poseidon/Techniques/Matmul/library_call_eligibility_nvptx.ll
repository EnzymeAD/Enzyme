; DEFINE: %{host} = %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %loadClangPoseidon -mllvm -poseidon-profile-use=%t/prof -mllvm -poseidon-early-prune -mllvm \
; DEFINE:   -poseidon-cache=%t/c -mllvm -poseidon-compilation-id=layout -S -emit-llvm -o -
; DEFINE: %{refused} = _Z([0-9]+(tri|short|offset|shifted|fused)_kernel|13matmul_kernfc|8unmapped)
; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: for f in tri short bound wrap nt nn_col_c fused_c row_skip_b gemm shifted fused two_store unmapped; do cp %S/../../Inputs/matmul_site_profile/matmul_body.fpprofile \
; RUN:   %t/prof/${f}_body.fpprofile; done
; RUN: %opt %s %loadPoseidon -passes=poseidon,poseidon-finalize -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-cache=%t/c -poseidon-compilation-id=layout \
; RUN:   -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-tau=1e-12 -poseidon-print -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv \
; RUN:   -S -o %t/out.ll 2> %t/err
; RUN: FileCheck %s < %t/err
; RUN: not grep -E 'wrote descriptor for %{refused}' %t/err
; RUN: FileCheck --check-prefix=FUSED %s < %t/err
; RUN: FileCheck --check-prefix=UNMAPPED --implicit-check-not='fissioned the product in preprocess_unmapped_body' %s < %t/err
; RUN: cat %t/c/launches_*.json | FileCheck --check-prefix=DESC %s
; RUN: not grep -E '%{refused}' %t/c/launches_*.json
; RUN: FileCheck --check-prefix=FISSION %s < %t/out.ll
; RUN: %opt %s %loadPoseidon -passes=poseidon,poseidon-finalize -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-cache=%t/q -poseidon-compilation-id=layout \
; RUN:   -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-tau=1e-12 -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv \
; RUN:   -S -o %t/quiet.ll 2> %t/quiet.err
; RUN: diff %t/out.ll %t/quiet.ll
; RUN: sed 's/ozaki_stub\.cu/layout.cu/g' %S/../../Inputs/library_call/ozaki_stub_host.ll > %t/nt.ll
; RUN: sed 's/ozaki_stub\.cu/layout.cu/g; s/matmul_kernel/matmul_kerncc/g' %S/../../Inputs/library_call/ozaki_stub_host.ll > %t/nncolc.ll
; RUN: %{host} %t/nt.ll 2>&1 | FileCheck --check-prefixes=HOST,HOST-NT %s
; RUN: %{host} %t/nncolc.ll 2>&1 | FileCheck --check-prefixes=HOST,HOST-NN %s

; Which products get a library call, which writes all of C from the kernel's
; own arguments in the storage order and leading dimension of the body's store:
; each refusal names its reason, each call written its descriptor.

; CHECK-DAG: [poseidon] preprocess_tri_body: a condition gating the GEMM store is not a thread-index bound
; CHECK-DAG: [poseidon] preprocess_short_body: the GEMM store is gated by row < 8, which drops rows of the 16 x 16 product the library call writes
; CHECK-DAG: [poseidon] wrote descriptor for _Z12bound_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)
; CHECK-DAG: [poseidon] wrote descriptor for _Z11wrap_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)
; CHECK-DAG: [poseidon] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)
; CHECK-DAG: [poseidon] wrote descriptor for _Z13matmul_kernccPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)
; CHECK-DAG: [poseidon] preprocess_fused_c_body: the product's store address is not a row-major or column-major element of C
; CHECK-DAG: [poseidon] wrote descriptor for _Z13matmul_kernrsPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)
; CHECK-DAG: [poseidon] preprocess_gemm_body: an operand or dimension of the product is not an argument of the kernel _Z13offset_kernelPKdS0_PdS1_, from which the host reads it
; CHECK-DAG: [poseidon] preprocess_shifted_body: an operand of the product does not start where its pointer parameter points

; FUSED: Matmul[0]: 256x1x16 type=f64 origin=ConstantShape
; FUSED: [poseidon] preprocess_fused_body: the profile does not give the extent of the product's rows and columns
; FUSED-NEXT: [poseidon] matmul[0] 256x1x16: library-call Ozaki-II not proposed: the body is not a library-call GEMM

; UNMAPPED: [poseidon] preprocess_unmapped_body: an operand or dimension of the product is not an argument of the kernel _Z8unmappedPKdS0_PdS1_, from which the host reads it
; UNMAPPED: [poseidon] preprocess_unmapped_body: tau={{[^ ]+}} S={{[^ ]+}} A0=0.000000e+00 selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00

; DESC-DAG: "_Z12bound_kernelPKdS0_PdS1_ 2 0 1 1 12 0 0 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"
; DESC-DAG: "_Z11wrap_kernelPKdS0_PdS1_ 2 0 1 1 12 0 0 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"
; DESC-DAG: "_Z13matmul_kernelPKdS0_PdS1_ 2 0 1 1 12 0 1 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"
; DESC-DAG: "_Z13matmul_kernccPKdS0_PdS1_ 2 0 1 1 12 0 0 1 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"
; DESC-DAG: "_Z13matmul_kernrsPKdS0_PdS1_ 2 0 1 1 12 0 0 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"
; DESC-DAG: "_Z9two_storePKdS0_PdS1_ 2 0 1 0 12 0 0 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"

; FISSION-LABEL: define {{.*}} @preprocess_two_store_body(
; FISSION-SAME: ptr {{.*}}, ptr {{.*}}, ptr {{.*}}[[C:%.+]], ptr {{.*}}[[D:%.+]])
; FISSION: [[CADDR:%.+]] = getelementptr inbounds nuw [8 x i8], ptr [[C]]
; FISSION-NEXT: [[PRODUCT:%.+]] = load double, ptr [[CADDR]]
; FISSION-NEXT: [[DADDR:%.+]] = getelementptr inbounds nuw [8 x i8], ptr [[D]]
; FISSION-NEXT: store double [[PRODUCT]], ptr [[DADDR]]
; FISSION-NEXT: ret void

; HOST: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; HOST: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; HOST-NT: call void @__poseidon_ozakiii_dgemm(ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 1, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]], i32 12)
; HOST-NN: call void @__poseidon_ozakiii_dgemm(ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, i32 1, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]], i32 12)

source_filename = "layout.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@out = dso_local addrspace(1) externally_initialized global ptr null, align 8
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

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

define dso_local void @nt_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  %col = shl nuw nsw i32 %tx, 4
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %fma, %loop ]
  %aidx = add nuw nsw i32 %k, %row
  %aidx64 = zext nneg i32 %aidx to i64
  %aaddr = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %aidx64
  %a = load double, ptr %aaddr, align 8
  %bidx = add nuw nsw i32 %col, %k
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

define dso_local void @nn_col_c_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
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
  %col = shl nuw nsw i32 %tx, 4
  %cidx = add nuw nsw i32 %col, %ty
  %cidx64 = zext nneg i32 %cidx to i64
  %caddr = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %cidx64
  store double %fma, ptr %caddr, align 8
  ret void
}

define dso_local void @fused_c_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
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
  %cidx16 = shl nuw nsw i32 %cidx, 4
  %cidx64 = zext nneg i32 %cidx16 to i64
  %caddr = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %cidx64
  store double %fma, ptr %caddr, align 8
  ret void
}

define dso_local void @row_skip_b_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
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
  %k2 = shl nuw nsw i32 %k, 1
  %k64 = zext nneg i32 %k2 to i64
  %brow = getelementptr inbounds [64 x i8], ptr %B, i64 %k64
  %tx64 = zext nneg i32 %tx to i64
  %baddr = getelementptr inbounds [8 x i8], ptr %brow, i64 %tx64
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

define dso_local void @shifted_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %row = shl nuw nsw i32 %ty, 4
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %fma, %loop ]
  %ak = add nuw nsw i32 %k, %row
  %aidx = add nuw nsw i32 %ak, 3
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

define dso_local void @fused_body(ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, ptr noundef writeonly captures(none) %C) #0 {
entry:
  %tx = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ty = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %tz = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %ty16 = shl nuw nsw i32 %ty, 4
  %r = add nuw nsw i32 %ty16, %tx
  %arow = shl nuw nsw i32 %r, 4
  %bcol = shl nuw nsw i32 %tz, 4
  br label %loop

loop:
  %k = phi i32 [ 0, %entry ], [ %k.next, %loop ]
  %acc = phi double [ 0.000000e+00, %entry ], [ %fma, %loop ]
  %aidx = add nuw nsw i32 %arow, %k
  %aidx64 = zext nneg i32 %aidx to i64
  %aaddr = getelementptr inbounds nuw [8 x i8], ptr %A, i64 %aidx64
  %a = load double, ptr %aaddr, align 8
  %bidx = add nuw nsw i32 %bcol, %k
  %bidx64 = zext nneg i32 %bidx to i64
  %baddr = getelementptr inbounds nuw [8 x i8], ptr %B, i64 %bidx64
  %b = load double, ptr %baddr, align 8
  %fma = tail call double @llvm.fmuladd.f64(double %a, double %b, double %acc)
  %k.next = add nuw nsw i32 %k, 1
  %done = icmp eq i32 %k.next, 16
  br i1 %done, label %exit, label %loop

exit:
  %tz256 = shl nuw nsw i32 %tz, 8
  %cidx = add nuw nsw i32 %tz256, %r
  %cidx64 = zext nneg i32 %cidx to i64
  %caddr = getelementptr inbounds nuw [8 x i8], ptr %C, i64 %cidx64
  store double %fma, ptr %caddr, align 8
  ret void
}

define dso_local void @two_store_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2, ptr noundef writeonly captures(none) %3) #3 {
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

define dso_local void @unmapped_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2, ptr noundef writeonly captures(none) %3) #3 {
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

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @nt_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z13matmul_kernccPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @nn_col_c_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z13matmul_kernfcPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @fused_c_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z13matmul_kernrsPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @row_skip_b_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z13offset_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  %Cp = getelementptr inbounds double, ptr %C, i64 256
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @gemm_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %Cp, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z14shifted_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @shifted_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z12fused_kernelPKdS0_PdS1_(ptr noundef %A, ptr noundef %B, ptr noundef %C, ptr noundef %dC) #1 {
  %const = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %dup = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @fused_body, i32 noundef %const, ptr noundef %A, i32 noundef %const, ptr noundef %B, i32 noundef %dup, ptr noundef %C, ptr noundef %dC) #2
  ret void
}

define dso_local ptx_kernel void @_Z9two_storePKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #4 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr noundef nonnull @two_store_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %5, ptr noundef %2, i32 noundef %5, ptr noundef %3)
  ret void
}

define dso_local ptx_kernel void @_Z8unmappedPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) #4 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %c = load ptr, ptr addrspacecast (ptr addrspace(1) @out to ptr), align 8
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr noundef nonnull @unmapped_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %5, ptr noundef %c, i32 noundef %5, ptr noundef %3)
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #2
declare void @__poseidon_fp_optimize(ptr, ...)
declare double @llvm.fmuladd.f64(double, double, double)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare noundef range(i32 0, 64) i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y()

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #2 = { convergent nounwind "uniform-work-group-size" }
attributes #3 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" }
attributes #4 = { convergent mustprogress noinline norecurse nounwind "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" }

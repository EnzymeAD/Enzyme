; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir %t
; RUN: %opt %s %loadPoseidon -passes=poseidon,poseidon-finalize \
; RUN:   -poseidon-profile-use=%S/../Inputs/ozaki_dispatch_fused \
; RUN:   -poseidon-cache=%t/cache -poseidon-enable-herbie=false \
; RUN:   -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-tau=1e-12 \
; RUN:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; RUN:   -S -o %t/out.ll 2>&1 | FileCheck --check-prefix=LOG %s
; RUN: cat %t/cache/launches_*.json | FileCheck --check-prefix=DESC %s
; RUN: FileCheck --input-file=%t/out.ll %s

; A body that stores the product into two arguments: the host dispatch writes
; it where the body first stores it, and the fissioned body reads it back from
; there for the other store.

; LOG: wrote descriptor for _Z9two_storePKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (fused)
; DESC: "_Z9two_storePKdS0_PdS1_ 2 0 1 16 0 12 16 16 16 16 16 16 0 0 0 0"

; CHECK-LABEL: define {{.*}} @preprocess_two_store_body(
; CHECK-SAME: ptr {{.*}}, ptr {{.*}}, ptr {{.*}}[[C:%.+]], ptr {{.*}}[[D:%.+]])
; CHECK: [[CADDR:%.+]] = getelementptr inbounds nuw [8 x i8], ptr [[C]]
; CHECK-NEXT: [[PRODUCT:%.+]] = load double, ptr [[CADDR]]
; CHECK-NEXT: [[DADDR:%.+]] = getelementptr inbounds nuw [8 x i8], ptr [[D]]
; CHECK-NEXT: store double [[PRODUCT]], ptr [[DADDR]]
; CHECK-NEXT: ret void

; ModuleID = 'two_store.cu'
source_filename = "two_store.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
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
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr noundef nonnull @two_store_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %5, ptr noundef %2, i32 noundef %5, ptr noundef %3)
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

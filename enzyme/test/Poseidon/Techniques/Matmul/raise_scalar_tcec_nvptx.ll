; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} %s -poseidon-cache=%t/c -poseidon-cost-budget=-1e-9 > %t/out 2>&1
; RUN: FileCheck %s < %t/out
; RUN: FileCheck --check-prefix=GUARD %s < %t/out

; A scalar FMA reduction loop is raised in-kernel: the direct and TCEC classes
; are priced from the fixture's inkernel_rel rows and the most accurate
; cost-reducing one, TCEC n=3, is materialized. The IR (and, with its metadata,
; Inputs/stale_cache/matmul_device.ll) is clang -x cuda --cuda-device-only
; --cuda-gpu-arch=sm_120 -O2 -ffp-contract=on of an extern "C" noinline device
; matmul_body(A, B, D) storing D[ty * 16 + tx] = sum over k < 16 (#pragma unroll
; 1) of A[ty * 16 + k] * B[k * 16 + tx], called from matmul_kernel(A, B, D, dD)
; through __poseidon_fp_optimize with A, B const and D, dD dup. Its profile,
; Inputs/matmul_site_profile, is from two <<<1, dim3(16, 16)>>> launches on
; A[i] = 1.0 + 0.001 * i, B[i] = 0.5 + 0.003 * (i % 97), dD = 1.0.

; CHECK: [poseidon] Found 1 matrix product(s) in preprocess_matmul_body
; CHECK-NEXT: Matmul[0]: 16x16x16 type=f64 origin=ConstantShape
; CHECK: matmul[0] pricing: baseline/MAC=5.120000e+02 executions=8192
; CHECK-NEXT: {{#}}0 wmma m16n16k16 f16/f32 compCost/MAC=3.200000e+01 rel=6.250000e-02
; CHECK-NEXT: {{#}}1 tcec n=2 wmma m16n16k16 f16/f32 compCost/MAC=4.000000e+01 rel=7.812500e-02
; CHECK-NEXT: {{#}}2 tcec n=3 wmma m16n16k16 f16/f32 compCost/MAC=4.800000e+01 rel=9.375000e-02
; CHECK: [poseidon] matmul[0] candidates (initial cost=5.120000e+02, executions=8192, globalM=16, globalN=16, globalK=0, K=16, gridCTAs=1):
; CHECK-NEXT: wmma m16n16k16 f16/f32: {{.*}}cost=-3.932160e+06
; CHECK-NEXT: tcec n=2 wmma m16n16k16 f16/f32: {{.*}}cost=-3.866624e+06
; CHECK-NEXT: tcec n=3 wmma m16n16k16 f16/f32: {{.*}}cost=-3.801088e+06
; CHECK: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; CHECK-LABEL: define {{.*}} @preprocess_matmul_body(
; CHECK-SAME: ptr {{[^,]*}}[[AARG:%[0-9]+]], ptr {{[^,]*}}[[BARG:%[0-9]+]], ptr {{[^,]*}}[[OARG:%[0-9]+]])
; CHECK: {{^}}raise.entry:
; CHECK: br i1 %{{[0-9]+}}, label %[[FILL:[^,]+]], label
; CHECK: {{^}}[[FILL]]:
; CHECK-DAG: [[B2P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_b2_f16_16x16 to ptr), i64 [[OFF:%[0-9]+]]
; CHECK-DAG: [[B1P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_b1_f16_16x16 to ptr), i64 [[OFF]]
; CHECK-DAG: [[B0P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_b0_f16_16x16 to ptr), i64 [[OFF]]
; CHECK-DAG: [[A2P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_a2_f16_16x16 to ptr), i64 [[OFF]]
; CHECK-DAG: [[A1P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_a1_f16_16x16 to ptr), i64 [[OFF]]
; CHECK-DAG: [[A0P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_a0_f16_16x16 to ptr), i64 [[OFF]]
; CHECK-DAG: [[BSRC:%[0-9]+]] = getelementptr {{.*}}ptr [[BARG]], i64 [[OFF8:%[0-9]+]]
; CHECK-DAG: [[ASRC:%[0-9]+]] = getelementptr {{.*}}ptr [[AARG]], i64 [[OFF8]]
; CHECK: [[A:%[0-9]+]] = load double, ptr [[ASRC]], align 8
; CHECK-NEXT: [[AF:%[0-9]+]] = fptrunc double [[A]] to float
; CHECK-NEXT: [[AH0:%[0-9]+]] = fptrunc float [[AF]] to half
; CHECK-NEXT: [[AH0F:%[0-9]+]] = fpext half [[AH0]] to float
; CHECK-NEXT: [[AR1:%[0-9]+]] = fsub float [[AF]], [[AH0F]]
; CHECK-NEXT: [[AS1:%[0-9]+]] = fmul float [[AR1]], 2.048000e+03
; CHECK-NEXT: [[AH1:%[0-9]+]] = fptrunc float [[AS1]] to half
; CHECK-NEXT: [[AH1F:%[0-9]+]] = fpext half [[AH1]] to float
; CHECK-NEXT: [[AR2:%[0-9]+]] = fsub float [[AS1]], [[AH1F]]
; CHECK-NEXT: [[AS2:%[0-9]+]] = fmul float [[AR2]], 2.048000e+03
; CHECK-NEXT: [[AH2:%[0-9]+]] = fptrunc float [[AS2]] to half
; CHECK-NEXT: store half [[AH0]], ptr [[A0P]], align 2
; CHECK-NEXT: store half [[AH1]], ptr [[A1P]], align 2
; CHECK-NEXT: store half [[AH2]], ptr [[A2P]], align 2
; CHECK: [[B:%[0-9]+]] = load double, ptr [[BSRC]], align 8
; CHECK: store half %{{[0-9]+}}, ptr [[B0P]], align 2
; CHECK-NEXT: store half %{{[0-9]+}}, ptr [[B1P]], align 2
; CHECK-NEXT: store half %{{[0-9]+}}, ptr [[B2P]], align 2
; CHECK: [[LA0:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_a0_f16_16x16 to ptr), i32 16)
; CHECK: [[LB0:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_b0_f16_16x16 to ptr), i32 16)
; CHECK: [[A0E:%[0-9]+]] = extractvalue {{.*}} [[LA0]], 0
; CHECK: [[B0E:%[0-9]+]] = extractvalue {{.*}} [[LB0]], 0
; CHECK: [[M00:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A0E]], {{.*}}, <2 x half> [[B0E]], {{.*}}, float 0.000000e+00, float {{.*}})
; CHECK: [[M00E:%[0-9]+]] = extractvalue {{.*}} [[M00]], 0
; CHECK-NEXT: [[D00:%[0-9]+]] = fadd float [[M00E]], 0.000000e+00
; CHECK: [[LB1:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_b1_f16_16x16 to ptr), i32 16)
; CHECK: [[B1E:%[0-9]+]] = extractvalue {{.*}} [[LB1]], 0
; CHECK: [[M01:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A0E]], {{.*}}, <2 x half> [[B1E]], {{.*}}, float 0.000000e+00, float {{.*}})
; CHECK: [[M01E:%[0-9]+]] = extractvalue {{.*}} [[M01]], 0
; CHECK: [[LB2:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_b2_f16_16x16 to ptr), i32 16)
; CHECK: [[B2E:%[0-9]+]] = extractvalue {{.*}} [[LB2]], 0
; CHECK: [[M02:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A0E]], {{.*}}, <2 x half> [[B2E]], {{.*}}, float 0.000000e+00, float {{.*}})
; CHECK: [[M02E:%[0-9]+]] = extractvalue {{.*}} [[M02]], 0
; CHECK: [[LA1:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_a1_f16_16x16 to ptr), i32 16)
; CHECK: [[A1E:%[0-9]+]] = extractvalue {{.*}} [[LA1]], 0
; CHECK: [[M10:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A1E]], {{.*}}, <2 x half> [[B0E]], {{.*}}, float [[M01E]], float {{.*}})
; CHECK: [[C1:%[0-9]+]] = extractvalue {{.*}} [[M10]], 0
; CHECK: [[M11:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A1E]], {{.*}}, <2 x half> [[B1E]], {{.*}}, float [[M02E]], float {{.*}})
; CHECK: [[M11E:%[0-9]+]] = extractvalue {{.*}} [[M11]], 0
; CHECK: [[LA2:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_a2_f16_16x16 to ptr), i32 16)
; CHECK: [[A2E:%[0-9]+]] = extractvalue {{.*}} [[LA2]], 0
; CHECK: [[M20:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A2E]], {{.*}}, <2 x half> [[B0E]], {{.*}}, float [[M11E]], float {{.*}})
; CHECK: [[C2:%[0-9]+]] = extractvalue {{.*}} [[M20]], 0
; CHECK: [[PH0:%[^ ]+]] = phi float [ [[D00]], %{{[^ ]+}} ], [ 0.000000e+00, %{{[^ ]+}} ]
; CHECK: [[PH1:%[^ ]+]] = phi float [ [[C1]], %{{[^ ]+}} ], [ 0.000000e+00, %{{[^ ]+}} ]
; CHECK: [[PH2:%[^ ]+]] = phi float [ [[C2]], %{{[^ ]+}} ], [ 0.000000e+00, %{{[^ ]+}} ]
; CHECK: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_d0_f32_16x16 to ptr), float [[PH0]], float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, i32 16)
; CHECK: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_d1_f32_16x16 to ptr), float [[PH1]], float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, i32 16)
; CHECK: call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_d2_f32_16x16 to ptr), float [[PH2]], float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, float %{{[^,]+}}, i32 16)
; CHECK: [[E0P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_d0_f32_16x16 to ptr), i64 [[EI:%[0-9]+]]
; CHECK-NEXT: [[E0:%[0-9]+]] = load float, ptr [[E0P]], align 4
; CHECK-NEXT: [[E1P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_d1_f32_16x16 to ptr), i64 [[EI]]
; CHECK-NEXT: [[E1:%[0-9]+]] = load float, ptr [[E1P]], align 4
; CHECK-NEXT: [[E1S:%[0-9]+]] = fmul float [[E1]], {{f0x3A000000|0x3F40000000000000}}
; CHECK-NEXT: [[S1:%[0-9]+]] = fadd float [[E0]], [[E1S]]
; CHECK-NEXT: [[E2P:%[0-9]+]] = getelementptr {{.*}}@__poseidon_ozaki_preprocess_matmul_body_d2_f32_16x16 to ptr), i64 [[EI]]
; CHECK-NEXT: [[E2:%[0-9]+]] = load float, ptr [[E2P]], align 4
; CHECK-NEXT: [[E2S:%[0-9]+]] = fmul float [[E2]], {{f0x34800000|0x3E90000000000000}}
; CHECK-NEXT: [[S2:%[0-9]+]] = fadd float [[S1]], [[E2S]]
; CHECK-NEXT: [[R:%[0-9]+]] = fpext float [[S2]] to double
; CHECK-NEXT: br label %[[EXIT:.+]]{{$}}

; Blocks of another shape than the profiled 16 x 16 x 1 run the loop as written.
; GUARD-LABEL: define {{.*}} @preprocess_matmul_body(
; GUARD: [[NX:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; GUARD-NEXT: [[X:%[0-9]+]] = icmp eq i32 [[NX]], 16
; GUARD-NEXT: [[NY:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
; GUARD-NEXT: [[Y:%[0-9]+]] = icmp eq i32 [[NY]], 16
; GUARD-NEXT: [[XY:%[0-9]+]] = and i1 [[X]], [[Y]]
; GUARD-NEXT: [[NZ:%[0-9]+]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
; GUARD-NEXT: [[Z:%[0-9]+]] = icmp eq i32 [[NZ]], 1
; GUARD-NEXT: [[SAME:%[0-9]+]] = and i1 [[Z]], [[XY]]
; GUARD-NEXT: br i1 [[SAME]], label %raise.entry, label %[[LOOP:.+]]{{$}}
; GUARD: {{^}}[[LOOP]]:
; GUARD: tail call double @llvm.fmuladd.f64(
; GUARD: br label %[[EXIT:.+]]{{$}}
; GUARD: {{^}}[[EXIT]]:
; GUARD: [[RES:%.+]] = phi double [ %{{[0-9]+}}, %[[LOOP]] ], [ %{{[0-9]+}}, %ozaki_store_gate.merge ]
; GUARD: store double [[RES]], ptr %{{[0-9]+}}, align 8
; GUARD-NEXT: ret void

source_filename = "tcec_matmul.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @matmul_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = shl nuw nsw i32 %5, 4
  br label %11

7:                                                ; preds = %11
  %8 = add nuw nsw i32 %6, %4
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %23, ptr %10, align 8
  ret void

11:                                               ; preds = %3, %11
  %12 = phi i32 [ 0, %3 ], [ %24, %11 ]
  %13 = phi double [ 0.000000e+00, %3 ], [ %23, %11 ]
  %14 = add nuw nsw i32 %12, %6
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %15
  %17 = load double, ptr %16, align 8
  %18 = shl nuw nsw i32 %12, 4
  %19 = add nuw nsw i32 %18, %4
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %20
  %22 = load double, ptr %21, align 8
  %23 = tail call double @llvm.fmuladd.f64(double %17, double %22, double %13)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %7, label %11
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

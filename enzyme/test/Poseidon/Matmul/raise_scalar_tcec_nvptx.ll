; REQUIRES: scev-trunc-iv-nowrap
; DEFINE: %{opt} = %opt %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-enable-pt=false -poseidon-raise-wmma \
; DEFINE:   -poseidon-ozaki-host-dispatch=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} %s -poseidon-cache=%t/c -poseidon-cost-budget=-1e-9 > %t/out 2>&1
; RUN: FileCheck %s < %t/out
; RUN: FileCheck --check-prefix=CLEAN %s < %t/out

; A scalar FMA reduction loop is raised in-kernel: the direct and TCEC classes
; are priced from the fixture's wmma_inkernel_rel rows, and the most accurate
; cost-reducing one, TCEC n=3, is materialized. IR and profile from
; Inputs/raise_tcec/tcec_matmul.cu (clang -O2 -ffp-contract=on
; --cuda-gpu-arch=sm_120).

; CHECK: [poseidon] Found 1 AbstractMatmul(s) for preprocess_matmul_body
; CHECK-NEXT: Matmul[0]: 16x16x16 a=f64 b=f64 acc=f64 d=f64 origin=ScalarLoopReduction
; CHECK: matmul[0] pricing: baseline/MAC=5.120000e+02 executions=8192
; CHECK-NEXT: {{#}}0 wmma m16n16k16 f16/f32 compCost/MAC=3.200000e+01 rel=6.250000e-02
; CHECK-NEXT: {{#}}1 tcec n=2 wmma m16n16k16 f16/f32 compCost/MAC=4.000000e+01 rel=7.812500e-02
; CHECK-NEXT: {{#}}2 tcec n=3 wmma m16n16k16 f16/f32 compCost/MAC=4.800000e+01 rel=9.375000e-02
; CHECK: Matmul[0] candidates (initial cost=5.120000e+02, executions=8192, globalM=16, globalN=16, globalK=0, K=16, gridCTAs=1):
; CHECK-NEXT: wmma m16n16k16 f16/f32: {{.*}}cost=-3.932160e+06
; CHECK-NEXT: tcec n=2 wmma m16n16k16 f16/f32: {{.*}}cost=-3.866624e+06
; CHECK-NEXT: tcec n=3 wmma m16n16k16 f16/f32: {{.*}}cost=-3.801088e+06
; CHECK: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; CHECK-LABEL: define {{.*}} @preprocess_matmul_body(
; CHECK-SAME: ptr {{[^,]*}}[[AARG:%[0-9]+]], ptr {{[^,]*}}[[BARG:%[0-9]+]], ptr {{[^,]*}}[[OARG:%[0-9]+]])
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
; CHECK-NEXT: [[BF:%[0-9]+]] = fptrunc double [[B]] to float
; CHECK-NEXT: [[BH0:%[0-9]+]] = fptrunc float [[BF]] to half
; CHECK-NEXT: [[BH0F:%[0-9]+]] = fpext half [[BH0]] to float
; CHECK-NEXT: [[BR1:%[0-9]+]] = fsub float [[BF]], [[BH0F]]
; CHECK-NEXT: [[BS1:%[0-9]+]] = fmul float [[BR1]], 2.048000e+03
; CHECK-NEXT: [[BH1:%[0-9]+]] = fptrunc float [[BS1]] to half
; CHECK-NEXT: [[BH1F:%[0-9]+]] = fpext half [[BH1]] to float
; CHECK-NEXT: [[BR2:%[0-9]+]] = fsub float [[BS1]], [[BH1F]]
; CHECK-NEXT: [[BS2:%[0-9]+]] = fmul float [[BR2]], 2.048000e+03
; CHECK-NEXT: [[BH2:%[0-9]+]] = fptrunc float [[BS2]] to half
; CHECK-NEXT: store half [[BH0]], ptr [[B0P]], align 2
; CHECK-NEXT: store half [[BH1]], ptr [[B1P]], align 2
; CHECK-NEXT: store half [[BH2]], ptr [[B2P]], align 2
; CHECK: [[LA0:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_a0_f16_16x16 to ptr), i32 16)
; CHECK: [[LB0:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_b0_f16_16x16 to ptr), i32 16)
; CHECK: [[A0E:%[0-9]+]] = extractvalue {{.*}} [[LA0]], 0
; CHECK: [[B0E:%[0-9]+]] = extractvalue {{.*}} [[LB0]], 0
; CHECK: [[M00:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A0E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> [[B0E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00)
; CHECK: [[M00E:%[0-9]+]] = extractvalue {{.*}} [[M00]], 0
; CHECK-NEXT: [[D00:%[0-9]+]] = fadd float [[M00E]], 0.000000e+00
; CHECK: [[LB1:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_b1_f16_16x16 to ptr), i32 16)
; CHECK: [[B1E:%[0-9]+]] = extractvalue {{.*}} [[LB1]], 0
; CHECK: [[M01:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A0E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> [[B1E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00)
; CHECK: [[M01E:%[0-9]+]] = extractvalue {{.*}} [[M01]], 0
; CHECK: [[LB2:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_b2_f16_16x16 to ptr), i32 16)
; CHECK: [[B2E:%[0-9]+]] = extractvalue {{.*}} [[LB2]], 0
; CHECK: [[M02:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A0E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> [[B2E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00)
; CHECK: [[M02E:%[0-9]+]] = extractvalue {{.*}} [[M02]], 0
; CHECK: [[LA1:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_a1_f16_16x16 to ptr), i32 16)
; CHECK: [[A1E:%[0-9]+]] = extractvalue {{.*}} [[LA1]], 0
; CHECK: [[M10:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A1E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> [[B0E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, float [[M01E]], float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}})
; CHECK: [[C1:%[0-9]+]] = extractvalue {{.*}} [[M10]], 0
; CHECK: [[M11:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A1E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> [[B1E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, float [[M02E]], float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}})
; CHECK: [[M11E:%[0-9]+]] = extractvalue {{.*}} [[M11]], 0
; CHECK: [[LA2:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr addrspacecast (ptr addrspace(3) @__poseidon_ozaki_preprocess_matmul_body_a2_f16_16x16 to ptr), i32 16)
; CHECK: [[A2E:%[0-9]+]] = extractvalue {{.*}} [[LA2]], 0
; CHECK: [[M20:%[0-9]+]] = {{.*}}call {{.*}} @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> [[A2E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> [[B0E]], <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, <2 x half> %{{[0-9]+}}, float [[M11E]], float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}})
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
; CHECK-NEXT: [[OP:%[0-9]+]] = getelementptr {{.*}}ptr [[OARG]], i64 [[EI]]
; CHECK-NEXT: store double [[R]], ptr [[OP]], align 8
; CHECK-NEXT: ret void

; CLEAN-LABEL: define {{.*}} @preprocess_matmul_body(
; CLEAN-NOT: call double @llvm.fmuladd.f64
; CLEAN: ret void

; A tolerance on the marker (poseidon_tau), with no -poseidon-tau and no cost
; budget, selects as the flag does at the same value. sed makes the marker
; declaration variadic and appends the tolerance pair the front end emits for
; `poseidon_tau, <value>`.
; RUN: sed -e 's/ptr noundef, ptr noundef) local_unnamed_addr #3/ptr noundef, ptr noundef, ...) local_unnamed_addr #3/' -e 's/tail call void @_Z22__poseidon_fp_optimize\([^(]*\)(\(.*\)) #5/tail call void (ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ...) @_Z22__poseidon_fp_optimize\1(\2, metadata !"poseidon_tau", double 1.000000e-04) #5/' %s > %t/tau4.ll
; RUN: %{opt} %t/tau4.ll -poseidon-cache=%t/c4 2>&1 | FileCheck --check-prefix=SITE4 %s
; SITE4: Poseidon error-budget: matmul -> tcec n=2 wmma m16n16k16 f16/f32 (domain err 1.572576e-07 <= 1.000000e-04 at confidence 9.500000e-01)
; SITE4: Applying solution for matmul[0] -> tcec n=2 wmma m16n16k16 f16/f32 (#1)
;
; RUN: sed -e 's/ptr noundef, ptr noundef) local_unnamed_addr #3/ptr noundef, ptr noundef, ...) local_unnamed_addr #3/' -e 's/tail call void @_Z22__poseidon_fp_optimize\([^(]*\)(\(.*\)) #5/tail call void (ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ...) @_Z22__poseidon_fp_optimize\1(\2, metadata !"poseidon_tau", double 1.000000e-07) #5/' %s > %t/tau7.ll
; RUN: %{opt} %t/tau7.ll -poseidon-cache=%t/c7 2>&1 | FileCheck --check-prefix=SITE7 %s
; SITE7: Poseidon error-budget: matmul -> tcec n=3 wmma m16n16k16 f16/f32 (domain err 5.874975e-08 <= 1.000000e-07 at confidence 9.500000e-01)
; SITE7: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)

; A tolerance nothing on the frontier meets refuses the site and keeps FP64.
; RUN: sed -e 's/ptr noundef, ptr noundef) local_unnamed_addr #3/ptr noundef, ptr noundef, ...) local_unnamed_addr #3/' -e 's/tail call void @_Z22__poseidon_fp_optimize\([^(]*\)(\(.*\)) #5/tail call void (ptr, i32, ptr, i32, ptr, i32, ptr, ptr, ...) @_Z22__poseidon_fp_optimize\1(\2, metadata !"poseidon_tau", double 1.000000e-08) #5/' %s > %t/tau8.ll
; RUN: %{opt} %t/tau8.ll -poseidon-cache=%t/c8 2>&1 | FileCheck --check-prefix=SITE8 %s
; SITE8: Poseidon error-budget: matmul -> baseline F64 (no rewrite clears 1.000000e-08 at confidence 9.500000e-01 faster than F64)
; SITE8-NOT: Applying solution for matmul[0]
; SITE8: no rewrite applied for preprocess_matmul_body; calling original body matmul_body (bit-transparent no-op)
; SITE8-NOT: Applying solution for matmul[0]

; ModuleID = 'tcec_matmul.cu'
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

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}

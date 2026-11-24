; RUN: rm -rf %t && mkdir %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/c -poseidon-cost-budget=-1e-9 -poseidon-print -S > %t/out 2>&1
; RUN: FileCheck %s < %t/out
; RUN: FileCheck --check-prefix=CLEAN %s < %t/out
; RUN: mkdir -p %t/noscalar && grep -v '^# scalar_types=' %cost_models/cm_sm_120_synthetic.csv > %t/noscalar/cm.csv
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-cost-model=%t/noscalar/cm.csv -poseidon-cache=%t/noscalar -S 2>&1 | FileCheck --check-prefix=NOSCALAR %s
; Double-single (two-component FP32 expansion) materialization of an fmuladd chain on NVPTX; IR and profile from Inputs/ds_expansion/ds_fms.cu (clang -O2 -ffp-contract=on --cuda-gpu-arch=sm_120, profiled on an sm_120 GPU).

; ModuleID = 'ds_fms.cu'
source_filename = "ds_fms.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @fms_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = fneg double %13
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %15)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double %14)
  %18 = fneg double %11
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %18)
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %16, double %17)
  %21 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %20, ptr %21, align 8
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z10fms_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @fms_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}

; CHECK: Candidates:
; CHECK: All FP64(0%) + FP32(100%)
; CHECK: All FP64(0%) + Expansion2(100%)
; CHECK: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; CHECK-LABEL: define {{.*}} @preprocess_fms_body(
; CHECK-SAME: ptr {{[^,]*}}%[[XARG:[0-9]+]], ptr {{[^,]*}}%[[YARG:[0-9]+]], ptr {{[^,]*}}%[[OARG:[0-9]+]])
; CHECK: %[[IDX:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; CHECK-NEXT: %[[XP:[0-9]+]] = getelementptr {{.*}}ptr %[[XARG]], i64 %[[IDX]]
; CHECK-NEXT: %[[X:[0-9]+]] = load double, ptr %[[XP]], align 8
; CHECK-NEXT: %[[YP:[0-9]+]] = getelementptr {{.*}}ptr %[[YARG]], i64 %[[IDX]]
; CHECK-NEXT: %[[Y:[0-9]+]] = load double, ptr %[[YP]], align 8
; CHECK-NEXT: %[[XHI:ds.hi[0-9]*]] = fptrunc double %[[X]] to float
; CHECK-NEXT: %[[XHIB:ds.hib[0-9]*]] = fpext float %[[XHI]] to double
; CHECK-NEXT: %[[XLOS:ds.los[0-9]*]] = fsub double %[[X]], %[[XHIB]]
; CHECK-NEXT: %[[XLO:ds.lo[0-9]*]] = fptrunc double %[[XLOS]] to float
; CHECK-NEXT: %[[YHI:ds.hi[0-9]*]] = fptrunc double %[[Y]] to float
; CHECK-NEXT: %[[YHIB:ds.hib[0-9]*]] = fpext float %[[YHI]] to double
; CHECK-NEXT: %[[YLOS:ds.los[0-9]*]] = fsub double %[[Y]], %[[YHIB]]
; CHECK-NEXT: %[[YLO:ds.lo[0-9]*]] = fptrunc double %[[YLOS]] to float
; CHECK-NEXT: %[[P:dsf.p[0-9]*]] = fmul float %[[XHI]], %[[YHI]]
; CHECK-NEXT: %[[NP:dsf.np[0-9]*]] = fneg float %[[P]]
; CHECK-NEXT: %[[E:dsf.e[0-9]*]] = {{.*}}call float @llvm.fma.f32(float %[[XHI]], float %[[YHI]], float %[[NP]])
; CHECK-NEXT: %[[E1:dsf.e1[0-9]*]] = {{.*}}call float @llvm.fma.f32(float %[[XHI]], float %[[YLO]], float %[[E]])
; CHECK-NEXT: %[[E2:dsf.e2[0-9]*]] = {{.*}}call float @llvm.fma.f32(float %[[XLO]], float %[[YHI]], float %[[E1]])
; CHECK-NEXT: %[[S:ts.s[0-9]*]] = fadd float %[[P]], %[[XHI]]
; CHECK-NEXT: %[[AP:ts.ap[0-9]*]] = fsub float %[[S]], %[[XHI]]
; CHECK-NEXT: %[[BP:ts.bp[0-9]*]] = fsub float %[[S]], %[[AP]]
; CHECK-NEXT: %[[DA:ts.da[0-9]*]] = fsub float %[[P]], %[[AP]]
; CHECK-NEXT: %[[DB:ts.db[0-9]*]] = fsub float %[[XHI]], %[[BP]]
; CHECK-NEXT: %[[TE:ts.e[0-9]*]] = fadd float %[[DA]], %[[DB]]
; CHECK-NEXT: %[[T0:dsf.t0[0-9]*]] = fadd float %[[E2]], %[[XLO]]
; CHECK-NEXT: %[[T1:dsf.t1[0-9]*]] = fadd float %[[T0]], %[[TE]]
; CHECK-NEXT: %[[FS:fts.s[0-9]*]] = fadd float %[[S]], %[[T1]]
; CHECK-NEXT: %[[FBP:fts.bp[0-9]*]] = fsub float %[[FS]], %[[S]]
; CHECK-NEXT: %[[FE:fts.e[0-9]*]] = fsub float %[[T1]], %[[FBP]]
; CHECK-NEXT: %[[NYHI:[^ ]+]] = fneg float %[[YHI]]
; CHECK-NEXT: %[[P2:[^ ]+]] = fmul float %[[FS]], %[[FS]]
; CHECK-NEXT: %[[NP2:[^ ]+]] = fneg float %[[P2]]
; CHECK-NEXT: %[[E2A:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FS]], float %[[FS]], float %[[NP2]])
; CHECK-NEXT: %[[E2B:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FS]], float %[[FE]], float %[[E2A]])
; CHECK-NEXT: %[[E2C:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FE]], float %[[FS]], float %[[E2B]])
; CHECK-NEXT: %[[S2:[^ ]+]] = fsub float %[[P2]], %[[YHI]]
; CHECK-NEXT: %[[AP2:[^ ]+]] = fadd float %[[S2]], %[[YHI]]
; CHECK-NEXT: %[[BP2:[^ ]+]] = fsub float %[[S2]], %[[AP2]]
; CHECK-NEXT: %[[DA2:[^ ]+]] = fsub float %[[P2]], %[[AP2]]
; CHECK-NEXT: %[[DB2:[^ ]+]] = fsub float %[[NYHI]], %[[BP2]]
; CHECK-NEXT: %[[TE2:[^ ]+]] = fadd float %[[DA2]], %[[DB2]]
; CHECK-NEXT: %[[T02:[^ ]+]] = fsub float %[[E2C]], %[[YLO]]
; CHECK-NEXT: %[[T12:[^ ]+]] = fadd float %[[T02]], %[[TE2]]
; CHECK-NEXT: %[[FS2:[^ ]+]] = fadd float %[[S2]], %[[T12]]
; CHECK-NEXT: %[[D2:[^ ]+]] = fsub float %[[S2]], %[[FS2]]
; CHECK-NEXT: %[[FE2:[^ ]+]] = fadd float %[[T12]], %[[D2]]
; CHECK-NEXT: %[[P3:[^ ]+]] = fmul float %[[FS2]], %[[YHI]]
; CHECK-NEXT: %[[NP3:[^ ]+]] = fneg float %[[P3]]
; CHECK-NEXT: %[[E3A:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FS2]], float %[[YHI]], float %[[NP3]])
; CHECK-NEXT: %[[E3B:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FS2]], float %[[YLO]], float %[[E3A]])
; CHECK-NEXT: %[[E3C:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FE2]], float %[[YHI]], float %[[E3B]])
; CHECK-NEXT: %[[S3:[^ ]+]] = fadd float %[[FS]], %[[P3]]
; CHECK-NEXT: %[[AP3:[^ ]+]] = fsub float %[[S3]], %[[FS]]
; CHECK-NEXT: %[[BP3:[^ ]+]] = fsub float %[[S3]], %[[AP3]]
; CHECK-NEXT: %[[DA3:[^ ]+]] = fsub float %[[P3]], %[[AP3]]
; CHECK-NEXT: %[[DB3:[^ ]+]] = fsub float %[[FS]], %[[BP3]]
; CHECK-NEXT: %[[TE3:[^ ]+]] = fadd float %[[DA3]], %[[DB3]]
; CHECK-NEXT: %[[T03:[^ ]+]] = fadd float %[[FE]], %[[E3C]]
; CHECK-NEXT: %[[T13:[^ ]+]] = fadd float %[[T03]], %[[TE3]]
; CHECK-NEXT: %[[FS3:[^ ]+]] = fadd float %[[S3]], %[[T13]]
; CHECK-NEXT: %[[FBP3:[^ ]+]] = fsub float %[[FS3]], %[[S3]]
; CHECK-NEXT: %[[FE3:[^ ]+]] = fsub float %[[T13]], %[[FBP3]]
; CHECK-NEXT: %[[NXHI:[^ ]+]] = fneg float %[[XHI]]
; CHECK-NEXT: %[[P4:[^ ]+]] = fmul float %[[FS3]], %[[FS3]]
; CHECK-NEXT: %[[NP4:[^ ]+]] = fneg float %[[P4]]
; CHECK-NEXT: %[[E4A:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FS3]], float %[[FS3]], float %[[NP4]])
; CHECK-NEXT: %[[E4B:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FS3]], float %[[FE3]], float %[[E4A]])
; CHECK-NEXT: %[[E4C:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FE3]], float %[[FS3]], float %[[E4B]])
; CHECK-NEXT: %[[S4:[^ ]+]] = fsub float %[[P4]], %[[XHI]]
; CHECK-NEXT: %[[AP4:[^ ]+]] = fadd float %[[S4]], %[[XHI]]
; CHECK-NEXT: %[[BP4:[^ ]+]] = fsub float %[[S4]], %[[AP4]]
; CHECK-NEXT: %[[DA4:[^ ]+]] = fsub float %[[P4]], %[[AP4]]
; CHECK-NEXT: %[[DB4:[^ ]+]] = fsub float %[[NXHI]], %[[BP4]]
; CHECK-NEXT: %[[TE4:[^ ]+]] = fadd float %[[DA4]], %[[DB4]]
; CHECK-NEXT: %[[T04:[^ ]+]] = fsub float %[[E4C]], %[[XLO]]
; CHECK-NEXT: %[[T14:[^ ]+]] = fadd float %[[T04]], %[[TE4]]
; CHECK-NEXT: %[[FS4:[^ ]+]] = fadd float %[[S4]], %[[T14]]
; CHECK-NEXT: %[[D4:[^ ]+]] = fsub float %[[S4]], %[[FS4]]
; CHECK-NEXT: %[[FE4:[^ ]+]] = fadd float %[[T14]], %[[D4]]
; CHECK-NEXT: %[[P5:[^ ]+]] = fmul float %[[FS2]], %[[FS4]]
; CHECK-NEXT: %[[NP5:[^ ]+]] = fneg float %[[P5]]
; CHECK-NEXT: %[[E5A:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FS4]], float %[[FS2]], float %[[NP5]])
; CHECK-NEXT: %[[E5B:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FS4]], float %[[FE2]], float %[[E5A]])
; CHECK-NEXT: %[[E5C:[^ ]+]] = {{.*}}call float @llvm.fma.f32(float %[[FE4]], float %[[FS2]], float %[[E5B]])
; CHECK-NEXT: %[[S5:[^ ]+]] = fadd float %[[FS3]], %[[P5]]
; CHECK-NEXT: %[[AP5:[^ ]+]] = fsub float %[[S5]], %[[FS3]]
; CHECK-NEXT: %[[BP5:[^ ]+]] = fsub float %[[S5]], %[[AP5]]
; CHECK-NEXT: %[[DA5:[^ ]+]] = fsub float %[[P5]], %[[AP5]]
; CHECK-NEXT: %[[DB5:[^ ]+]] = fsub float %[[FS3]], %[[BP5]]
; CHECK-NEXT: %[[TE5:[^ ]+]] = fadd float %[[DA5]], %[[DB5]]
; CHECK-NEXT: %[[T05:[^ ]+]] = fadd float %[[FE3]], %[[E5C]]
; CHECK-NEXT: %[[T15:[^ ]+]] = fadd float %[[T05]], %[[TE5]]
; CHECK-NEXT: %[[FS5:[^ ]+]] = fadd float %[[S5]], %[[T15]]
; CHECK-NEXT: %[[FBP5:[^ ]+]] = fsub float %[[FS5]], %[[S5]]
; CHECK-NEXT: %[[FE5:[^ ]+]] = fsub float %[[T15]], %[[FBP5]]
; CHECK-NEXT: %[[HI64:ds.hi64[0-9]*]] = fpext float %[[FS5]] to double
; CHECK-NEXT: %[[LO64:ds.lo64[0-9]*]] = fpext float %[[FE5]] to double
; CHECK-NEXT: %[[J:ds.f64[0-9]*]] = fadd double %[[HI64]], %[[LO64]], !poseidon.ds.join
; CHECK-NEXT: %[[OP:[0-9]+]] = getelementptr {{.*}}ptr %[[OARG]], i64 %[[IDX]]
; CHECK-NEXT: store double %[[J]], ptr %[[OP]], align 8
; CHECK-NEXT: ret void

; CLEAN-LABEL: define {{.*}} @preprocess_fms_body(
; CLEAN-NOT: call double @llvm.fmuladd.f64
; CLEAN-NOT: fmul double
; CLEAN-NOT: fneg double
; CLEAN: ret void

; A device model that does not name the formats it has hardware for stops the
; compile.
; NOSCALAR: error: <unknown>:0:0: in function preprocess_fms_body void (ptr, ptr, ptr): Poseidon: cost model {{.*}}noscalar/cm.csv carries no '# scalar_types=' header, so the formats the device has hardware for are unknown

; RUN: rm -rf %t && mkdir %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../../Inputs/expansion_two_components -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 -poseidon-precision-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/c -poseidon-cost-budget=-1e-9 -poseidon-print -S > %t/out 2>&1
; RUN: FileCheck %s < %t/out
; RUN: FileCheck --check-prefix=CLEAN %s < %t/out
; RUN: mkdir -p %t/noscalar && grep -v '^# scalar_types=' %cost_models/cm_sm_120_synthetic.csv > %t/noscalar/cm.csv
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../../Inputs/expansion_two_components -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=true -poseidon-cost-model=%t/noscalar/cm.csv -poseidon-cache=%t/noscalar -S 2>&1 | FileCheck --check-prefix=NOSCALAR %s
; Double-single (two-component FP32 expansion) materialization of an fmuladd
; chain on NVPTX. The IR is from
;   template <typename R, typename... T>
;   __device__ R __poseidon_fp_optimize(void *, T...);
;   __device__ int enzyme_const;
;   __device__ int enzyme_dup;
;   extern "C" __device__ __attribute__((noinline)) void
;   fms_body(const double *x, const double *y, double *out) {
;     unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
;     double a = x[i], b = y[i];
;     double t = a * b + a;
;     double u = t * t - b;
;     double v = u * b + t;
;     double w = v * v - a;
;     out[i] = w * u + v;
;   }
;   __global__ void fms_kernel(const double *x, const double *y, double *out,
;                              double *dout) {
;     __poseidon_fp_optimize<void>((void *)fms_body, enzyme_const, x,
;                                  enzyme_const, y, enzyme_dup, out, dout);
;   }
; (ds_fms.cu) via clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -O2
; -ffp-contract=on -S -emit-llvm. Inputs/expansion_two_components is its profile: the
; source compiled with -poseidon-profile-generate and launched three times as
; fms_kernel<<<4, 64>>> on x[i] = 0.25 + i / 512.0 and
; y[i] = 0.5 + (i % 17) / 32.0, with dout set to 1.0, on an sm_120 GPU.

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
; CHECK: Applying solution for precision change All FP64(0%) + Expansion2(100%)
; Each input is split into a float pair, the chain runs on float pairs only
; (Techniques/Precision/expansion_model_identity.cpp and expansion_reference_identity.cpp
; hold each routine to its reference bit for bit), and the last pair is joined
; into the stored double.
; CHECK-LABEL: define {{.*}} @preprocess_fms_body(
; CHECK-SAME: ptr {{[^,]*}}%[[XARG:[0-9]+]], ptr {{[^,]*}}%[[YARG:[0-9]+]], ptr {{[^,]*}}%[[OARG:[0-9]+]])
; CHECK: %[[IDX:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; CHECK-NEXT: %[[XP:[0-9]+]] = getelementptr {{.*}}ptr %[[XARG]], i64 %[[IDX]]
; CHECK-NEXT: %[[X:[0-9]+]] = load double, ptr %[[XP]], align 8
; CHECK-NEXT: %[[YP:[0-9]+]] = getelementptr {{.*}}ptr %[[YARG]], i64 %[[IDX]]
; CHECK-NEXT: %[[Y:[0-9]+]] = load double, ptr %[[YP]], align 8
; CHECK-NEXT: %[[XHI:pair.hi[0-9]*]] = fptrunc double %[[X]] to float
; CHECK-NEXT: %[[XHIB:pair.hib[0-9]*]] = fpext float %[[XHI]] to double
; CHECK-NEXT: %[[XLOS:pair.los[0-9]*]] = fsub double %[[X]], %[[XHIB]]
; CHECK-NEXT: %[[XLO:pair.lo[0-9]*]] = fptrunc double %[[XLOS]] to float
; CHECK-NEXT: %[[YHI:pair.hi[0-9]*]] = fptrunc double %[[Y]] to float
; CHECK-NEXT: %[[YHIB:pair.hib[0-9]*]] = fpext float %[[YHI]] to double
; CHECK-NEXT: %[[YLOS:pair.los[0-9]*]] = fsub double %[[Y]], %[[YHIB]]
; CHECK-NEXT: %[[YLO:pair.lo[0-9]*]] = fptrunc double %[[YLOS]] to float
; CHECK-NEXT: %[[P:dsf.p[0-9]*]] = fmul float %[[XHI]], %[[YHI]]
; CHECK-NEXT: %[[NP:dsf.np[0-9]*]] = fneg float %[[P]]
; CHECK-NEXT: %[[E:dsf.e[0-9]*]] = {{.*}}call float @llvm.fma.f32(float %[[XHI]], float %[[YHI]], float %[[NP]])
; CHECK-NOT: double
; CHECK: %[[HI64:pair.hi64[0-9]*]] = fpext float %{{[^ ]+}} to double
; CHECK-NEXT: %[[LO64:pair.lo64[0-9]*]] = fpext float %{{[^ ]+}} to double
; CHECK-NEXT: %[[J:pair.f64[0-9]*]] = fadd double %[[HI64]], %[[LO64]], !poseidon.pair.join
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

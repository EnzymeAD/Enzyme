; DEFINE: %{opt} = env FAKE_HERBIE_DIR=%S/Inputs/rewrite_after_precision %opt %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" -poseidon-early-prune -poseidon-enable-algebraic=true \
; DEFINE:   -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-enable-precision=true -S -poseidon-print
; DEFINE: %{up} = %t/upstream.ll -poseidon-profile-use=%S/../../Inputs/subgraph_split \
; DEFINE:   -poseidon-subgraph-split-uses=3 -poseidon-subgraph-split-ops=4 -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; DEFINE: %{carried} = %t/carried.ll -poseidon-profile-use=%S/Inputs/rewrite_after_precision \
; DEFINE:   -poseidon-precision-expansions=true -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv
; RUN: rm -rf %t && split-file %s %t
; RUN: %{opt} %{up} "-poseidon-pin=P0=All FP64(0%%) + FP32(100%%),A4=(fma.f64 v2 v1 (+.f64 v2 v0))" -poseidon-cache=%t/pt-first -o %t/pt-first.ll 2>&1 | FileCheck --check-prefix=UP-APPLY %s
; RUN: FileCheck --check-prefix=UP-IR %s < %t/pt-first.ll
; RUN: %{opt} %{up} "-poseidon-pin=A4=(fma.f64 v2 v1 (+.f64 v2 v0)),P0=All FP64(0%%) + FP32(100%%)" -poseidon-cache=%t/rewrite-first -o %t/rewrite-first.ll 2>&1 | FileCheck --check-prefix=UP-APPLY %s
; RUN: diff %t/pt-first.ll %t/rewrite-first.ll
; RUN: %{opt} %{carried} "-poseidon-pin=P0=All FP64(0%%) + Expansion2(100%%),A2=(-.f64 (*.f64 v1 v1) (*.f64 v2 v2))" -poseidon-cache=%t/expansion-first -o %t/expansion-first.ll 2>&1 | FileCheck --check-prefix=CARRIED-APPLY %s
; RUN: FileCheck --check-prefix=CARRIED-IR %s < %t/expansion-first.ll
; RUN: %{opt} %{carried} "-poseidon-pin=A2=(-.f64 (*.f64 v1 v1) (*.f64 v2 v2)),P0=All FP64(0%%) + Expansion2(100%%)" -poseidon-cache=%t/carried-rewrite-first -o %t/carried-rewrite-first.ll 2>&1 | FileCheck --check-prefix=CARRIED-APPLY %s
; RUN: diff %t/expansion-first.ll %t/carried-rewrite-first.ll

; A rewrite of an FP subgraph that reads a value a precision change of another
; subgraph produces: the steps apply in the solve's order, the rewrite first,
; whatever the order of the pinned items. upstream.ll (the module and profile of
; Solve/subgraphs.ll): P0 moves %t to binary32 and A4 reads it extended
; back as v2. carried.ll: P0 carries the loop accumulator as two float PHIs and
; A2 reads the joined pair as v1. Herbie's answers are canned in
; Inputs/rewrite_after_precision; carried.ll is carried.cu (an accumulator loop
; under __poseidon_fp_optimize) via clang -x cuda --cuda-device-only
; --cuda-gpu-arch=sm_120 -O2 -ffp-contract=on, profile from
; carried_kernel<<<2, 64>>> on an sm_120 GPU.

; UP-APPLY: Applying solution for (+ (* v2 v1) (+ v2 v0)) --(0)-> (fma.f64 v2 v1 (+.f64 v2 v0))
; UP-APPLY-NEXT: Applying solution for precision change All FP64(0%) + FP32(100%)

; UP-IR-LABEL: define void @preprocess_tester(
; UP-IR: %[[T:[0-9]+]] = fsub float
; UP-IR-NEXT: %[[TD:[0-9]+]] = fpext float %[[T]] to double
; UP-IR-NEXT: [[U2:%[^ ,()]+]] = fadd double %y, %[[TD]]
; UP-IR-NEXT: [[U3:%[^ ,()]+]] = fdiv double %[[TD]], %x
; UP-IR-NEXT: [[HERBIE_FMA:%[^ ,()]+]] = tail call double @llvm.fma.f64(double %[[TD]], double %x, double [[U2]])
; UP-IR-NEXT: [[R2:%[^ ,()]+]] = fmul double [[U3]], [[U2]]
; UP-IR-NEXT: store double [[HERBIE_FMA]], ptr %o1

; CARRIED-APPLY: Applying solution for (* (- v1 v2) (+ v2 v1)) --(0)-> (-.f64 (*.f64 v1 v1) (*.f64 v2 v2))
; CARRIED-APPLY-NEXT: Applying solution for precision change All FP64(0%) + Expansion2(100%) (#3)

; CARRIED-IR-LABEL: define dso_local void @preprocess_carried_body(
; CARRIED-IR-SAME: ptr {{.*}}%0, double noundef %[[V2:[0-9]+]], ptr {{.*}}%[[OUT:[0-9]+]])
; CARRIED-IR-NOT: phi double
; CARRIED-IR: %[[V2SQ:[^ ]+]] = fmul double %[[V2]], %[[V2]]
; CARRIED-IR-NOT: phi double
; CARRIED-IR: %pair.phi.hi = phi float [ %[[NHI:[^ ]+]], %{{[0-9]+}} ], [ 0.000000e+00, %{{[0-9]+}} ]
; CARRIED-IR-NEXT: %pair.phi.lo = phi float [ %[[NLO:[^ ]+]], %{{[0-9]+}} ], [ 0.000000e+00, %{{[0-9]+}} ]
; CARRIED-IR-NEXT: %[[HI:.+]] = fpext float %pair.phi.hi to double
; CARRIED-IR-NEXT: %[[LO:.+]] = fpext float %pair.phi.lo to double
; CARRIED-IR-NEXT: %[[ACC:.+]] = fadd double %[[HI]], %[[LO]]
; CARRIED-IR-NOT: phi double
; CARRIED-IR: %fpcore.mul = fmul double %[[ACC]], %[[ACC]]
; CARRIED-IR-NEXT: %fpcore.sub = fsub double %fpcore.mul, %[[V2SQ]]
; CARRIED-IR-NEXT: %[[IDX:[0-9]+]] = add {{.*}}i64
; CARRIED-IR-NEXT: %[[P:[0-9]+]] = getelementptr {{.*}}ptr %[[OUT]], i64 %[[IDX]]
; CARRIED-IR-NEXT: store double %fpcore.sub, ptr %[[P]], align 8
; CARRIED-IR: %[[PH:[^ ]+]] = fmul float %pair.phi.hi, 5.000000e-01
; CARRIED-IR: call float @llvm.fma.f32(float %pair.phi.lo, float 5.000000e-01, float
; CARRIED-IR: %[[NHI]] = fadd float %[[TS:[^ ]+]], %[[T1:[^ ]+]]
; CARRIED-IR-NEXT: %[[D:[0-9]+]] = fsub float %[[TS]], %[[NHI]]
; CARRIED-IR-NEXT: %[[NLO]] = fadd float %[[T1]], %[[D]]
; CARRIED-IR-NOT: phi double
; CARRIED-IR: {{^}}}{{$}}

;--- upstream.ll
define void @tester(double %x, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %a = fmul double %x, %y
  %b = fadd double %a, %x
  %c = fmul double %b, %b
  %t = fsub double %c, %y
  %u1 = fmul double %t, %x
  %u2 = fadd double %t, %y
  %u3 = fdiv double %t, %x
  %r1 = fadd double %u1, %u2
  %r2 = fmul double %u3, %u2
  store double %r1, ptr %o1, align 8
  store double %r2, ptr %o2, align 8
  ret void
}

define void @site(double %x, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2)
  ret void
}

declare void @__poseidon_fp_optimize(ptr, ...)
attributes #0 = { "target-cpu"="x86-64" }

;--- carried.ll
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @carried_body(ptr nofree noundef readonly captures(none) %0, double noundef %1, ptr nofree noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = shl i32 %8, 4
  br label %11

10:                                               ; preds = %11
  ret void

11:                                               ; preds = %3, %11
  %12 = phi i32 [ 0, %3 ], [ %24, %11 ]
  %13 = phi double [ 0.000000e+00, %3 ], [ %23, %11 ]
  %14 = fsub double %13, %1
  %15 = fadd double %1, %13
  %16 = fmul double %14, %15
  %17 = add nuw nsw i32 %12, %9
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %18
  store double %16, ptr %19, align 8
  %20 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %18
  %21 = load double, ptr %20, align 8
  %22 = fmul double %21, %21
  %23 = tail call double @llvm.fmuladd.f64(double %13, double 5.000000e-01, double %22)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %10, label %11
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z14carried_kernelPKddPdS1_(ptr noundef %0, double noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdidiPdS2_EET_PvDpT0_(ptr noundef nonnull @carried_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, double noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdidiPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, double noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

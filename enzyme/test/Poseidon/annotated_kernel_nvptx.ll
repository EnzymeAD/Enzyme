; RUN: rm -rf %t && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/annotated_kernel -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t -poseidon-comp-cost-budget=-1 -poseidon-print -S > %t.out 2>&1
; RUN: FileCheck %s < %t.out
; RUN: rm -rf %t2 && %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-kernels=fms.* -poseidon-profile-use=%S/Inputs/annotated_kernel -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t2 -poseidon-comp-cost-budget=-1 -poseidon-print -S > %t2.out 2>&1
; RUN: FileCheck --check-prefix=REGEX %s < %t2.out
; RUN: %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" -S 2>%t3.err | FileCheck --check-prefix=NOPROFILE %s
; RUN: FileCheck --check-prefix=NOPROFILE-WARN %s < %t3.err
; RUN: rm -rf %t4 && %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" -poseidon-profile-use=%S/Inputs/annotated_kernel -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t4 -poseidon-comp-cost-budget=0 -S | FileCheck --check-prefix=NOOP %s
; RUN: rm -rf %t5 && %opt %s %loadPoseidon -passes="poseidon,always-inline,poseidon-finalize" -poseidon-profile-use=%S/Inputs/annotated_kernel -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t5 -poseidon-comp-cost-budget=-1 -S | FileCheck --check-prefix=INLINED %s
; REQUIRES: poseidon
; A kernel carrying POSEIDON_OPTIMIZE is a site whose annotated computation is
; its whole body: same code and same profile as ds_expansion_nvptx.ll, with no
; marker call and no shadow arguments. The second run reaches the same site by
; name with -poseidon-kernels instead of the attribute.

; ModuleID = 'annotated_kernel.cu'
source_filename = "annotated_kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [20 x i8] c"annotated_kernel.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @fms_kernel, ptr @.str, ptr @.str.1, i32 2, ptr null }], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local ptx_kernel void @fms_kernel(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8, !tbaa !9
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8, !tbaa !9
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = fneg double %13
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %15)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double %14)
  %18 = fneg double %11
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %18)
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %16, double %17)
  %21 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %20, ptr %21, align 8, !tbaa !9
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git a70419505471bd8240ef3451dcdd541f8676477c)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"double", !7, i64 0}

; CHECK: Candidates:
; CHECK: All FP64(0%) + FP32(100%)
; CHECK: All FP64(0%) + Expansion2(100%)
; CHECK: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; CHECK-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; CHECK: %[[X:.+]] = load double, ptr
; CHECK: %[[Y:.+]] = load double, ptr
; CHECK: %[[XHI:ds.hi[0-9]*]] = fptrunc double %[[X]] to float
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
; CHECK: %[[HI64:ds.hi64[0-9]*]] = fpext float %{{.+}} to double
; CHECK-NEXT: %[[LO64:ds.lo64[0-9]*]] = fpext float %{{.+}} to double
; CHECK-NEXT: %[[J:ds.f64[0-9]*]] = fadd double %[[HI64]], %[[LO64]], !poseidon.ds.join
; CHECK: store double %[[J]], ptr

; CLEAN-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; CLEAN-NOT: call double @llvm.fmuladd.f64
; CLEAN-NOT: fmul double
; CLEAN-NOT: fneg double
; CLEAN: ret void

; REGEX: Applying solution for CS: All FP64(0%) + Expansion2(100%)
; REGEX-LABEL: define {{.*}} @fms_kernel(
; REGEX: call void @preprocess_fms_kernel_poseidon_body(

; The annotation alone must not change the kernel: with no profile the site is
; left as written (no outline, no call), and with a profile whose solve applies
; nothing the outline is folded back. With a rewrite applied the materialized
; body is folded back the same way. In every case the kernel is one entry
; function with the arithmetic inline and no _poseidon_body symbol survives.
; NOPROFILE-WARN: warning: {{.*}}compiled without -poseidon-profile-generate or -poseidon-profile-use
; NOPROFILE-LABEL: define {{.*}} @fms_kernel(
; NOPROFILE-NOT: call void @
; NOPROFILE: call double @llvm.fmuladd.f64
; NOPROFILE: ret void
; NOPROFILE-NOT: _poseidon_body

; NOOP-LABEL: define {{.*}} @fms_kernel(
; NOOP-NOT: call void @
; NOOP: call double @llvm.fmuladd.f64
; NOOP: ret void
; NOOP-NOT: _poseidon_body

; INLINED-LABEL: define {{.*}} @fms_kernel(
; INLINED-NOT: call void @
; INLINED: fptrunc double
; INLINED: !poseidon.ds.join
; INLINED: ret void
; INLINED-NOT: _poseidon_body

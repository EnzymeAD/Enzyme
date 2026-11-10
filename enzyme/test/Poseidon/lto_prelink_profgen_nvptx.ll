; RUN: rm -rf %t && mkdir -p %t
; RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O2 -flto -x ir %s %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/full -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=REPRODUCER --implicit-check-not="exit code" %s
; RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O2 -flto=thin -x ir %s %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/thin -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=REPRODUCER --implicit-check-not="exit code" %s
; RUN: %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O2 -flto -x ir %s %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/prelink -mllvm -enzyme-lto-prelink=1 -S -emit-llvm -o - 2>/dev/null | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_autodiff %s
; RUN: %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O0 -flto=thin -x ir %s %clangLoadPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/thin0 -S -emit-llvm -o - 2>/dev/null | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_autodiff %s
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-generate -poseidon-cache=%t/byname -S -o /dev/null 2>&1 | FileCheck --check-prefix=BYNAME %s
; REQUIRES: poseidon, enzyme
; Profile generation of a device module in an LTO pre-link pipeline, the phase
; every HIP device compile is in. Enzyme leaves derivative requests to the
; post-link run there (full LTO at any level, ThinLTO above O0) unless
; -enzyme-lto-prelink is given, and a HIP device link loads no plugin, so the
; probes' request has to be lowered in this compile: with the flag it is, and
; without it the pass says why it was not. Run by pipeline name the pass is not
; told its phase and names the deferral only as a possibility. Either way this
; is a configuration error, not a crash: the frontend exits with status 1, not
; clang's crash status, and no reproducer is written. The kernel is the one of
; annotated_kernel_nvptx.ll.

; DEFERRED: Poseidon: profile generation emitted a derivative request that nothing lowered. This is an LTO pre-link compile, where Enzyme leaves derivative requests to the post-link run unless -enzyme-lto-prelink is given; pass -mllvm -enzyme-lto-prelink=1
; DEFERRED: exit status 1

; LOWERED: define {{.*}}@diffepreprocess_fms_kernel_poseidon_body(

; BYNAME: Poseidon: profile generation emitted a derivative request that nothing lowered. Add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin. In an LTO pre-link pipeline Enzyme also leaves derivative requests to the post-link run unless -enzyme-lto-prelink is given.

source_filename = "lto_prelink_profgen.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [23 x i8] c"lto_prelink_profgen.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @fms_kernel, ptr @.str, ptr @.str.1, i32 2, ptr null }], section "llvm.metadata"

define dso_local ptx_kernel void @fms_kernel(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %6 = mul i32 %4, %5
  %7 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %9
  %11 = load double, ptr %10, align 8, !tbaa !3
  %12 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %9
  %13 = load double, ptr %12, align 8, !tbaa !3
  %14 = tail call double @llvm.fmuladd.f64(double %11, double %13, double %11)
  %15 = fneg double %13
  %16 = tail call double @llvm.fmuladd.f64(double %14, double %14, double %15)
  %17 = tail call double @llvm.fmuladd.f64(double %16, double %13, double %14)
  %18 = fneg double %11
  %19 = tail call double @llvm.fmuladd.f64(double %17, double %17, double %18)
  %20 = tail call double @llvm.fmuladd.f64(double %19, double %16, double %17)
  %21 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %20, ptr %21, align 8, !tbaa !3
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double) #1
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!4, !4, i64 0}
!4 = !{!"double", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}

; REQUIRES: enzyme, nvptx-device-profiler
; RUN: rm -rf %t && mkdir %t
; RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O2 -flto -x ir %s %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/full -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=REPRODUCER --implicit-check-not="exit code" --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE %s
; RUN: (env TMPDIR=%t %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O2 -flto=thin -x ir %s %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/thin -S -emit-llvm -o /dev/null 2>&1; echo "exit status $?") | FileCheck --check-prefix=DEFERRED --implicit-check-not=REPRODUCER --implicit-check-not="exit code" --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE %s
; RUN: %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O2 -flto -x ir %s %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/prelink -mllvm -enzyme-lto-prelink=1 -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_autodiff %s
; RUN: %clang --target=nvptx64-nvidia-cuda -march=sm_120 -O0 -flto=thin -x ir %s %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/thin0 -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=LOWERED --implicit-check-not=__enzyme_autodiff %s
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-generate -poseidon-cache=%t/byname -S -o /dev/null 2>&1 | FileCheck --check-prefix=BYNAME %s
; Profile generation of a device module in an LTO pre-link compile, the phase
; of every HIP device compile. Enzyme defers derivative requests to the
; post-link run there (full LTO at any level, ThinLTO above O0) unless
; -enzyme-lto-prelink is given, and a HIP device link loads no plugin, so the
; probes' requests must be lowered in this compile: with the flag they are;
; without it the compile stops with a configuration error that says why (exit
; status 1, not a crash; no reproducer). Run by pipeline name, the pass is not
; told its phase and names the deferral only as a possibility. The kernel is
; the one of annotated_kernel_nvptx.ll.

; DEFERRED: {{^}}error: Poseidon: profile generation emitted a derivative request that nothing lowered, which Enzyme leaves to the post-link run of an LTO pre-link compile; pass -mllvm -enzyme-lto-prelink=1 (for HIP, -Xclang -mllvm -Xclang -enzyme-lto-prelink=1){{$}}
; DEFERRED-NEXT: {{^}}1 error generated.{{$}}
; DEFERRED-NEXT: exit status 1

; The kernel calls the derivative with each pointer's shadow next to it, and the
; derivative carries the probes.
; LOWERED-LABEL: define ptx_kernel void @fms_kernel(
; LOWERED-SAME: ptr {{[^,]*}}[[A:%[0-9]+]], ptr {{[^,]*}}[[B:%[0-9]+]], ptr {{[^,]*}}[[C:%[0-9]+]], ptr {{[^,]*}}[[DA:%[0-9]+]], ptr {{[^,]*}}[[DB:%[0-9]+]], ptr {{[^,]*}}[[DC:%[0-9]+]])
; LOWERED-NEXT: call {{.*}}void @diffepreprocess_fms_kernel_poseidon_body(ptr [[A]], ptr [[DA]], ptr [[B]], ptr [[DB]], ptr [[C]], ptr [[DC]])
; LOWERED-NEXT: ret void
; LOWERED: define {{.*}}@diffepreprocess_fms_kernel_poseidon_body(
; LOWERED: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess_fms_kernel_poseidon_body to ptr), i64 0,
; LOWERED: call void @__poseidon_device_log_value(ptr addrspacecast (ptr addrspace(1) @poseidon.site.preprocess_fms_kernel_poseidon_body to ptr), i64 6,

; BYNAME: {{^}}error: Poseidon: profile generation emitted a derivative request that nothing lowered; add -fpass-plugin=<ClangEnzyme>.so after the Poseidon plugin, and in an LTO pre-link compile -mllvm -enzyme-lto-prelink=1{{$}}

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

; What poseidon.h defines in device code under POSEIDON_DEVICE_MATH.
define linkonce_odr void @__poseidon_device_math_defined() {
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

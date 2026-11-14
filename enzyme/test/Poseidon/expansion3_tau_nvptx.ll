; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/expansion3 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-print -S -poseidon-expansion-components=2 -poseidon-tau=1e-16 -poseidon-cache=%t/two 2>&1 | FileCheck --check-prefix=TWO %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/expansion3 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-print -S -poseidon-expansion-components=3 -poseidon-tau=1e-16 -poseidon-cache=%t/three 2>&1 | FileCheck --check-prefix=THREE %s
; RUN: %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/expansion3 -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-print -S -poseidon-expansion-components=3 -poseidon-tau=1e-15 -poseidon-cache=%t/loose 2>&1 | FileCheck --check-prefix=LOOSE %s
; REQUIRES: poseidon
; Three-component FP32 expansion (-poseidon-expansion-components=3, the N-limb arm of Expansion.cpp). A
; 24-step logistic map (72 FP64 operations on one input) under the site tolerance 1e-16: the two-component
; expansion models 4.2e-16 and misses it, so with two components the cheapest point that clears it is the
; FP64 body itself and nothing is rewritten; admitting three components adds an Expansion3 candidate that
; clears it and is cheaper than FP64 on this cost model, and that is the pick. At 1e-15 the cheaper
; two-component point clears the tolerance and wins even with three admitted. The materialized body
; splits the FP64 input into three FP32 components (mfx.split*), computes in FP32, and restores FP64
; as the sum of three widened components (mfx.tof64); the double-single join tag is not used.
; IR: Inputs/expansion3/logistic.cu via clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -O2
; -ffp-contract=on -I<enzyme>/include -S -emit-llvm (LLVM 24; the LLVM-24-only nocreateundeforpoison
; intrinsic attribute removed). Profile: the same source through poseidon-clang++
; -poseidon-profile-generate, run on an RTX 5090.

; ModuleID = 'logistic.cu'
source_filename = "logistic.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [12 x i8] c"logistic.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z15logistic_kernelPKdPd, ptr @.str, ptr @.str1, i32 5, ptr null }], section "llvm.metadata"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local ptx_kernel void @_Z15logistic_kernelPKdPd(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef writeonly captures(none) %1) #0 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = mul i32 %3, %4
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = add i32 %5, %6
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %8
  %10 = load double, ptr %9, align 8, !tbaa !10
  %11 = fmul double %10, 3.900000e+00
  %12 = fsub double 1.000000e+00, %10
  %13 = fmul double %11, %12
  %14 = fmul double %13, 3.900000e+00
  %15 = fsub double 1.000000e+00, %13
  %16 = fmul double %14, %15
  %17 = fmul double %16, 3.900000e+00
  %18 = fsub double 1.000000e+00, %16
  %19 = fmul double %17, %18
  %20 = fmul double %19, 3.900000e+00
  %21 = fsub double 1.000000e+00, %19
  %22 = fmul double %20, %21
  %23 = fmul double %22, 3.900000e+00
  %24 = fsub double 1.000000e+00, %22
  %25 = fmul double %23, %24
  %26 = fmul double %25, 3.900000e+00
  %27 = fsub double 1.000000e+00, %25
  %28 = fmul double %26, %27
  %29 = fmul double %28, 3.900000e+00
  %30 = fsub double 1.000000e+00, %28
  %31 = fmul double %29, %30
  %32 = fmul double %31, 3.900000e+00
  %33 = fsub double 1.000000e+00, %31
  %34 = fmul double %32, %33
  %35 = fmul double %34, 3.900000e+00
  %36 = fsub double 1.000000e+00, %34
  %37 = fmul double %35, %36
  %38 = fmul double %37, 3.900000e+00
  %39 = fsub double 1.000000e+00, %37
  %40 = fmul double %38, %39
  %41 = fmul double %40, 3.900000e+00
  %42 = fsub double 1.000000e+00, %40
  %43 = fmul double %41, %42
  %44 = fmul double %43, 3.900000e+00
  %45 = fsub double 1.000000e+00, %43
  %46 = fmul double %44, %45
  %47 = fmul double %46, 3.900000e+00
  %48 = fsub double 1.000000e+00, %46
  %49 = fmul double %47, %48
  %50 = fmul double %49, 3.900000e+00
  %51 = fsub double 1.000000e+00, %49
  %52 = fmul double %50, %51
  %53 = fmul double %52, 3.900000e+00
  %54 = fsub double 1.000000e+00, %52
  %55 = fmul double %53, %54
  %56 = fmul double %55, 3.900000e+00
  %57 = fsub double 1.000000e+00, %55
  %58 = fmul double %56, %57
  %59 = fmul double %58, 3.900000e+00
  %60 = fsub double 1.000000e+00, %58
  %61 = fmul double %59, %60
  %62 = fmul double %61, 3.900000e+00
  %63 = fsub double 1.000000e+00, %61
  %64 = fmul double %62, %63
  %65 = fmul double %64, 3.900000e+00
  %66 = fsub double 1.000000e+00, %64
  %67 = fmul double %65, %66
  %68 = fmul double %67, 3.900000e+00
  %69 = fsub double 1.000000e+00, %67
  %70 = fmul double %68, %69
  %71 = fmul double %70, 3.900000e+00
  %72 = fsub double 1.000000e+00, %70
  %73 = fmul double %71, %72
  %74 = fmul double %73, 3.900000e+00
  %75 = fsub double 1.000000e+00, %73
  %76 = fmul double %74, %75
  %77 = fmul double %76, 3.900000e+00
  %78 = fsub double 1.000000e+00, %76
  %79 = fmul double %77, %78
  %80 = fmul double %79, 3.900000e+00
  %81 = fsub double 1.000000e+00, %79
  %82 = fmul double %80, %81
  %83 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %8
  store double %82, ptr %83, align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3, !4}
!llvm.errno.tbaa = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 24.0.0git (https://github.com/llvm/llvm-project.git e56c2cefc3e7978de4a2d799fa3abec77d75af21)"}
!4 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!5 = !{!6, !7, i64 0}
!6 = !{!"__libc_errno", !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}

; TWO: 2.013455e-07		-868192474		All FP64(0%) + Expansion2(100%)
; TWO-NOT: Expansion3
; TWO: tau=1.000000e-16 S=4.913424e+08 A0=6.201450e-09 selected cost=0 accCost=0.000000e+00 rel=1.262144e-17
; TWO: no rewrite applied for preprocess__Z15logistic_kernelPKdPd_poseidon_body
; TWO-NOT: Applying solution

; THREE: 2.013455e-07		-868192474		All FP64(0%) + Expansion2(100%)
; THREE: 0.000000e+00		-796398147		All FP64(0%) + Expansion3(100%)
; THREE: tau=1.000000e-16 S=4.913424e+08 A0=6.201450e-09 selected cost=-796398147 accCost=0.000000e+00 rel=1.262144e-17
; THREE: Applying solution for CS: All FP64(0%) + Expansion3(100%) (#3)
; THREE-LABEL: define internal void @preprocess__Z15logistic_kernelPKdPd_poseidon_body(
; THREE: %mfx.split = fptrunc double %{{.+}} to float
; THREE-NEXT: %mfx.splitb = fpext float %mfx.split to double
; THREE-NEXT: %mfx.splitr = fsub double %{{.+}}, %mfx.splitb
; THREE-NEXT: %mfx.split1 = fptrunc double %mfx.splitr to float
; THREE-NEXT: %mfx.splitb2 = fpext float %mfx.split1 to double
; THREE-NEXT: %mfx.splitr3 = fsub double %mfx.splitr, %mfx.splitb2
; THREE-NEXT: %mfx.split4 = fptrunc double %mfx.splitr3 to float
; THREE-NOT: fmul double
; THREE-NOT: poseidon.ds.join
; THREE: %mfx.e = fpext float %{{.+}} to double
; THREE-NEXT: %mfx.e{{[0-9]+}} = fpext float %{{.+}} to double
; THREE-NEXT: %mfx.tof64 = fadd double
; THREE-NEXT: %mfx.e{{[0-9]+}} = fpext float %{{.+}} to double
; THREE-NEXT: %[[OUT:mfx.tof64[0-9]+]] = fadd double %mfx.tof64,
; THREE-NEXT: getelementptr
; THREE-NEXT: store double %[[OUT]], ptr
; THREE-NEXT: ret void

; LOOSE: 0.000000e+00		-796398147		All FP64(0%) + Expansion3(100%)
; LOOSE: tau=1.000000e-15 S=4.913424e+08 A0=6.201450e-09 selected cost=-868192474 accCost=2.013455e-07 rel=4.224080e-16
; LOOSE: Applying solution for CS: All FP64(0%) + Expansion2(100%) (#1)
; LOOSE-LABEL: define internal void @preprocess__Z15logistic_kernelPKdPd_poseidon_body(
; LOOSE: !poseidon.ds.join
; LOOSE-NOT: mfx.split

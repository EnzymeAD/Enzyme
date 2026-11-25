; DEFINE: %{opt} = %opt %s %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/expansion3 -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-enable-pt=true -poseidon-enable-multifloat=true \
; DEFINE:   -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-expansion-components=2 -poseidon-tau=1e-16 -poseidon-cache=%t/two 2>&1 \
; RUN:   | FileCheck --check-prefix=TWO --implicit-check-not="Applying solution" %s
; RUN: %{opt} -poseidon-expansion-components=3 -poseidon-tau=1e-16 -poseidon-cache=%t/three 2>&1 \
; RUN:   | FileCheck --check-prefix=THREE --implicit-check-not=poseidon.ds.join %s
; RUN: %{opt} -poseidon-expansion-components=3 -poseidon-tau=1e-15 -poseidon-cache=%t/loose 2>&1 \
; RUN:   | FileCheck --check-prefix=LOOSE --implicit-check-not=mfx. %s

; TWO: 2.013455e-07		-8.835072e+06		All FP64(0%) + Expansion2(100%)
; TWO-NOT: Expansion3
; TWO: tau=1.000000e-16 S=4.913424e+08 A0=6.201450e-09 selected cost=0.000000e+00 accCost=0.000000e+00 rel=1.262144e-17
; TWO: no rewrite applied for preprocess__Z15logistic_kernelPKdPd_poseidon_body; calling original body _Z15logistic_kernelPKdPd_poseidon_body (bit-transparent no-op)
; TWO-NOT: Applying solution
; TWO-LABEL: define dso_local ptx_kernel void @_Z15logistic_kernelPKdPd(
; TWO-NEXT: entry:
; TWO-NEXT: call void @_Z15logistic_kernelPKdPd_poseidon_body(ptr %0, ptr %1)
; TWO-NEXT: ret void

; THREE: 2.013455e-07		-8.835072e+06		All FP64(0%) + Expansion2(100%)
; THREE: -6.201441e-09		-8.142848e+06		All FP64(0%) + Expansion3(100%)
; THREE: tau=1.000000e-16 S=4.913424e+08 A0=6.201450e-09 selected cost=-8.628472e-01 accCost=-6.201441e-09 rel=1.869795e-23
; THREE: Applying solution for precision change All FP64(0%) + Expansion3(100%) (#3)
; THREE-LABEL: define internal void @preprocess__Z15logistic_kernelPKdPd_poseidon_body(
; THREE: %[[IDX:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; THREE-NEXT: %[[IN:[0-9]+]] = getelementptr {{.*}}ptr %0, i64 %[[IDX]]
; THREE-NEXT: %[[X:[0-9]+]] = load double, ptr %[[IN]], align 8
; THREE-NEXT: [[MFX_SPLIT:%[^ ,()]+]] = fptrunc double %[[X]] to float
; THREE-NEXT: [[MFX_SPLITB:%[^ ,()]+]] = fpext float [[MFX_SPLIT]] to double
; THREE-NEXT: [[MFX_SPLITR:%[^ ,()]+]] = fsub double %[[X]], [[MFX_SPLITB]]
; THREE-NEXT: [[MFX_SPLIT1:%[^ ,()]+]] = fptrunc double [[MFX_SPLITR]] to float
; THREE-NEXT: [[MFX_SPLITB2:%[^ ,()]+]] = fpext float [[MFX_SPLIT1]] to double
; THREE-NEXT: [[MFX_SPLITR3:%[^ ,()]+]] = fsub double [[MFX_SPLITR]], [[MFX_SPLITB2]]
; THREE-NEXT: [[MFX_SPLIT4:%[^ ,()]+]] = fptrunc double [[MFX_SPLITR3]] to float
; THREE-NEXT: [[TP_P:%[^ ,()]+]] = fmul float [[MFX_SPLIT]], [[K39:3\.900000e\+00|0x400F333340000000]]
; THREE-NEXT: [[TP_NP:%[^ ,()]+]] = fneg float [[TP_P]]
; THREE-NEXT: [[TP_E:%[^ ,()]+]] = {{.*}}call float @llvm.fma.f32(float [[MFX_SPLIT]], float [[K39]], float [[TP_NP]])
; THREE-NOT: double
; THREE: call float @llvm.fma.f32(float [[MFX_SPLIT1]], float [[K39]],
; THREE-NOT: double
; THREE: call float @llvm.fma.f32(float [[MFX_SPLIT4]], float [[K39]],
; THREE-NOT: double
; THREE-NOT: fmul double
; THREE-NOT: poseidon.ds.join
; THREE: [[MFX_E:%[^ ,()]+]] = fpext float [[C2:%[^ ,()]+]] to double
; THREE-NEXT: [[E1:%[^ ,()]+]] = fpext float [[C1:%[^ ,()]+]] to double
; THREE-NEXT: [[MFX_TOF64:%[^ ,()]+]] = fadd double [[E1]], [[MFX_E]]
; THREE-NEXT: [[E2:%[^ ,()]+]] = fpext float [[C0:%[^ ,()]+]] to double
; THREE-NEXT: [[OUT:%[^ ,()]+]] = fadd double [[MFX_TOF64]], [[E2]]
; THREE-NEXT: %[[OP:[0-9]+]] = getelementptr {{.*}}ptr %1, i64 %[[IDX]]
; THREE-NEXT: store double [[OUT]], ptr %[[OP]], align 8
; THREE-NEXT: ret void

; LOOSE: -6.201441e-09		-8.142848e+06		All FP64(0%) + Expansion3(100%)
; LOOSE: tau=1.000000e-15 S=4.913424e+08 A0=6.201450e-09 selected cost=-9.361979e-01 accCost=2.013455e-07 rel=4.224080e-16
; LOOSE: Applying solution for precision change All FP64(0%) + Expansion2(100%) (#1)
; LOOSE-LABEL: define internal void @preprocess__Z15logistic_kernelPKdPd_poseidon_body(
; LOOSE: %[[IDX:[0-9]+]] = zext i32 %{{[0-9]+}} to i64
; LOOSE-NEXT: %[[IN:[0-9]+]] = getelementptr {{.*}}ptr %0, i64 %[[IDX]]
; LOOSE-NEXT: %[[X:[0-9]+]] = load double, ptr %[[IN]], align 8
; LOOSE-NEXT: [[DS_HI:%[^ ,()]+]] = fptrunc double %[[X]] to float
; LOOSE-NEXT: [[DS_HIB:%[^ ,()]+]] = fpext float [[DS_HI]] to double
; LOOSE-NEXT: [[DS_LOS:%[^ ,()]+]] = fsub double %[[X]], [[DS_HIB]]
; LOOSE-NEXT: [[DS_LO:%[^ ,()]+]] = fptrunc double [[DS_LOS]] to float
; LOOSE-NEXT: [[TP_P:%[^ ,()]+]] = fmul float [[DS_HI]], [[K39:3\.900000e\+00|0x400F333340000000]]
; LOOSE-NOT: double
; LOOSE: [[DS_HI64:%[^ ,()]+]] = fpext float %[[HI:[^ ]+]] to double
; LOOSE-NEXT: [[DS_LO64:%[^ ,()]+]] = fpext float %[[LO:[^ ]+]] to double
; LOOSE-NEXT: [[DS_F64:%[^ ,()]+]] = fadd double [[DS_HI64]], [[DS_LO64]], !poseidon.ds.join
; LOOSE-NEXT: %[[OP:[0-9]+]] = getelementptr {{.*}}ptr %1, i64 %[[IDX]]
; LOOSE-NEXT: store double [[DS_F64]], ptr %[[OP]], align 8
; LOOSE-NEXT: ret void

; -poseidon-expansion-components=3 admits a three-component FP32 expansion
; (Techniques/Expansion.cpp). On a 24-step logistic map (72 FP64 operations on
; one input) at tolerance 1e-16 the two-component expansion misses, so with two
; components the body is left as written; with three, the three-component
; expansion clears it, is cheaper than FP64 on this cost model, and models an
; error below the FP64 body's (a negative accuracy cost). At 1e-15 the
; two-component point clears and wins with three admitted. The three-component
; body splits the input into three FP32 components and sums them back to FP64,
; without the double-single join tag.
; The IR is from
;   POSEIDON_OPTIMIZE __global__ void logistic_kernel(const double *x0,
;                                                     double *out) {
;     unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
;     double x = x0[i];
;   #pragma unroll
;     for (int k = 0; k < 24; ++k)
;       x = 3.9 * x * (1.0 - x);
;     out[i] = x;
;   }
; (logistic.cu) via clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_120 -O2
; -ffp-contract=on -I<enzyme>/include -S -emit-llvm (LLVM 24; the LLVM-24-only
; nocreateundeforpoison intrinsic attribute removed). Profile: the same source
; through poseidon-clang++ -poseidon-profile-generate, launched as
; logistic_kernel<<<4, 64>>> on x0[i] = 0.2 + 0.6 * i / 256 on an sm_120 GPU.

; ModuleID = 'logistic.cu'
source_filename = "logistic.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [12 x i8] c"logistic.cu\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @_Z15logistic_kernelPKdPd, ptr @.str, ptr @.str1, i32 5, ptr null }], section "llvm.metadata"

define dso_local ptx_kernel void @_Z15logistic_kernelPKdPd(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef writeonly captures(none) %1) #0 {
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %5 = mul i32 %3, %4
  %6 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = add i32 %5, %6
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %8
  %10 = load double, ptr %9, align 8
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
  store double %82, ptr %83, align 8
  ret void
}

declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 9]}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!2 = !{i32 7, !"frame-pointer", i32 2}

; DEFINE: %{opt} = %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-enable-algebraic=false \
; DEFINE:   -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-precision-tier-step=100 \
; DEFINE:   -poseidon-precision-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-print -S
; DEFINE: %{mm} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-print \
; DEFINE:   -S -o /dev/null
; DEFINE: %{mmcm} = -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv
; RUN: rm -rf %t && mkdir -p %t/prof
; RUN: sed s/1.000000e-07/1.000000e-12/ %s > %t/mid.ll
; RUN: sed s/1.000000e-07/1.000000e-18/ %s > %t/tight.ll
; RUN: cp %S/../Inputs/expansion_two_components/fms_body.fpprofile %t/prof/fms_kernel.fpprofile

; A site's accuracy target (a per-operation relative tolerance) at three
; values against one cached DP table: the cheapest point that clears it, read
; back from the cache; a target below the FP64 body's own error (A0/S =
; 2.16e-17) is refused and the original body kept bit for bit.
; RUN: mkdir %t/ladder
; RUN: %opt %s %{opt} -poseidon-cache=%t/ladder/c -poseidon-explain=%t/ladder/loose 2>&1 | tee %t/ladder/loose.log | FileCheck --check-prefix=LADDER-LOOSE %s
; RUN: %opt %t/mid.ll %{opt} -poseidon-cache=%t/ladder/c 2>&1 | FileCheck --check-prefixes=LADDER-MID,EXP2 --implicit-check-not="Solving the DP" %s
; RUN: %opt %t/tight.ll %{opt} -poseidon-cache=%t/ladder/c -poseidon-explain=%t/ladder/tight 2>&1 | tee %t/ladder/tight.log | FileCheck --check-prefix=LADDER-TIGHT --implicit-check-not="Solving the DP" --implicit-check-not="!poseidon.pair.join" %s
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/ladder/loose.log %t/ladder/loose | FileCheck --check-prefix=LOOSE-JSON %s
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/ladder/tight.log %t/ladder/tight | FileCheck --check-prefix=TIGHT-JSON %s
; RUN: FileCheck --check-prefix=TIGHT-BEST %s < %t/ladder/tight/fms_kernel.explain.json

; LADDER-LOOSE: Per-operation relative error tolerance: 1.000000e-07
; LADDER-LOOSE-NEXT: Solving the DP for preprocess_fms_kernel_poseidon_body.
; LADDER-LOOSE: DP table for preprocess_fms_kernel_poseidon_body cached to
; LADDER-LOOSE: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-07 S=1.187943e+04 A0=2.571104e-13 selected cost=-9.223869e-01 accCost=1.697296e-04 rel=1.428768e-08
; LADDER-LOOSE: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; LADDER-LOOSE: [poseidon] pin: fms_kernel:P0=All FP64(0%) + FP32(100%)
; LADDER-LOOSE-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; LADDER-LOOSE: call float @llvm.fmuladd.f32(
; LADDER-LOOSE-NOT: !poseidon.pair.join

; LADDER-MID: Per-operation relative error tolerance: 1.000000e-12
; LADDER-MID-NEXT: Cache entry for preprocess_fms_kernel_poseidon_body found. Loaded DP tables from cache.
; LADDER-MID: [poseidon] preprocess_fms_kernel_poseidon_body: tau=1.000000e-12 S=1.187943e+04 A0=2.571104e-13 selected cost=-1.692668e-01 accCost=8.072609e-12 rel=7.011882e-16
; LADDER-MID: Applying solution for precision change All FP64(0%) + Expansion2(100%) (#2)
; LADDER-MID: [poseidon] pin: fms_kernel:P0=All FP64(0%) + Expansion2(100%)

; LOOSE-JSON: preprocess_fms_kernel_poseidon_body: rewritten{{$}}
; TIGHT-JSON: preprocess_fms_kernel_poseidon_body: no_solution; no frontier point meets the accuracy target 1e-18{{$}}
; TIGHT-BEST: "best_achievable_rel_error": 2.16433{{[0-9]+}}e-17,

; LADDER-TIGHT: Per-operation relative error tolerance: 1.000000e-18
; LADDER-TIGHT-NEXT: Cache entry for preprocess_fms_kernel_poseidon_body found. Loaded DP tables from cache.
; LADDER-TIGHT: [poseidon] No solution found that meets accuracy tolerance 1.000000e-18; the best relative accuracy in the DP table is 2.164332e-17{{$}}
; LADDER-TIGHT: [poseidon] pin: {{$}}
; LADDER-TIGHT: [poseidon] no rewrite applied for preprocess_fms_kernel_poseidon_body; calling original body fms_kernel_poseidon_body (bit-transparent no-op)
; LADDER-TIGHT-LABEL: define {{.*}} @fms_kernel(
; LADDER-TIGHT-SAME: ptr {{.*}}[[KA:%[0-9]+]], ptr {{.*}}[[KB:%[0-9]+]], ptr {{.*}}[[KC:%[0-9]+]])
; LADDER-TIGHT-NEXT: entry:
; LADDER-TIGHT-NEXT: call void @fms_kernel_poseidon_body(ptr [[KA]], ptr [[KB]], ptr [[KC]])
; LADDER-TIGHT-NEXT: ret void
; LADDER-TIGHT-LABEL: define internal void @fms_kernel_poseidon_body(
; LADDER-TIGHT-COUNT-5: call double @llvm.fmuladd.f64(
; LADDER-TIGHT-NOT: preprocess_
; LADDER-TIGHT-NOT: !poseidon.pair.join

; Expansion2 joins the result pair into the stored double.
; EXP2-LABEL: define {{.*}} @preprocess_fms_kernel_poseidon_body(
; EXP2: fadd double %{{.+}}, %{{.+}}, !poseidon.pair.join

; One matrix product under a ladder of targets crossing each candidate's
; modelled error: each target takes the cheapest candidate within it, else the
; loop as written. Of two within 0.6% in cost the one of smaller error is taken
; (tf32 0.4% above f16 is taken, 0.7% above is not).
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-2 2>&1 | FileCheck --check-prefix=F16 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-3 2>&1 | FileCheck --check-prefix=F16 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-4 2>&1 | FileCheck --check-prefix=NM8 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-8 2>&1 | FileCheck --check-prefix=NM8 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-9 2>&1 | FileCheck --check-prefix=NM9 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-10 2>&1 | FileCheck --check-prefix=NM10 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-11 2>&1 | FileCheck --check-prefix=NM11 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-12 2>&1 | FileCheck --check-prefix=NM12 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-13 2>&1 | FileCheck --check-prefix=NM13 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-14 2>&1 | FileCheck --check-prefix=NM14 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-15 2>&1 | FileCheck --check-prefix=NM14 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-16 2>&1 | FileCheck --check-prefix=NATIVE %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/c -poseidon-tau=1e-18 2>&1 | FileCheck --check-prefix=NATIVE %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/k -poseidon-matmul-library-calls=false -poseidon-tau=1e-3 2>&1 | FileCheck --check-prefix=WMMA %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/k -poseidon-matmul-library-calls=false -poseidon-tau=1e-6 2>&1 | FileCheck --check-prefix=TCEC2 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/k -poseidon-matmul-library-calls=false -poseidon-tau=1e-7 2>&1 | FileCheck --check-prefix=TCEC3 %s
; RUN: %{mm} %{mmcm} -poseidon-cache=%t/k -poseidon-matmul-library-calls=false -poseidon-tau=1e-8 2>&1 | FileCheck --check-prefix=WRITTEN %s
; RUN: sed 's/^library_direct_rel,tf32_f32,.*/library_direct_rel,tf32_f32,0.00784375/' %cost_models/cm_sm_120_synthetic_library.csv > %t/tie.csv
; RUN: sed 's/^library_direct_rel,tf32_f32,.*/library_direct_rel,tf32_f32,0.0078671875/' %cost_models/cm_sm_120_synthetic_library.csv > %t/apart.csv
; RUN: %{mm} -poseidon-cost-model=%t/tie.csv -poseidon-cache=%t/tie -poseidon-tau=1e-3 2>&1 | FileCheck --check-prefix=TF32 %s
; RUN: %{mm} -poseidon-cost-model=%t/apart.csv -poseidon-cache=%t/apart -poseidon-tau=1e-3 2>&1 | FileCheck --check-prefix=F16 %s

; F16: [poseidon] pin: matmul_body:M0=direct-dispatch cublas f16/f32{{$}}
; TF32: [poseidon] pin: matmul_body:M0=direct-dispatch cublas tf32/f32{{$}}
; NM8: [poseidon] pin: matmul_body:M0=ozaki-ii nm=8 wmma m16n16k16 s8/s32{{$}}
; NM9: [poseidon] pin: matmul_body:M0=ozaki-ii nm=9 wmma m16n16k16 s8/s32{{$}}
; NM10: [poseidon] pin: matmul_body:M0=ozaki-ii nm=10 wmma m16n16k16 s8/s32{{$}}
; NM11: [poseidon] pin: matmul_body:M0=ozaki-ii nm=11 wmma m16n16k16 s8/s32{{$}}
; NM12: [poseidon] pin: matmul_body:M0=ozaki-ii nm=12 wmma m16n16k16 s8/s32{{$}}
; NM13: [poseidon] pin: matmul_body:M0=ozaki-ii nm=13 wmma m16n16k16 s8/s32{{$}}
; NM14: [poseidon] pin: matmul_body:M0=ozaki-ii nm=14 wmma m16n16k16 s8/s32{{$}}
; NATIVE: [poseidon] pin: matmul_body:M0=native-dgemm cublas f64/f64{{$}}
; WMMA: [poseidon] pin: matmul_body:M0=wmma m16n16k16 f16/f32{{$}}
; TCEC2: [poseidon] pin: matmul_body:M0=tcec n=2 wmma m16n16k16 f16/f32{{$}}
; TCEC3: [poseidon] pin: matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32{{$}}
; WRITTEN-NOT: Applying solution for
; WRITTEN: [poseidon] no rewrite applied for preprocess_matmul_body; calling original body matmul_body (bit-transparent no-op){{$}}

source_filename = "annotated_kernel.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [14 x i8] c"tau_ladder.cu\00", section "llvm.metadata"
@.args = private unnamed_addr constant { double } { double 1.000000e-07 }, section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @fms_kernel, ptr @.str, ptr @.str.1, i32 2, ptr @.args }], section "llvm.metadata"

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

declare double @llvm.fmuladd.f64(double, double, double) #1
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4
attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

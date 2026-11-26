; DEFINE: %{ds} = %S/../../Inputs/expansion_two_components/ds_fms.ll
; DEFINE: %{opt} = %opt %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/expansion_two_components -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-precision-tier-step=100 -poseidon-precision-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-1e-9 -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} %s -poseidon-early-prune -poseidon-enable-precision=true -poseidon-precision-expansions=true -poseidon-cache=%t/c -poseidon-print > %t/out 2>&1
; RUN: FileCheck %s < %t/out
; RUN: FileCheck --check-prefix=CLEAN %s < %t/out
; RUN: mkdir -p %t/noscalar && grep -v '^# scalar_types=' %cost_models/cm_sm_120_synthetic.csv > %t/noscalar/cm.csv
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../../Inputs/expansion_two_components -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=true -poseidon-cost-model=%t/noscalar/cm.csv -poseidon-cache=%t/noscalar -S 2>&1 | FileCheck --check-prefix=NOSCALAR %s
; RUN: sed 's/tail call double @llvm.fmuladd/tail call contract double @llvm.fmuladd/' %{ds} > %t/contract.ll
; RUN: sed 's/tail call double @llvm.fmuladd/tail call fast double @llvm.fmuladd/; s/= fneg double/= fneg fast double/' %{ds} > %t/fast.ll
; RUN: sed 's/^attributes #0 = { /attributes #0 = { "unsafe-fp-math"="true" "no-nans-fp-math"="true" "no-infs-fp-math"="true" "approx-func-fp-math"="true" "no-signed-zeros-fp-math"="true" /' %{ds} > %t/attr.ll
; RUN: %if !global-fp-contract %{ %{opt} %{ds} -poseidon-early-prune -poseidon-cache=%t/plain -poseidon-print -o %t/plain.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s %}
; RUN: %if !global-fp-contract %{ %{opt} %t/contract.ll -poseidon-early-prune -poseidon-cache=%t/contract -poseidon-print -o %t/contract.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s %}
; RUN: %if !global-fp-contract %{ %{opt} %t/fast.ll -poseidon-early-prune -poseidon-cache=%t/fast -poseidon-print -o %t/fast.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s %}
; RUN: %if !global-fp-contract %{ %{opt} %t/attr.ll -poseidon-early-prune -poseidon-cache=%t/attr -poseidon-print -o %t/attr.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s %}
; RUN: %if !global-fp-contract %{ %{opt} %t/contract.ll -poseidon-early-prune -poseidon-cache=%t/strict -poseidon-strict-mode -poseidon-print -o %t/strict.out.ll 2>&1 | FileCheck --check-prefix=APPLIED %s %}
; RUN: %if !global-fp-contract %{ FileCheck --check-prefix=BODY %s < %t/fast.out.ll %}
; RUN: %if !global-fp-contract %{ sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/plain.out.ll > %t/plain.body.ll %}
; RUN: %if !global-fp-contract %{ sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/contract.out.ll | diff %t/plain.body.ll - %}
; RUN: %if !global-fp-contract %{ sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/fast.out.ll | diff %t/plain.body.ll - %}
; RUN: %if !global-fp-contract %{ sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/attr.out.ll | diff %t/plain.body.ll - %}
; RUN: %if !global-fp-contract %{ sed -n '/^define .*@preprocess_fms_body/,/^}/p' %t/strict.out.ll | diff %t/plain.body.ll - %}
; RUN: %if global-fp-contract %{ %{opt} %t/contract.ll -poseidon-cache=%t/rcontract -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=REFUSED %s %}
; RUN: %if global-fp-contract %{ %{opt} %t/fast.ll -poseidon-cache=%t/rfast -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=REFUSED %s %}
; RUN: %if global-fp-contract %{ not %{opt} %t/contract.ll -poseidon-cache=%t/rstrict -poseidon-strict-mode -o /dev/null 2>&1 | FileCheck --check-prefix=STRICT %s %}

; Double-single materialization of the fmuladd chain of ds_fms.cu (fms_body:
; t = a*b+a, u = t*t-b, v = u*b+t, w = v*v-a, out = w*u+v; profile in
; Inputs/expansion_two_components) via clang -x cuda --cuda-device-only
; --cuda-gpu-arch=sm_120 -O2 -ffp-contract=on (profile from three launches
; <<<4, 64>>> on an sm_120 GPU): inputs split into float pairs, the chain runs
; on pairs only, the last pair is joined into the stored double; a cost model
; without '# scalar_types=' stops the compile.
; Fast-math flags or attributes on the chain: from LLVM 24 (APPLIED, BODY) the
; site still gets its expansion, emitted without flags and identical to the
; plain one; up to LLVM 23, whose code generator also fuses flagless pairs in a
; -ffp-contract=fast unit, a 'contract' site gets no expansion candidate, with
; one warning (REFUSED), fatal under strict mode (STRICT).

; CHECK: Candidates:
; CHECK: All FP64(0%) + FP32(100%)
; CHECK: All FP64(0%) + Expansion2(100%)
; CHECK: Applying solution for precision change All FP64(0%) + Expansion2(100%)
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

; NOSCALAR: error: <unknown>:0:0: in function preprocess_fms_body void (ptr, ptr, ptr): Poseidon: cost model {{.*}}noscalar/cm.csv carries no '# scalar_types=' header, so the formats the device has hardware for are unknown

; APPLIED-NOT: are not proposed
; APPLIED: Applying solution for precision change All FP64(0%) + Expansion2(100%)

; BODY-LABEL: define {{.*}} @preprocess_fms_body(
; BODY-NOT: {{ (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) }}
; BODY: call float @llvm.fma.f32(
; BODY-NOT: {{ (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) }}
; BODY: fadd double %{{.+}}, %{{.+}}, !poseidon.pair.join
; BODY-NOT: {{ (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) }}
; BODY: ret void

; REFUSED: warning: {{.*}}preprocess_fms_body{{.*}}: Poseidon: floating-point expansions are not proposed: its floating-point operations carry 'contract' (the first llvm.fmuladd.f64), as under -ffp-contract=fast or -ffast-math, with which this LLVM's code generator also fuses multiply-add pairs that carry no flag; compile with -ffp-contract=on or -ffp-contract=off, without -ffast-math{{$}}
; REFUSED-NOT: are not proposed
; REFUSED-NOT: Expansion{{[0-9]}}
; REFUSED: Applying solution for precision change All FP64(0%) + FP32(100%)

; STRICT: error: {{.*}}Poseidon: strict mode: site 'preprocess_fms_body': its floating-point operations carry 'contract' (the first llvm.fmuladd.f64), as under -ffp-contract=fast or -ffast-math, with which this LLVM's code generator also fuses multiply-add pairs that carry no flag; floating-point expansions need -ffp-contract=on or -ffp-contract=off

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

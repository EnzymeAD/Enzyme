; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/rewrite_after_upstream_precision %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../../Inputs/subgraph_split -poseidon-early-prune -poseidon-subgraph-split-uses=3 -poseidon-subgraph-split-ops=4 -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-enable-precision=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv "-poseidon-pin=P0=All FP64(0%%) + FP32(100%%),R4=(fma.f64 v2 v1 (+.f64 v2 v0))" -poseidon-cache=%t/pt-first -S -poseidon-print -o %t/pt-first.ll 2>&1 | FileCheck --check-prefix=APPLY %s
; RUN: FileCheck --check-prefix=IR %s < %t/pt-first.ll
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/rewrite_after_upstream_precision %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../../Inputs/subgraph_split -poseidon-early-prune -poseidon-subgraph-split-uses=3 -poseidon-subgraph-split-ops=4 -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-enable-precision=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv "-poseidon-pin=R4=(fma.f64 v2 v1 (+.f64 v2 v0)),P0=All FP64(0%%) + FP32(100%%)" -poseidon-cache=%t/rewrite-first -S -poseidon-print -o %t/rewrite-first.ll 2>&1 | FileCheck --check-prefix=APPLY %s
; RUN: diff %t/pt-first.ll %t/rewrite-first.ll

; A precision change of one FP subgraph and a rewrite of the subgraph it feeds.
; The module and profile are subgraph_split.ll's: the split at %t makes the
; upstream subgraph {%a %b %c %t} (P0) and the downstream one (R4 rewrites its
; output %r1 = %t * %x + (%t + %y), Herbie's answer canned in
; Techniques/Precision/Inputs/rewrite_after_upstream_precision). P0 moves %t to binary32, its users
; reading %t extended back to binary64; the rewrite reads that same value as v2.
; The steps apply in the solve's order, the rewrite first, whatever the order
; of the items.

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

; APPLY: Applying solution for (+ (* v2 v1) (+ v2 v0)) --(0)-> (fma.f64 v2 v1 (+.f64 v2 v0))
; APPLY-NEXT: Applying solution for precision change All FP64(0%) + FP32(100%)

; IR-LABEL: define void @preprocess_tester(
; IR: %[[T:[0-9]+]] = fsub float
; IR-NEXT: %[[TD:[0-9]+]] = fpext float %[[T]] to double
; IR-NEXT: [[U2:%[^ ,()]+]] = fadd double %y, %[[TD]]
; IR-NEXT: [[U3:%[^ ,()]+]] = fdiv double %[[TD]], %x
; IR-NEXT: [[HERBIE_FMA:%[^ ,()]+]] = tail call double @llvm.fma.f64(double %[[TD]], double %x, double [[U2]])
; IR-NEXT: [[R2:%[^ ,()]+]] = fmul double [[U3]], [[U2]]
; IR-NEXT: store double [[HERBIE_FMA]], ptr %o1

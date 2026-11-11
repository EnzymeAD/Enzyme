; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/rewrite_after_upstream_precision %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/chain_split -poseidon-min-uses-split=3 -poseidon-min-ops-split=4 -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/rewrite_after_upstream_precision/fake-herbie.sh -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv "-poseidon-apply-rewrites=PT0=All FP64(0%%) + FP32(100%%),R4=(fma.f64 v2 v1 (+.f64 v2 v0))" -poseidon-cache=%t/pt-first -S -o %t/pt-first.ll 2> %t/pt-first.err
; RUN: FileCheck --check-prefix=APPLY %s < %t/pt-first.err
; RUN: FileCheck --check-prefix=IR %s < %t/pt-first.ll
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/rewrite_after_upstream_precision %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/chain_split -poseidon-min-uses-split=3 -poseidon-min-ops-split=4 -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/rewrite_after_upstream_precision/fake-herbie.sh -poseidon-enable-pt=true -poseidon-cost-model=%cost_models/cm_cpu_native.csv "-poseidon-apply-rewrites=R4=(fma.f64 v2 v1 (+.f64 v2 v0)),PT0=All FP64(0%%) + FP32(100%%)" -poseidon-cache=%t/rewrite-first -S -o %t/rewrite-first.ll
; RUN: diff %t/pt-first.ll %t/rewrite-first.ll

; A precision change of one FP subgraph, then a rewrite of the subgraph it
; feeds. The module and profile are chain_split.ll's: the split at %t makes the
; upstream subgraph {%a %b %c %t} (PT0) and the downstream one (R4 rewrites its
; output %r1 = %t * %x + (%t + %y), with Herbie's answer canned in
; Inputs/rewrite_after_upstream_precision). PT0 moves %t to binary32 and erases
; it, its users reading %t extended back to binary64; the rewrite, which names
; %t as v2, reads that same value. Applied in either order, the two steps emit
; one program.

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

; APPLY: Applying solution for CS: All FP64(0%) + FP32(100%)
; APPLY-NEXT: Applying solution for (+ (* v2 v1) (+ v2 v0)) --(0)-> (fma.f64 v2 v1 (+.f64 v2 v0))

; IR-LABEL: define void @preprocess_tester(
; IR: %[[T:[0-9]+]] = fsub float
; IR-NEXT: %[[TD:[0-9]+]] = fpext float %[[T]] to double
; IR-NEXT: %u2 = fadd double %y, %[[TD]]
; IR-NEXT: %u3 = fdiv double %[[TD]], %x
; IR-NEXT: %herbie.fma = tail call double @llvm.fma.f64(double %[[TD]], double %x, double %u2)
; IR-NEXT: %r2 = fmul double %u3, %u2
; IR-NEXT: store double %herbie.fma, ptr %o1

; RUN: rm -rf %t
; RUN: %opt < %s %loadPoseidon -passes="assign-guid,poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/fma_site_profile -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=true -poseidon-precision-expansions=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t -S | FileCheck %s
; REQUIRES: guid-metadata

; A function's GUID is kept as !guid metadata, which a clone copies. Where the
; GUIDs were assigned before the pass runs, as in a full-LTO link of pre-link
; bitcode, the canonical clone of a site that ships gets the GUID of its own
; name, never the original's: LTO keys per-function state by GUID.

define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @site(double %x, double %y, double %z, double %xs, double %ys) {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, metadata !"enzyme_dup", double %x, double %xs, metadata !"enzyme_dup", double %y, double %ys, double %z, metadata !"poseidon_tau", double 1.0e-2)
  ret double %0
}

declare double @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; CHECK: define double @tester({{.*}}!guid ![[ORIGINAL:[0-9]+]]
; CHECK: define {{.*}}@preprocess_tester({{.*}}!guid ![[CLONE:[0-9]+]]
; CHECK-DAG: ![[ORIGINAL]] = !{i64 -2743526145958358539}
; CHECK-DAG: ![[CLONE]] = !{i64 3151465777686795207}

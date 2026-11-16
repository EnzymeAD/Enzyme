; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/unbounded_input \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -pass-remarks-missed=poseidon -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cost-budget=-0.3 -poseidon-cache=%t/budget -o %t/budget.ll 2>&1 | FileCheck --check-prefix=BUDGET %s
; RUN: FileCheck --check-prefix=IR %s < %t/budget.ll
; RUN: not %{opt} -poseidon-tau=1e-3 -poseidon-cache=%t/tau -o /dev/null 2>&1 | FileCheck --check-prefix=TAU %s

; A subgraph that cannot be sampled has no priced precision candidate: it is
; left as written and the compile goes on. The profiled run saw @tester's x at
; infinity, and @tester_nan only takes the square root of negative numbers.
; Under a tolerance the profile of @tester_nan, which carries no sensitivity,
; is an error of its own.
;
; Profiles: this module through the host profiler with
; Inputs/unbounded_input/driver.c.

define void @tester(double %x, double %y, ptr %o) #0 {
entry:
  %a = fadd double %x, %y
  %b = fmul double %a, %y
  store double %b, ptr %o, align 8
  ret void
}

define void @tester_nan(double %x, double %y, ptr %o) #0 {
entry:
  %a = fmul double %x, %y
  %s = call double @llvm.sqrt.f64(double %a)
  %b = fadd double %s, %y
  store double %b, ptr %o, align 8
  ret void
}

; IR-LABEL: define void @site(
; IR: call void @tester(
define void @site(double %x, double %y, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

; IR-LABEL: define void @site_nan(
; IR: call void @tester_nan(
define void @site_nan(double %x, double %y, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_nan, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

declare double @llvm.sqrt.f64(double)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

; BUDGET: input to sampled subgraph has unprofiled bounds [2.000000e+00, INF]
; BUDGET: [poseidon] subgraph cannot be priced (the profiled input box is empty or unbounded); discarding its 2 precision candidate(s) and keeping the original body.
; BUDGET: remark: <unknown>:0:0: Poseidon left (? + ?) * ? as written: its candidates could not be priced: the profiled input box is empty or unbounded
; BUDGET: [poseidon] subgraph cannot be priced (every sample of the original expression scored NaN); discarding its 4 precision candidate(s) and keeping the original body.
; BUDGET: remark: <unknown>:0:0: Poseidon left sqrt(? * ?) + ? as written: its candidates could not be priced: every sample of the original expression scored NaN

; TAU: [poseidon] preprocess_tester: 1 subgraph output(s) carry no priced candidate, so the baseline below covers only the priced ones and the reported relative error is a lower bound.
; TAU: error: {{.*}} Poseidon: site preprocess_tester_nan was given an accuracy tolerance but its profile carries a total sensitivity of 0.000000e+00

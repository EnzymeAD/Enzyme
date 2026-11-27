; DEFINE: %{opt} = %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/Inputs/unbounded_input -poseidon-early-prune \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=true \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -pass-remarks-missed=poseidon -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-cost-budget=-0.3 -poseidon-cache=%t/budget -poseidon-print -o %t/budget.ll 2>&1 | FileCheck --check-prefix=BUDGET %s
; RUN: FileCheck --check-prefix=IR --implicit-check-not=preprocess %s < %t/budget.ll
; RUN: not %{opt} -poseidon-tau=1e-3 -poseidon-cache=%t/tau -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=TAU %s
; RUN: mkdir %t/foreign && cp %S/Inputs/unbounded_input/*.fpprofile %t/foreign/
; RUN: sed -i -e '10s/NumOperands = 2/NumOperands = 0/' -e '11,12d' %t/foreign/tester.fpprofile
; RUN: not %{opt} -poseidon-profile-use=%t/foreign -poseidon-early-prune -poseidon-cost-budget=-0.3 -poseidon-cache=%t/foreign-cache -o /dev/null 2>&1 | FileCheck --check-prefix=FOREIGN %s

; A subgraph whose profiled input range holds no finite interval cannot be
; sampled: no priced precision candidate, left as written (BUDGET, IR); under a
; tolerance the profile of @tester_nan, with no sensitivity, is an error (TAU).
; A profile giving an operation no operand range was not generated from this
; program: the compile stops naming the operand (FOREIGN). Profiles from the
; host profiler: @tester saw x = infinity, @tester_nan only sqrt of negatives.

; IR-LABEL: define void @tester(double %x, double %y, ptr %o)
; IR-NEXT: entry:
; IR-NEXT: %a = fadd double %x, %y
; IR-NEXT: %b = fmul double %a, %y
; IR-NEXT: store double %b, ptr %o, align 8
; IR-NEXT: ret void
define void @tester(double %x, double %y, ptr %o) #0 {
entry:
  %a = fadd double %x, %y
  %b = fmul double %a, %y
  store double %b, ptr %o, align 8
  ret void
}

; IR-LABEL: define void @tester_nan(double %x, double %y, ptr %o)
; IR-NEXT: entry:
; IR-NEXT: %a = fmul double %x, %y
; IR-NEXT: %s = call double @llvm.sqrt.f64(double %a)
; IR-NEXT: %b = fadd double %s, %y
; IR-NEXT: store double %b, ptr %o, align 8
; IR-NEXT: ret void
define void @tester_nan(double %x, double %y, ptr %o) #0 {
entry:
  %a = fmul double %x, %y
  %s = call double @llvm.sqrt.f64(double %a)
  %b = fadd double %s, %y
  store double %b, ptr %o, align 8
  ret void
}

; IR-LABEL: define void @site(
; IR-NEXT: entry:
; IR-NEXT: call void @tester(double %x, double %y, ptr %o)
; IR-NEXT: ret void
define void @site(double %x, double %y, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

; IR-LABEL: define void @site_nan(
; IR-NEXT: entry:
; IR-NEXT: call void @tester_nan(double %x, double %y, ptr %o)
; IR-NEXT: ret void
define void @site_nan(double %x, double %y, ptr %o, ptr %do) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_nan, double %x, double %y, metadata !"enzyme_dup", ptr %o, ptr %do)
  ret void
}

declare double @llvm.sqrt.f64(double)
declare void @__poseidon_fp_optimize(ptr, ...)
attributes #0 = { "target-cpu"="x86-64" }

; BUDGET: [poseidon] subgraph cannot be priced (the profiled range [2.000000e+00, INF] of the subgraph input double %x is not an interval of finite values); discarding its 2 precision change candidate(s) and keeping the original body{{$}}
; BUDGET: [poseidon] No solution found within the computation cost budget
; BUDGET-NEXT: remark: <unknown>:0:0: Poseidon left site tester as written: no frontier point is within the cost budget -0.3{{$}}
; BUDGET-NEXT: remark: <unknown>:0:0: Poseidon left (? + ?) * ? as written: its candidates could not be priced: the profiled range [2.000000e+00, INF] of the subgraph input double %x is not an interval of finite values and the algebraic search is disabled
; BUDGET: [poseidon] no rewrite applied for preprocess_tester; calling original body {{[^ ]+}} (bit-transparent no-op){{$}}
; BUDGET: [poseidon] subgraph cannot be priced (every sample of the original expression scored NaN); discarding its 4 precision change candidate(s) and keeping the original body{{$}}
; BUDGET: [poseidon] No solution found within the computation cost budget
; BUDGET-NEXT: remark: <unknown>:0:0: Poseidon left site tester_nan as written: no frontier point is within the cost budget -0.3{{$}}
; BUDGET-NEXT: remark: <unknown>:0:0: Poseidon left sqrt(? * ?) + ? as written: its candidates could not be priced: every sample of the original expression scored NaN and the algebraic search is disabled
; BUDGET: [poseidon] no rewrite applied for preprocess_tester_nan; calling original body {{[^ ]+}} (bit-transparent no-op){{$}}

; TAU: [poseidon] preprocess_tester: 1 subgraph output(s) carry no priced candidate, so the baseline below covers only the priced ones and the reported relative error is a lower bound.
; TAU: [poseidon] no rewrite applied for preprocess_tester; calling original body {{[^ ]+}} (bit-transparent no-op){{$}}
; TAU: [poseidon] preprocess_tester_nan: 1 subgraph output(s) carry no priced candidate, so the baseline below covers only the priced ones and the reported relative error is a lower bound.
; TAU-NEXT: error: <unknown>:0:0: in function preprocess_tester_nan void (double, double, ptr): Poseidon: site preprocess_tester_nan was given an accuracy tolerance but its profile carries a total sensitivity of 0.000000e+00, so no rewrite's modelled error can be expressed as a per-operation relative level; re-run profile generation for this site

; FOREIGN: error: <unknown>:0:0: in function preprocess_tester void (double, double, ptr): Poseidon: the profile gives this operation 0 operand range(s), and none for its operand 0 (double %x); re-run profile generation for this program

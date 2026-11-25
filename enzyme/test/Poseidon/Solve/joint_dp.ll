; DEFINE: %{pass} = %loadPoseidon -passes="poseidon,poseidon-finalize"
; DEFINE: %{passfn} = %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)"
; DEFINE: %{prof} = -poseidon-profile-use=%S/../Inputs/joint_profiles
; DEFINE: %{cm} = -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; RUN: rm -rf %t && mkdir %t

; Two annotated sites solved under one shared budget (-poseidon-joint-dp).
; Their profiles come from running this module through the host profiler,
; driven by
;   for (int i = 0; i < 64; ++i) {
;     double x = 1.0 + i * 0.125, y = 0.5 + (i % 7) * 0.25;
;     s += site_a(x, y) + site_b(x, y);
;   }

; The explanation of each site lists the joint table's breakpoints, a budget
; of -0.4 rewrites both sites to FP32, a budget of -0.2 is spent on tester_b
; alone and site_a is restored to its original body.
; RUN: mkdir %t/sites
; RUN: %opt %s %{passfn} %{prof} -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp %{cm} -S -poseidon-cache=%t/sites/both -poseidon-explain=%t/sites/both/ex -poseidon-cost-budget=-0.4 -poseidon-print 2>&1 | FileCheck --check-prefix=SITES-BOTH %s
; RUN: %poseidon_explain %t/sites/both/ex --format budgets | FileCheck --check-prefix=SITES-BUDGETS %s
; RUN: %opt %s %{passfn} %{prof} -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp %{cm} -S -poseidon-cache=%t/sites/one -poseidon-cost-budget=-0.2 -poseidon-print 2>&1 | FileCheck --check-prefix=SITES-ONE %s

; SITES-BOTH: Poseidon JOINT: collecting candidates from 2 marked function(s)
; SITES-BOTH: Critical computation cost range: [-4.500000e-01, 0.000000e+00]
; SITES-BOTH-NEXT: DP table contains 4 entries.
; SITES-BOTH: Minimum accuracy cost within budget: 1.154131e-04
; SITES-BOTH-NEXT: Computation cost budget used: -4.500000e-01
; SITES-BOTH: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; SITES-BOTH: [poseidon] pin: tester_a:PT0=All FP64(0%) + FP32(100%)
; SITES-BOTH: [poseidon] Finished optimizing preprocess_tester_a
; SITES-BOTH: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; SITES-BOTH: [poseidon] pin: tester_b:PT0=All FP64(0%) + FP32(100%)
; SITES-BOTH: [poseidon] Finished optimizing preprocess_tester_b
; SITES-BOTH-LABEL: define double @site_a(
; SITES-BOTH-NEXT: entry:
; SITES-BOTH-NEXT: call double @preprocess_tester_a(double %x, double %y)
; SITES-BOTH-LABEL: define double @site_b(
; SITES-BOTH-NEXT: entry:
; SITES-BOTH-NEXT: call double @preprocess_tester_b(double %x, double %y)
; SITES-BOTH-LABEL: define double @preprocess_tester_a(
; SITES-BOTH-NEXT: entry:
; SITES-BOTH-NEXT: fptrunc fast double %x to float
; SITES-BOTH-LABEL: define double @preprocess_tester_b(
; SITES-BOTH-NEXT: entry:
; SITES-BOTH-NEXT: fptrunc fast double %y to float

; SITES-BUDGETS: -0.45000000000000001,-0.29999999999999999,-0.050000000000000003,0

; SITES-ONE: Critical computation cost range: [-4.500000e-01, 0.000000e+00]
; SITES-ONE-NEXT: DP table contains 4 entries.
; SITES-ONE: Minimum accuracy cost within budget: 2.184370e-06
; SITES-ONE-NEXT: Computation cost budget used: -3.000000e-01
; SITES-ONE-NEXT: [poseidon] pin: {{$}}
; SITES-ONE: [poseidon] Finished optimizing preprocess_tester_a
; SITES-ONE: [poseidon] no rewrite applied for preprocess_tester_a; wrapper call restored to original body tester_a (bit-transparent no-op)
; SITES-ONE: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; SITES-ONE: [poseidon] pin: tester_b:PT0=All FP64(0%) + FP32(100%)
; SITES-ONE: [poseidon] Finished optimizing preprocess_tester_b
; SITES-ONE-LABEL: define double @site_a(
; SITES-ONE-NEXT: entry:
; SITES-ONE-NEXT: call double @tester_a(double %x, double %y)
; SITES-ONE-LABEL: define double @site_b(
; SITES-ONE-NEXT: entry:
; SITES-ONE-NEXT: call double @preprocess_tester_b(double %x, double %y)
; SITES-ONE-NOT: define double @preprocess_tester_a(

; A cost given where the budget takes a share (-400) asks to save more than the
; whole profiled cost, which no solution can: a solve that selects by the budget
; stops with an error at the site rather than selecting nothing, as it does
; for a share that is not finite. A compile that solves nothing, or solves
; under an accuracy target, does not read the budget.
; A number flag given no number, an empty value or nan, is an error at the
; first site whose solve reads it, not 0 (what LLVM's parser of a double makes
; of an empty value): the budget is read by a solve that selects by it, the
; other number flags by every solve.
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-cost-budget=-400 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-STALE %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-joint-dp -poseidon-cost-budget=inf -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONFINITE %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-cost-budget=1e300 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-HUGE %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-cost-budget= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-cost-budget %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-joint-dp -poseidon-cost-budget=nan -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-cost-budget %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-tau= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-tau %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-min-cost-share= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-min-cost-share %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-max-broken-share= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-max-broken-share %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false %{cm} -poseidon-cache= -poseidon-exponent-penalty= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-exponent-penalty %s
; RUN: %opt %s %loadPoseidon -passes=poseidon -poseidon-tau= -poseidon-cost-budget= -poseidon-exponent-penalty= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-UNUSED --allow-empty %s
; RUN: %opt %s %loadPoseidon -passes=poseidon -poseidon-cost-budget=-400 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-UNUSED --allow-empty %s
; RUN: %opt %s %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-herbie=false -poseidon-enable-pt=false %{cm} -poseidon-cache= -poseidon-tau=0.5 -poseidon-cost-budget=nan -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-UNUSED --allow-empty %s

; SITES-STALE-NOT: Starting accuracy DP solver
; SITES-STALE: error: {{.*}}Poseidon: -poseidon-cost-budget=-400 asks a solution to save more than the whole profiled cost. It takes the share of a site's profiled cost a solution may add, negative for one it has to save (-0.25 requires saving a quarter), not a cost. The frontier of a site's solve, with the budget that selects each point, is in the <site>.explain.json -poseidon-explain=<dir> writes.
; SITES-HUGE: error: {{.*}}Poseidon: -poseidon-cost-budget=1.0000000000000001e+300 lets a solution add more than 2^30 times the profiled cost, which the DP's cost axis does not hold; give a share of at most 1073741824.
; SITES-NONFINITE-NOT: Starting accuracy DP solver
; SITES-NONFINITE: error: {{.*}}Poseidon: -poseidon-cost-budget=inf is not a finite number. It takes the share of a site's profiled cost a solution may add, negative for one it has to save (-0.25 requires saving a quarter), not a cost. The frontier of a site's solve, with the budget that selects each point, is in the <site>.explain.json -poseidon-explain=<dir> writes.
; SITES-UNUSED-NOT: error:
; SITES-UNUSED-NOT: -poseidon-cost-budget

; SITES-NONUMBER-NOT: Starting accuracy DP solver
; SITES-NONUMBER: error: {{.*}}Poseidon: -[[FLAG]] was given no number: its value is empty or nan. Write a number after '=', or leave the flag out for its default.
; SITES-NONUMBER-NOT: Starting accuracy DP solver

; One shared budget is spent against one accuracy target, so the sites of a
; -poseidon-joint-dp solve have to mean the same one: a site's own target, else
; -poseidon-tau, else none. Different targets are a compile error naming each
; site; equal ones solve as one site would, wherever they are written.
; RUN: mkdir %t/targets
; RUN: sed 's/@tester_a, double %x, double %y)/@tester_a, double %x, double %y, metadata !"poseidon_tau", double 1.0e-02)/' %s > %t/targets/a.ll
; RUN: sed 's/@tester_b, double %x, double %y)/@tester_b, double %x, double %y, metadata !"poseidon_tau", double 1.0e-03)/' %t/targets/a.ll > %t/targets/ab_diff.ll
; RUN: sed 's/@tester_b, double %x, double %y)/@tester_b, double %x, double %y, metadata !"poseidon_tau", double 1.0e-02)/' %t/targets/a.ll > %t/targets/ab_same.ll
; RUN: not %opt %t/targets/a.ll %{pass} %{prof} -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp %{cm} -poseidon-cache=%t/targets/c1 -S -o /dev/null 2>&1 | FileCheck --check-prefix=TARGETS-NONE %s
; RUN: not %opt %t/targets/ab_diff.ll %{pass} %{prof} -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp %{cm} -poseidon-cache=%t/targets/c2 -S -o /dev/null 2>&1 | FileCheck --check-prefix=TARGETS-DIFF %s
; RUN: %opt %t/targets/a.ll %{passfn} %{prof} -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp %{cm} -poseidon-cache=%t/targets/c3 -poseidon-tau=1e-2 -poseidon-explain=%t/targets/ex -S -o %t/targets/flag.out.ll 2> %t/targets/flag.err
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/targets/flag.err %t/targets/ex | FileCheck --check-prefix=TARGETS-JSON %s
; RUN: FileCheck --check-prefix=TARGETS-SITES %s < %t/targets/ex/preprocess_tester_a.explain.json
; RUN: %opt %t/targets/ab_same.ll %{passfn} %{prof} -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-joint-dp %{cm} -poseidon-cache=%t/targets/c4 -S -o %t/targets/site.out.ll 2> %t/targets/site.err
; RUN: FileCheck --check-prefix=TARGETS-SOLVE %s < %t/targets/flag.err
; RUN: FileCheck --check-prefix=TARGETS-SOLVE %s < %t/targets/site.err
; RUN: sed -e 's/, metadata !"poseidon_tau", double 1.000000e-02//' -e '/^; ModuleID/d' -e '/^source_filename/d' %t/targets/flag.out.ll > %t/targets/flag.cmp.ll
; RUN: sed -e 's/, metadata !"poseidon_tau", double 1.000000e-02//g' -e '/^; ModuleID/d' -e '/^source_filename/d' %t/targets/site.out.ll > %t/targets/site.cmp.ll
; RUN: diff %t/targets/flag.cmp.ll %t/targets/site.cmp.ll

; TARGETS-NONE: error: {{.*}}Poseidon: the sites of one -poseidon-joint-dp solve have different accuracy targets: tester_a (1.000000e-02) tester_b (none); give them one target
; TARGETS-DIFF: error: {{.*}}Poseidon: the sites of one -poseidon-joint-dp solve have different accuracy targets: tester_a (1.000000e-02) tester_b (1.000000e-03); give them one target
; TARGETS-JSON: preprocess_tester_a: rewritten
; TARGETS-JSON-NEXT: preprocess_tester_b: rewritten
; TARGETS-SITES: "selector": "joint",
; TARGETS-SITES-NEXT: "sites": [
; TARGETS-SITES-NEXT: "preprocess_tester_a",
; TARGETS-SITES-NEXT: "preprocess_tester_b"
; TARGETS-SOLVE: [poseidon] preprocess_tester_a,preprocess_tester_b: tau=1.000000e-02 S=4.732407e+03 A0=1.003014e-13 selected cost=-4.500000e-01 accCost=1.154131e-04 rel=2.438782e-08

define double @tester_a(double %x, double %y) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %x
  ret double %mul
}

define double @tester_b(double %x, double %y) #0 {
entry:
  %mul = fmul fast double %x, %y
  %sub = fsub fast double %mul, %y
  %div = fdiv fast double %sub, %x
  ret double %div
}

define double @site_a(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_a, double %x, double %y)
  ret double %0
}

define double @site_b(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_b, double %x, double %y)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

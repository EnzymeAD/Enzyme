; DEFINE: %{pass} = %loadPoseidon -passes="poseidon,poseidon-finalize"
; DEFINE: %{passfn} = %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)"
; DEFINE: %{prof} = -poseidon-profile-use=%S/../Inputs/joint_profiles -poseidon-early-prune
; DEFINE: %{cm} = -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; DEFINE: %{mm} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/fp16_range/matmul_tiny -poseidon-early-prune \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-print -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/@tester_a, double %x, double %y)/@tester_a, double %x, double %y, metadata !"poseidon_tau", double 1.0e-02)/' %S/Inputs/two_sites.ll > %t/a.ll

; Two annotated sites solved under one shared budget (-poseidon-solve-jointly);
; profiles from the host profiler on x = 1 + i/8, y = 0.5 + (i % 7)/4, i < 64.
; Budget -0.4 rewrites both sites, -0.2 only tester_b.
; RUN: mkdir %t/sites
; RUN: %opt %S/Inputs/two_sites.ll %{passfn} %{prof} -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-solve-jointly %{cm} -S -poseidon-cache=%t/sites/both -poseidon-explain=%t/sites/both/ex -poseidon-cost-budget=-0.4 -poseidon-print 2>&1 | FileCheck --check-prefix=SITES-BOTH %s
; RUN: %poseidon_explain %t/sites/both/ex --format budgets | FileCheck --check-prefix=SITES-BUDGETS %s
; RUN: %opt %S/Inputs/two_sites.ll %{passfn} %{prof} -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-solve-jointly %{cm} -S -poseidon-cache=%t/sites/one -poseidon-cost-budget=-0.2 -poseidon-print 2>&1 | FileCheck --check-prefix=SITES-ONE %s

; SITES-BOTH: [poseidon] joint solve: collecting candidates from 2 marked function(s)
; SITES-BOTH: Critical computation cost range: [-4.500000e-01, 0.000000e+00]
; SITES-BOTH-NEXT: DP table contains 4 entries.
; SITES-BOTH: Minimum accuracy cost within budget: 1.154131e-04
; SITES-BOTH-NEXT: Computation cost budget used: -4.500000e-01
; SITES-BOTH: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; SITES-BOTH: [poseidon] pin: tester_a:P0=All FP64(0%) + FP32(100%)
; SITES-BOTH: [poseidon] Finished optimizing preprocess_tester_a
; SITES-BOTH: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; SITES-BOTH: [poseidon] pin: tester_b:P0=All FP64(0%) + FP32(100%)
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
; SITES-ONE: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; SITES-ONE: [poseidon] pin: tester_b:P0=All FP64(0%) + FP32(100%)
; SITES-ONE: [poseidon] Finished optimizing preprocess_tester_b
; SITES-ONE-LABEL: define double @site_a(
; SITES-ONE-NEXT: entry:
; SITES-ONE-NEXT: call double @tester_a(double %x, double %y)
; SITES-ONE-LABEL: define double @site_b(
; SITES-ONE-NEXT: entry:
; SITES-ONE-NEXT: call double @preprocess_tester_b(double %x, double %y)
; SITES-ONE-NOT: define double @preprocess_tester_a(

; A budget share that cannot be met (-400, inf, above 2^30) and a number flag
; given an empty value or nan are errors of a solve that reads them; a compile
; that solves nothing or solves under a target does not read the budget.
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-cost-budget=-400 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-STALE %s
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-solve-jointly -poseidon-cost-budget=inf -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONFINITE %s
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-cost-budget=1e300 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-HUGE -DBUDGET=1.0000000000000001e+300 %s
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-cost-budget=2e9 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-HUGE -DBUDGET=2000000000 %s
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-cost-budget= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-cost-budget %s
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-solve-jointly -poseidon-cost-budget=nan -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-cost-budget %s
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-tau= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-tau %s
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-min-cost-share= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-min-cost-share %s
; RUN: not %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false %{cm} -poseidon-cache= -poseidon-algebraic-max-broken-share= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-NONUMBER -DFLAG=poseidon-algebraic-max-broken-share %s
; RUN: %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon -poseidon-tau= -poseidon-cost-budget= -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-UNUSED --allow-empty %s
; RUN: %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon -poseidon-cost-budget=-400 -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-UNUSED --allow-empty %s
; RUN: %opt %S/Inputs/two_sites.ll %loadPoseidon -passes=poseidon %{prof} -poseidon-enable-algebraic=false -poseidon-enable-precision=false %{cm} -poseidon-cache= -poseidon-tau=0.5 -poseidon-cost-budget=nan -S -o /dev/null 2>&1 | FileCheck --check-prefix=SITES-UNUSED --allow-empty %s

; SITES-STALE-NOT: Starting accuracy DP solver
; SITES-STALE: error: {{.*}}Poseidon: -poseidon-cost-budget=-400 asks a solution to save more than the whole profiled cost; give the share of a site's profiled cost a solution may add, negative for one it has to save (-0.25 saves a quarter)
; SITES-HUGE: error: {{.*}}Poseidon: -poseidon-cost-budget=[[BUDGET]] is above 2^30, the largest share of the profiled cost the solve holds; give a share of at most 1073741824
; SITES-NONFINITE-NOT: Starting accuracy DP solver
; SITES-NONFINITE: error: {{.*}}Poseidon: -poseidon-cost-budget=inf is not a finite number; give the share of a site's profiled cost a solution may add, negative for one it has to save (-0.25 saves a quarter)
; SITES-UNUSED-NOT: error:
; SITES-UNUSED-NOT: -poseidon-cost-budget

; SITES-NONUMBER-NOT: Starting accuracy DP solver
; SITES-NONUMBER: error: {{.*}}Poseidon: -[[FLAG]] was given an empty value or nan; write a number after '=', or leave the flag out for its default
; SITES-NONUMBER-NOT: Starting accuracy DP solver

; The sites of a joint solve need one accuracy target (site's own, else
; -poseidon-tau): different ones are an error; equal ones solve alike wherever
; written.
; RUN: mkdir %t/targets
; RUN: sed 's/@tester_b, double %x, double %y)/@tester_b, double %x, double %y, metadata !"poseidon_tau", double 1.0e-03)/' %t/a.ll > %t/targets/ab_diff.ll
; RUN: sed 's/@tester_b, double %x, double %y)/@tester_b, double %x, double %y, metadata !"poseidon_tau", double 1.0e-02)/' %t/a.ll > %t/targets/ab_same.ll
; RUN: not %opt %t/a.ll %{pass} %{prof} -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-solve-jointly %{cm} -poseidon-cache=%t/targets/c1 -S -o /dev/null 2>&1 | FileCheck --check-prefix=TARGETS-NONE %s
; RUN: not %opt %t/targets/ab_diff.ll %{pass} %{prof} -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-solve-jointly %{cm} -poseidon-cache=%t/targets/c2 -S -o /dev/null 2>&1 | FileCheck --check-prefix=TARGETS-DIFF %s
; RUN: %opt %t/a.ll %{passfn} %{prof} -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-solve-jointly %{cm} -poseidon-cache=%t/targets/c3 -poseidon-tau=1e-2 -poseidon-explain=%t/targets/ex -S -poseidon-print -o %t/targets/flag.out.ll 2> %t/targets/flag.err
; RUN: %python %S/../Inputs/explain/stderr_matches.py %t/targets/flag.err %t/targets/ex | FileCheck --check-prefix=TARGETS-JSON %s
; RUN: FileCheck --check-prefix=TARGETS-SITES %s < %t/targets/ex/tester_a.explain.json
; RUN: %opt %t/targets/ab_same.ll %{passfn} %{prof} -poseidon-enable-algebraic=false -poseidon-enable-precision=true -poseidon-solve-jointly %{cm} -poseidon-cache=%t/targets/c4 -S -poseidon-print -o %t/targets/site.out.ll 2> %t/targets/site.err
; RUN: FileCheck --check-prefix=TARGETS-SOLVE %s < %t/targets/flag.err
; RUN: FileCheck --check-prefix=TARGETS-SOLVE %s < %t/targets/site.err
; RUN: sed -e 's/, metadata !"poseidon_tau", double 1.000000e-02//' -e '/^; ModuleID/d' -e '/^source_filename/d' %t/targets/flag.out.ll > %t/targets/flag.cmp.ll
; RUN: sed -e 's/, metadata !"poseidon_tau", double 1.000000e-02//g' -e '/^; ModuleID/d' -e '/^source_filename/d' %t/targets/site.out.ll > %t/targets/site.cmp.ll
; RUN: diff %t/targets/flag.cmp.ll %t/targets/site.cmp.ll

; TARGETS-NONE: error: {{.*}}Poseidon: the sites of one -poseidon-solve-jointly solve have different accuracy targets: tester_a (1.000000e-02) tester_b (none); give them one target
; TARGETS-DIFF: error: {{.*}}Poseidon: the sites of one -poseidon-solve-jointly solve have different accuracy targets: tester_a (1.000000e-02) tester_b (1.000000e-03); give them one target
; TARGETS-JSON: preprocess_tester_a: rewritten
; TARGETS-JSON-NEXT: preprocess_tester_b: rewritten
; TARGETS-SITES: "selector": "joint",
; TARGETS-SITES-NEXT: "sites": [
; TARGETS-SITES-NEXT: "preprocess_tester_a",
; TARGETS-SITES-NEXT: "preprocess_tester_b"
; TARGETS-SOLVE: [poseidon] preprocess_tester_a,preprocess_tester_b: tau=1.000000e-02 S=4.732407e+03 A0=1.003014e-13 selected cost=-4.500000e-01 accCost=1.154131e-04 rel=2.438782e-08

; Each budget the explanation lists selects its frontier point, per site and
; jointly. Operand A lies below FP16's subnormals, so the frontier starts at
; BF16.
; RUN: %{mm} -poseidon-cache=%t/site -poseidon-explain=%t/site/ex -poseidon-print-dp-table 2>&1 | FileCheck --check-prefix=TABLE %s
; RUN: %poseidon_explain %t/site/ex --format budgets > %t/site/budgets
; RUN: FileCheck --check-prefix=BUDGETS %s < %t/site/budgets
; RUN: for b in $(tr , ' ' < %t/site/budgets); do %{mm} -poseidon-cache=%t/site -poseidon-cost-budget=$b 2>&1 | grep -Fx "[poseidon] Computation cost budget used: $(printf %%e $b)" > /dev/null || { echo "budget $b does not select its point"; exit 1; }; done
; RUN: %{mm} -poseidon-cache=%t/joint -poseidon-explain=%t/joint/ex -poseidon-solve-jointly
; RUN: %poseidon_explain %t/joint/ex --format budgets > %t/joint/budgets
; RUN: diff %t/site/budgets %t/joint/budgets
; RUN: for b in $(tr , ' ' < %t/joint/budgets); do %{mm} -poseidon-cache=%t/joint -poseidon-solve-jointly -poseidon-cost-budget=$b 2>&1 | grep -Fx "[poseidon] Computation cost budget used: $(printf %%e $b)" > /dev/null || { echo "joint budget $b does not select its point"; exit 1; }; done

; TABLE: not offered: direct-dispatch cublas f16/f32: f16 flushes the smallest profiled operand magnitude, 1e-09, to zero
; TABLE-NOT: cublas f16/f32 (#
; TABLE: computation cost -9.882812e-01, accuracy cost 1.254595e-05, steps:
; TABLE-NEXT: matmul[0] -> direct-dispatch cublas bf16/f32 (#7)

; BUDGETS-NOT: -0.9921875
; BUDGETS: {{^}}-0.98828125,
; BUDGETS-NOT: -0.9921875


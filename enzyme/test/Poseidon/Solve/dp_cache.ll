; DEFINE: %{pass} = %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-print \
; DEFINE:   -S -o /dev/null
; DEFINE: %{cm} = -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; DEFINE: %{profiles} = -poseidon-profile-use=%S/../Inputs/pow_term/profiles
; DEFINE: %{out0} = -poseidon-profile-use=%S/../Inputs/pow_term/out0
; RUN: rm -rf %t && mkdir %t

; The DP table cache of the site below. %t/noscale.ll is the module without
; the call of scale_a.
; RUN: sed -e '/%ys = call double @scale_a/d' -e 's/fmul double %k, %ys/fmul double %k, %y/' -e '/^declare double @scale_[ab]/d' %s > %t/noscale.ll

; A table is cached as an entry of its own, filed under the key of everything
; it is solved from. Solved under profile A and then, against the same cache,
; under profile B of the same canonical form (Inputs/pow_term/out0, recorded
; with only the first output seeded), the site is solved again into a second
; entry and picks what a solve under B in an empty cache picks, with the
; budgets of B's frontier; a third compile reads B's table back, and one under
; A reads A's. A flag the table depends on (-poseidon-num-samples) solves into
; another entry; one that only selects from the table (the budget,
; -poseidon-tau-cheapest) does not. An entry without a key, or one that does
; not parse, is reported, solved again and overwritten.
; RUN: mkdir %t/key
; RUN: %opt %t/noscale.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c -poseidon-explain=%t/key/a 2>&1 | FileCheck --check-prefix=KEY-A %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c -poseidon-explain=%t/key/b 2>&1 | FileCheck --check-prefixes=KEY-B,KEY-PICK-B %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/fresh -poseidon-explain=%t/key/fresh 2>&1 | FileCheck --check-prefix=KEY-PICK-B %s
; RUN: ls %t/key/c/dp_*.json | count 2
; RUN: %poseidon_explain %t/key/a --format budgets > %t/key/budgets-a.txt
; RUN: %poseidon_explain %t/key/b --format budgets > %t/key/budgets-b.txt
; RUN: %poseidon_explain %t/key/fresh --format budgets > %t/key/budgets-fresh.txt
; RUN: not diff %t/key/budgets-a.txt %t/key/budgets-fresh.txt
; RUN: diff %t/key/budgets-fresh.txt %t/key/budgets-b.txt
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c 2>&1 | FileCheck --check-prefixes=KEY-HIT,KEY-PICK-B %s
; RUN: %opt %t/noscale.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c 2>&1 | FileCheck --check-prefixes=KEY-HIT,KEY-PICK-A %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/c -poseidon-num-samples=512 2>&1 | FileCheck --check-prefix=KEY-SAMPLES %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.4 -poseidon-cache=%t/key/c -poseidon-num-samples=512 -poseidon-tau-cheapest=false 2>&1 | FileCheck --check-prefix=KEY-HIT %s
; RUN: ls %t/key/c/dp_*.json | count 3
; RUN: cp -r %t/key/fresh %t/key/old
; RUN: for f in %t/key/old/dp_*.json; do echo '{"value": {"frontier": []}}' > $f; done
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/old 2>&1 | FileCheck --check-prefixes=KEY-OLD,KEY-PICK-B %s
; RUN: cp -r %t/key/fresh %t/key/bad
; RUN: for f in %t/key/bad/dp_*.json; do echo '{"key": nan}' > $f; done
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/bad 2>&1 | FileCheck --check-prefixes=KEY-BAD,KEY-PICK-B %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/key/bad 2>&1 | FileCheck --check-prefixes=KEY-HIT,KEY-PICK-B %s

; Each run's lines in the order they are printed.
; KEY-A-NOT: DP cache entry
; KEY-B-NOT: DP cache entry
; KEY-SAMPLES-NOT: DP cache entry
; KEY-HIT-NOT: DP cache entry
; KEY-OLD: [poseidon] DP cache entry for preprocess_tester was written in another cache format; re-solving.
; KEY-BAD: [poseidon] DP cache entry for preprocess_tester is unreadable ({{.*}}dp_{{[0-9A-F]+}}.json: {{.*}}); re-solving.
; KEY-A: Solving the DP for preprocess_tester.
; KEY-B: Solving the DP for preprocess_tester.
; KEY-SAMPLES: Solving the DP for preprocess_tester.
; KEY-OLD: Solving the DP for preprocess_tester.
; KEY-BAD: Solving the DP for preprocess_tester.
; KEY-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; KEY-A: DP table for preprocess_tester cached to {{.*}}/key/c/dp_{{[0-9A-F]+}}.json
; KEY-B: DP table for preprocess_tester cached to {{.*}}/key/c/dp_{{[0-9A-F]+}}.json
; KEY-OLD: DP table for preprocess_tester cached to
; KEY-BAD: DP table for preprocess_tester cached to
; KEY-HIT-NOT: cached to
; KEY-A: Applying solution for precision change All FP64(70%) + FP32(30%)
; KEY-PICK-A: Applying solution for precision change All FP64(70%) + FP32(30%)
; KEY-PICK-B: Applying solution for precision change Funcs FP64(70%) + FP32(30%)

; Each compile starts from the first one's cache and changes one input of the
; table (a profile byte, a cost-model row, a solve flag, a constant of the
; body, the function it calls): it solves again into an entry of its own. An
; unchanged compile reads the table back.
; RUN: mkdir %t/parts %t/parts/p
; RUN: sed '0,/Exec = 64/s//Exec = 65/' %S/../Inputs/pow_term/profiles/tester.fpprofile > %t/parts/p/tester.fpprofile
; RUN: sed 's/^fadd,float,1$/fadd,float,2/' %cost_models/cm_cpu_native.csv > %t/parts/cm.csv
; RUN: sed 's/fmul double %%p, 1.000000e-20/fmul double %%p, 2.000000e-20/' %s > %t/parts/const.ll
; RUN: sed 's/call double @scale_a(/call double @scale_b(/' %s > %t/parts/callee.ll
; RUN: not cmp -s %S/../Inputs/pow_term/profiles/tester.fpprofile %t/parts/p/tester.fpprofile
; RUN: not cmp -s %cost_models/cm_cpu_native.csv %t/parts/cm.csv
; RUN: not cmp -s %s %t/parts/const.ll
; RUN: not cmp -s %s %t/parts/callee.ll
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/base 2>&1 | FileCheck --check-prefix=PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/same
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/same 2>&1 | FileCheck --check-prefix=PARTS-HIT %s
; RUN: cp -r %t/parts/base %t/parts/profile
; RUN: %opt %s %{pass} -poseidon-profile-use=%t/parts/p %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/profile 2>&1 | FileCheck --check-prefix=PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/costmodel
; RUN: %opt %s %{pass} %{profiles} -poseidon-cost-model=%t/parts/cm.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/costmodel 2>&1 | FileCheck --check-prefix=PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/flag
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/flag -poseidon-random-seed=7 2>&1 | FileCheck --check-prefix=PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/const
; RUN: %opt %t/parts/const.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/const 2>&1 | FileCheck --check-prefix=PARTS-SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/callee
; RUN: %opt %t/parts/callee.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/callee 2>&1 | FileCheck --check-prefix=PARTS-SOLVED %s
; RUN: ls %t/parts/same/dp_*.json | count 1
; RUN: for d in profile costmodel flag const callee; do ls %t/parts/$d/dp_*.json | count 2; done

; PARTS-SOLVED-NOT: DP cache entry
; PARTS-SOLVED: Solving the DP for preprocess_tester.
; PARTS-SOLVED: DP table for preprocess_tester cached to
; PARTS-HIT-NOT: DP cache entry
; PARTS-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; PARTS-HIT-NOT: Solving the DP

; A cached entry whose key has parts this compile does not names the first of
; them by name, whatever order the stored key object holds them in.
; RUN: mkdir %t/extra
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/extra/c 2>&1 | FileCheck --check-prefix=EXTRA-SOLVED %s
; RUN: for f in %t/extra/c/dp_*.json; do sed -i 's/"key": {/"key": {"zz extra": "1", "aa extra": "1", "mm extra": "1",/' $f; done
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/extra/c 2>&1 | FileCheck --check-prefixes=EXTRA,EXTRA-SOLVED %s

; EXTRA: [poseidon] DP cache entry for preprocess_tester was solved for a different aa extra; re-solving.
; EXTRA-SOLVED: DP table for preprocess_tester cached to

; An entry whose key names no compiler, or another one, is a miss, and the
; next solve replaces it. The cached table gets other numbers (the point the
; solve picks made the worst) and loses its compiler; the next compile solves
; again.
; RUN: mkdir %t/compiler
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/compiler/c 2>&1 | FileCheck --check-prefixes=COMPILER-SOLVE,COMPILER-PICK %s
; RUN: %python -c "import glob, json; p = glob.glob('%t/compiler/c/dp_*.json')[0]; e = json.load(open(p)); del e['key']['compiler']; [q.update({'accuracy_cost': 1}) for q in e['value']['frontier'] if q['cost'] == -536870912]; json.dump(e, open(p, 'w'))"
; RUN: cat %t/compiler/c/dp_*.json | FileCheck --check-prefix=COMPILER-EDITED %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/compiler/c 2>&1 | FileCheck --check-prefixes=COMPILER-FOREIGN,COMPILER-SOLVE,COMPILER-PICK %s
; RUN: ls %t/compiler/c/dp_*.json | count 1

; COMPILER-EDITED: {"accuracy_cost": 1, "cost": -536870912,

; COMPILER-FOREIGN-NOT: Loaded DP tables from cache
; COMPILER-FOREIGN: [poseidon] DP cache entry for preprocess_tester was solved for a different compiler; re-solving.
; COMPILER-SOLVE: Solving the DP for preprocess_tester.
; COMPILER-SOLVE: DP table for preprocess_tester cached to

; COMPILER-PICK: Applying solution for precision change Funcs FP64(70%) + FP32(30%) (#10)

; An entry cut off in the middle (an interrupted write) is reported, solved
; again and overwritten; the compile after that reads it back.
; RUN: mkdir %t/trunc
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=TRUNC-SOLVE,TRUNC-PICK %s
; RUN: for f in %t/trunc/c/dp_*.json; do head -c $(( $(wc -c < $f) / 2 )) $f > %t/trunc/half.json && mv %t/trunc/half.json $f; done
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=TRUNC,TRUNC-SOLVE,TRUNC-PICK %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=TRUNC-HIT,TRUNC-PICK %s

; TRUNC: [poseidon] DP cache entry for preprocess_tester is unreadable ({{.*}}dp_{{[0-9A-F]+}}.json: {{.*}}); re-solving.
; TRUNC-SOLVE: Solving the DP for preprocess_tester.
; TRUNC-SOLVE: DP table for preprocess_tester cached to
; TRUNC-HIT-NOT: DP cache entry
; TRUNC-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; TRUNC-HIT-NOT: cached to
; TRUNC-PICK: Applying solution for precision change Funcs FP64(70%) + FP32(30%) (#10)

; Two compiles of different units into one cache directory at once, here the
; site with and without the call of scale_a, keep both tables: each is an
; entry of its own, and each compile after them reads its own back.
; RUN: mkdir %t/shared
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/shared/c 2> %t/shared/with.err & with=$!; %opt %t/noscale.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/shared/c 2> %t/shared/without.err; wait $with
; RUN: FileCheck --check-prefix=SHARED-SOLVE %s < %t/shared/with.err
; RUN: FileCheck --check-prefix=SHARED-SOLVE %s < %t/shared/without.err
; RUN: ls %t/shared/c/dp_*.json | count 2
; RUN: %opt %s %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/shared/c 2>&1 | FileCheck --check-prefix=SHARED-HIT %s
; RUN: %opt %t/noscale.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/shared/c 2>&1 | FileCheck --check-prefix=SHARED-HIT %s

; SHARED-SOLVE: Solving the DP for preprocess_tester.
; SHARED-SOLVE: DP table for preprocess_tester cached to
; SHARED-HIT-NOT: DP cache entry
; SHARED-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.

define void @tester(double %T, double %y, ptr %o1, ptr %o2) #0 {
entry:
  %l = call double @llvm.log.f64(double %T)
  %nl = fneg double %l
  %e = call double @llvm.fma.f64(double %nl, double 1.790000e-02, double -1.484800e+00)
  %p = call double @llvm.pow.f64(double %T, double %e)
  %k = fmul double %p, 1.000000e-20
  %ys = call double @scale_a(double %y)
  %r1 = fmul double %k, %ys
  %h = fmul double %nl, %y
  %r2 = fadd double %h, %T
  store double %r1, ptr %o1, align 8
  store double %r2, ptr %o2, align 8
  ret void
}

define void @site(double %T, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %T, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2)
  ret void
}

declare double @scale_a(double)
declare double @scale_b(double)
declare double @llvm.log.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.pow.f64(double, double)
declare void @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64" }

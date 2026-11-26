; DEFINE: %{pass} = %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-enable-precision=true -poseidon-print -S -o /dev/null
; DEFINE: %{cm} = -poseidon-cost-model=%cost_models/cm_cpu_native.csv
; DEFINE: %{profiles} = -poseidon-profile-use=%S/../Inputs/pow_term/profiles -poseidon-early-prune
; DEFINE: %{out0} = -poseidon-profile-use=%S/../Inputs/pow_term/out0 -poseidon-early-prune
; DEFINE: %{herbie} = %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-enable-rewrite=true \
; DEFINE:   -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-cost-budget=1e9 -poseidon-print
; DEFINE: %{keys} = env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %S/../Inputs/piecewise_rewrite/tester.ll \
; DEFINE:   %{herbie} -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh
; DEFINE: %{array} = %opt < %t/array.ll %{herbie} -poseidon-profile-use=%S/../Inputs/array_cores/profiles \
; DEFINE:   -poseidon-early-prune -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -S \
; DEFINE:   -o /dev/null
; RUN: rm -rf %t && split-file %s %t
; RUN: sed -e '/%ys = /d' -e 's/%k, %ys/%k, %y/' -e '/^declare double @scale_/d' %t/tester.ll > %t/noscale.ll
; RUN: sed 's/1.000000e-20/2.000000e-20/' %t/tester.ll > %t/const.ll
; RUN: sed 's/@scale_a(double %y)/@scale_b(double %y)/' %t/tester.ll > %t/callee.ll

; The DP table cache of the site below (noscale.ll: without the call of
; scale_a). An entry is filed under the key of everything it is solved from:
; profile B of the same canonical form (pow_term/out0) solves into a second
; entry and picks what a fresh solve under B picks; each profile then reads its
; own back. -poseidon-num-samples is part of the key; the budget and
; -poseidon-tau-cheapest are not. An entry without a key or that does not parse
; is reported, solved again and overwritten.
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

; Changing one input of the table (profile byte, cost-model row, solve flag,
; constant, callee) solves into an entry of its own; no change reads it back.
; RUN: mkdir %t/parts %t/parts/p
; RUN: sed '0,/Exec = 64/s//Exec = 65/' %S/../Inputs/pow_term/profiles/tester.fpprofile > %t/parts/p/tester.fpprofile
; RUN: sed 's/^fadd,float,1$/fadd,float,2/' %cost_models/cm_cpu_native.csv > %t/parts/cm.csv
; RUN: not cmp -s %S/../Inputs/pow_term/profiles/tester.fpprofile %t/parts/p/tester.fpprofile
; RUN: not cmp -s %cost_models/cm_cpu_native.csv %t/parts/cm.csv
; RUN: %opt %t/tester.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/base 2>&1 | FileCheck --check-prefixes=NOENTRY,SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/same
; RUN: %opt %t/tester.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/same 2>&1 | FileCheck --check-prefix=HIT %s
; RUN: cp -r %t/parts/base %t/parts/profile
; RUN: %opt %t/tester.ll %{pass} -poseidon-profile-use=%t/parts/p -poseidon-early-prune %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/profile 2>&1 | FileCheck --check-prefixes=NOENTRY,SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/costmodel
; RUN: %opt %t/tester.ll %{pass} %{profiles} -poseidon-cost-model=%t/parts/cm.csv -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/costmodel 2>&1 | FileCheck --check-prefixes=NOENTRY,SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/flag
; RUN: %opt %t/tester.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/flag -poseidon-random-seed=7 2>&1 | FileCheck --check-prefixes=NOENTRY,SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/const
; RUN: %opt %t/const.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/const 2>&1 | FileCheck --check-prefixes=NOENTRY,SOLVED %s
; RUN: cp -r %t/parts/base %t/parts/callee
; RUN: %opt %t/callee.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/parts/callee 2>&1 | FileCheck --check-prefixes=NOENTRY,SOLVED %s
; RUN: ls %t/parts/same/dp_*.json | count 1
; RUN: for d in profile costmodel flag const callee; do ls %t/parts/$d/dp_*.json | count 2; done

; A stored key with parts this compile lacks names the first of them by name.
; RUN: mkdir %t/extra
; RUN: %opt %t/tester.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/extra/c 2>&1 | FileCheck --check-prefix=SOLVED %s
; RUN: for f in %t/extra/c/dp_*.json; do sed -i 's/"key": {/"key": {"zz extra": "1", "aa extra": "1", "mm extra": "1",/' $f; done
; RUN: %opt %t/tester.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/extra/c 2>&1 | FileCheck --check-prefixes=EXTRA,SOLVED %s

; EXTRA: [poseidon] DP cache entry for preprocess_tester was solved for a different aa extra; re-solving.

; An entry whose key names no compiler (and whose numbers were edited) is a
; miss that the next solve replaces.
; RUN: mkdir %t/compiler
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/compiler/c 2>&1 | FileCheck --check-prefixes=SOLVED,PICK10 %s
; RUN: %python -c "import glob, json; p = glob.glob('%t/compiler/c/dp_*.json')[0]; e = json.load(open(p)); del e['key']['compiler']; [q.update({'accuracy_cost': 1}) for q in e['value']['frontier'] if q['cost'] == -536870912]; json.dump(e, open(p, 'w'))"
; RUN: cat %t/compiler/c/dp_*.json | FileCheck --check-prefix=COMPILER-EDITED %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/compiler/c 2>&1 | FileCheck --check-prefixes=COMPILER-FOREIGN,SOLVED,PICK10 %s
; RUN: ls %t/compiler/c/dp_*.json | count 1

; COMPILER-EDITED: {"accuracy_cost": 1, "cost": -536870912,

; COMPILER-FOREIGN-NOT: Loaded DP tables from cache
; COMPILER-FOREIGN: [poseidon] DP cache entry for preprocess_tester was solved for a different compiler; re-solving.

; A truncated entry is reported, solved again and overwritten, then read back.
; RUN: mkdir %t/trunc
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=SOLVED,PICK10 %s
; RUN: for f in %t/trunc/c/dp_*.json; do head -c $(( $(wc -c < $f) / 2 )) $f > %t/trunc/half.json && mv %t/trunc/half.json $f; done
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=TRUNC,SOLVED,PICK10 %s
; RUN: %opt %t/noscale.ll %{pass} %{out0} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/trunc/c 2>&1 | FileCheck --check-prefixes=HIT,PICK10 %s

; TRUNC: [poseidon] DP cache entry for preprocess_tester is unreadable ({{.*}}dp_{{[0-9A-F]+}}.json: {{.*}}); re-solving.

; Two concurrent compiles of different units into one cache keep both tables.
; RUN: mkdir %t/shared
; RUN: %opt %t/tester.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/shared/c 2> %t/shared/with.err & with=$!; %opt %t/noscale.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/shared/c 2> %t/shared/without.err; wait $with
; RUN: FileCheck --check-prefix=SOLVED %s < %t/shared/with.err
; RUN: FileCheck --check-prefix=SOLVED %s < %t/shared/without.err
; RUN: ls %t/shared/c/dp_*.json | count 2
; RUN: %opt %t/tester.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/shared/c 2>&1 | FileCheck --check-prefix=HIT %s
; RUN: %opt %t/noscale.ll %{pass} %{profiles} %{cm} -poseidon-cost-budget=-0.09 -poseidon-cache=%t/shared/c 2>&1 | FileCheck --check-prefix=HIT %s

; A cached DP table applies, under the same rewrite id, what its solve picked.
; RUN: mkdir %t/ids
; RUN: %{keys} -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-early-prune -poseidon-cache=%t/ids/c -poseidon-explain=%t/ids/solve -S -o /dev/null 2>&1 | FileCheck --check-prefix=IDS-SOLVE %s
; RUN: %{keys} -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-early-prune -poseidon-cache=%t/ids/c -poseidon-explain=%t/ids/hit -S -o /dev/null 2>&1 | FileCheck --check-prefix=IDS-HIT %s
; RUN: %python -c "import json, sys; print(*json.load(open(sys.argv[1]))['result']['applied'])" %t/ids/solve/tester.explain.json > %t/ids/solve.id
; RUN: FileCheck --check-prefix=IDS-ID %s < %t/ids/solve.id
; RUN: FileCheck --check-prefix=IDS-EXPLAIN -DID=$(cat %t/ids/solve.id) %s < %t/ids/solve/tester.explain.json
; RUN: FileCheck --check-prefix=IDS-EXPLAIN -DID=$(cat %t/ids/solve.id) %s < %t/ids/hit/tester.explain.json

; IDS-SOLVE: Solving the DP for preprocess_tester.

; IDS-HIT: Loaded DP tables from cache

; IDS-ID: {{^R[0-9]+_[0-9]+$}}

; IDS-EXPLAIN: "result": {
; IDS-EXPLAIN-NEXT: "applied": [
; IDS-EXPLAIN-NEXT: "[[ID]]"
; IDS-EXPLAIN: "fpcore": "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))",
; IDS-EXPLAIN-NOT: "id":
; IDS-EXPLAIN: "id": "[[ID]]",

; The table is keyed on its candidates: Herbie answering differently (the array
; core times out) solves again; that uncached answer, repeated, reads it back.
; RUN: mkdir %t/set
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/array_cores %{array} -poseidon-cache=%t/set/c 2>&1 | FileCheck --check-prefix=SET-FIRST %s
; RUN: rm %t/set/c/herbie_*.json
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/array_cores/timeout %{array} -poseidon-cache=%t/set/c 2>&1 | FileCheck --check-prefix=SET-DRIFT %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/array_cores/timeout %{array} -poseidon-cache=%t/set/c 2>&1 | FileCheck --check-prefix=SET-HIT %s

; SET-FIRST: Solving the DP for preprocess_tester.
; SET-FIRST: Applying solution for (let* ((t1 (- (sqrt (+ (* v0 v0) (* v1 v1))) v0))) (array t1 (* t1 0.5)))

; SET-DRIFT: array core returned nothing
; SET-DRIFT-NOT: DP cache entry
; SET-DRIFT: Solving the DP for preprocess_tester.
; SET-DRIFT-NOT: (array t1
; SET-DRIFT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0)

; SET-HIT: timed out on 2 of 4 FPCore(s) {{.*}}; the answer is not cached, so the next compile runs Herbie again{{$}}
; SET-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; SET-HIT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0)

; The checks of the cache runs above that solve (SOLVED, after NOENTRY when no
; entry may be reported), read the table back (HIT) and pick point #10 (PICK10).
; NOENTRY-NOT: DP cache entry
; SOLVED: Solving the DP for preprocess_tester.
; SOLVED: DP table for preprocess_tester cached to
; HIT-NOT: DP cache entry
; HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; HIT-NOT: Solving the DP
; HIT-NOT: cached to
; PICK10: Applying solution for precision change Funcs FP64(70%) + FP32(30%) (#10)

;--- tester.ll
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

;--- array.ll
define void @tester(double %x, double %y, ptr %o1, ptr %o2, ptr %o3) #0 {
entry:
  %xx = fmul double %x, %x
  %yy = fmul double %y, %y
  %s = fadd double %xx, %yy
  %r = call double @llvm.sqrt.f64(double %s)
  %d = fsub double %r, %x
  %h = fmul double %d, 5.000000e-01
  %z = fmul double %xx, 2.500000e+00
  store double %d, ptr %o1, align 8
  store double %h, ptr %o2, align 8
  store double %z, ptr %o3, align 8
  ret void
}

define void @site(double %x, double %y, ptr %o1, ptr %do1, ptr %o2, ptr %do2, ptr %o3, ptr %do3) #0 {
entry:
  tail call void (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester, double %x, double %y, metadata !"enzyme_dup", ptr %o1, ptr %do1, metadata !"enzyme_dup", ptr %o2, ptr %do2, metadata !"enzyme_dup", ptr %o3, ptr %do3)
  ret void
}

declare double @llvm.sqrt.f64(double)
declare void @__poseidon_fp_optimize(ptr, ...)
attributes #0 = { "target-cpu"="x86-64" }

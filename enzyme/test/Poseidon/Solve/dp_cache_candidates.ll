; DEFINE: %{herbie} = %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-herbie=true -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9
; DEFINE: %{keys} = env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %t/regime.ll %{herbie} \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh
; DEFINE: %{array} = %opt < %t/array.ll %{herbie} \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/array_cores/profiles \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/array_cores/fake-herbie.sh -poseidon-herbie-processes=1 \
; DEFINE:   -S -o /dev/null
; RUN: rm -rf %t && split-file %s %t

; A DP table read back from the cache applies the candidates its solve picked,
; and it is keyed on the candidates it was solved over.

; The replayed Herbie result lists a regime split first and its exact else arm
; second. Pricing drops the split (its then arm is unreachable in the profiled
; box), so the solve picks the second candidate at position 0 of the surviving
; list. A compile that reuses the table skips pricing and sees both
; candidates; it applies the same rewrite, not position 0.
; RUN: mkdir %t/keys
; RUN: %{keys} -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-print -poseidon-cache=%t/keys/c -S 2>&1 | FileCheck --check-prefix=KEYS-SOLVE %s
; RUN: %{keys} -poseidon-profile-use=%S/../Inputs/regime_split/profiles -poseidon-print -poseidon-cache=%t/keys/c -S 2>&1 | FileCheck --check-prefix=KEYS-HIT %s

; KEYS-SOLVE: dropped (unpriced arm)
; KEYS-SOLVE: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)
; KEYS-SOLVE: define double @preprocess_tester
; KEYS-SOLVE-NOT: select
; KEYS-SOLVE: fdiv
; KEYS-SOLVE-NOT: select
; KEYS-SOLVE: ret double

; KEYS-HIT: Loaded DP tables from cache
; KEYS-HIT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)
; KEYS-HIT: define double @preprocess_tester
; KEYS-HIT-NOT: select
; KEYS-HIT: fdiv
; KEYS-HIT-NOT: select
; KEYS-HIT: ret double

; With the same module and answer, the rewrite id the solve's _rewrites.json
; gives the applied rewrite is the id both compiles' explain reports give it.
; RUN: mkdir %t/ids %t/ids/prof
; RUN: sed '1i CanonicalHash = 1fa032681b79bf6d' %S/../Inputs/regime_split/profiles/preprocess_tester.fpprofile > %t/ids/prof/preprocess_tester.fpprofile
; RUN: %{keys} -poseidon-profile-use=%t/ids/prof -poseidon-cache=%t/ids/c -poseidon-report-path=%t/ids/solve -S -o /dev/null 2>&1 | FileCheck --check-prefix=IDS-SOLVE %s
; RUN: %{keys} -poseidon-profile-use=%t/ids/prof -poseidon-cache=%t/ids/c -poseidon-report-path=%t/ids/hit -S -o /dev/null 2>&1 | FileCheck --check-prefix=IDS-HIT %s
; RUN: FileCheck --check-prefix=IDS-REWRITES %s < %t/ids/solve/preprocess_tester_rewrites.json
; RUN: sed -n 's/^ *"id": "\(R[0-9]*_[0-9]*\)",$/\1/p' %t/ids/solve/preprocess_tester_rewrites.json > %t/ids/solve.id
; RUN: FileCheck --check-prefix=IDS-EXPLAIN -DID=$(cat %t/ids/solve.id) %s < %t/ids/solve/preprocess_tester.explain.json
; RUN: FileCheck --check-prefix=IDS-EXPLAIN -DID=$(cat %t/ids/solve.id) %s < %t/ids/hit/preprocess_tester.explain.json

; IDS-SOLVE: Solving the DP for preprocess_tester.

; IDS-HIT: Loaded DP tables from cache

; IDS-REWRITES: "id": "R{{[0-9]+}}_{{[0-9]+}}",
; IDS-REWRITES: "rewritten_expr": "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))"
; IDS-REWRITES: "total_rewrites": 1

; IDS-EXPLAIN: "result": {
; IDS-EXPLAIN-NEXT: "applied": [
; IDS-EXPLAIN-NEXT: "[[ID]]"
; IDS-EXPLAIN: "fpcore": "(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))",
; IDS-EXPLAIN-NOT: "id":
; IDS-EXPLAIN: "id": "[[ID]]",

; Herbie can answer the same cores differently: its results are deleted and
; the array core times out on the search that replaces them, so the outputs
; are sent separately and the site has other candidates under unchanged
; inputs. The table is solved again rather than replayed against candidates it
; never priced.
; RUN: mkdir %t/set
; RUN: %{array} -poseidon-cache=%t/set/c 2>&1 | FileCheck --check-prefix=SET-FIRST %s
; RUN: rm %t/set/c/cachedHerbieOutput_*
; RUN: env FAKE_HERBIE_ARRAY=timeout %{array} -poseidon-cache=%t/set/c 2>&1 | FileCheck --check-prefix=SET-DRIFT %s
; RUN: %{array} -poseidon-cache=%t/set/c 2>&1 | FileCheck --check-prefix=SET-HIT %s

; SET-FIRST: Solving the DP for preprocess_tester.
; SET-FIRST: Applying solution for (let* ((t1 (- (sqrt (+ (* v0 v0) (* v1 v1))) v0))) (array t1 (* t1 0.5)))

; SET-DRIFT: array core returned nothing
; SET-DRIFT: [poseidon] DP cache entry for preprocess_tester was solved for a different candidate set; re-solving.
; SET-DRIFT: Solving the DP for preprocess_tester.
; SET-DRIFT-NOT: (array t1
; SET-DRIFT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0)

; SET-HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.
; SET-HIT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0)

;--- regime.ll
define double @tester(double %x, double %y) #0 {
entry:
  %xx = fmul double %x, %x
  %yy = fmul double %y, %y
  %s = fadd double %xx, %yy
  %r = call double @llvm.sqrt.f64(double %s)
  %d = fsub double %r, %x
  ret double %d
}

define double @test_opt(double %x, double %y) {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester, double %x, double %y)
  ret double %0
}

declare double @llvm.sqrt.f64(double)
declare double @__poseidon_fp_optimize(double (double, double)*, ...)

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

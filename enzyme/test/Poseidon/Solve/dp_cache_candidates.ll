; DEFINE: %{herbie} = %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-rewrite=true -poseidon-enable-precision=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -poseidon-print
; DEFINE: %{keys} = env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %opt < %S/../Inputs/piecewise_rewrite/tester.ll %{herbie} \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh
; DEFINE: %{array} = %opt < %t/array.ll %{herbie} \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/array_cores/profiles \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 \
; DEFINE:   -S -o /dev/null
; RUN: rm -rf %t && split-file %s %t

; A DP table read back from the cache applies the candidates its solve picked,
; and it is keyed on the candidates it was solved over.

; With the same module and answer, the rewrite id the solve's explanation
; gives the applied rewrite is the one the explanation of the compile that
; reads the table back gives it.
; RUN: mkdir %t/ids
; RUN: %{keys} -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-cache=%t/ids/c -poseidon-explain=%t/ids/solve -S -o /dev/null 2>&1 | FileCheck --check-prefix=IDS-SOLVE %s
; RUN: %{keys} -poseidon-profile-use=%S/../Inputs/piecewise_rewrite/profiles -poseidon-cache=%t/ids/c -poseidon-explain=%t/ids/hit -S -o /dev/null 2>&1 | FileCheck --check-prefix=IDS-HIT %s
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

; Herbie can answer the same cores differently: its results are deleted and
; the array core times out on the search that replaces them, so the outputs
; are sent separately and the site has other candidates under unchanged
; inputs. The table is solved again rather than replayed against candidates it
; never priced. The answer with the timeout is not cached, so the next compile
; asks Herbie again; answered the same, it reads that table back.
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

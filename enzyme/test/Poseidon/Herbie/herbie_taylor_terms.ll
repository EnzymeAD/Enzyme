; DEFINE: %{opt} = %opt < %S/../Inputs/regime_split/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/regime_split/profiles \
; DEFINE:   -poseidon-enable-herbie=true -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-cost-budget=1e9 -S -o /dev/null
; RUN: rm -rf %t && mkdir %t && touch %t/calls24.log %t/calls4.log %t/calls46.log
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/herbie_taylor_terms FAKE_HERBIE_LOG=%t/calls24.log %{opt} \
; RUN:   -poseidon-herbie-binary=%S/../Inputs/herbie_taylor_terms/fake-herbie.sh -poseidon-herbie-processes=1 \
; RUN:   -poseidon-herbie-num-taylor-terms=4,2 -poseidon-cache=%t/c -poseidon-report-path=%t/r24 2>&1 \
; RUN:   | FileCheck --check-prefix=SEARCH24 %s
; RUN: FileCheck --check-prefix=CALLS24 %s < %t/calls24.log
; RUN: FileCheck --check-prefix=UNION24 %s < %t/r24/preprocess_tester.txt
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/herbie_taylor_terms FAKE_HERBIE_LOG=%t/calls4.log %{opt} \
; RUN:   -poseidon-herbie-binary=%S/../Inputs/herbie_taylor_terms/fake-herbie.sh -poseidon-herbie-processes=1 \
; RUN:   -poseidon-herbie-num-taylor-terms=4 -poseidon-cache=%t/c -poseidon-report-path=%t/r4 2>&1 \
; RUN:   | FileCheck --check-prefix=SEARCH4 %s
; RUN: FileCheck --allow-empty --check-prefix=NOCALL %s < %t/calls4.log
; RUN: FileCheck --check-prefix=UNION4 %s < %t/r4/preprocess_tester.txt
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/herbie_taylor_terms FAKE_HERBIE_LOG=%t/calls46.log %{opt} \
; RUN:   -poseidon-herbie-binary=%S/../Inputs/herbie_taylor_terms/fake-herbie.sh -poseidon-herbie-processes=1 \
; RUN:   -poseidon-herbie-num-taylor-terms=6,4 -poseidon-cache=%t/c -poseidon-report-path=%t/r46 2>&1 \
; RUN:   | FileCheck --check-prefix=SEARCH46 %s
; RUN: FileCheck --check-prefix=CALLS46 %s < %t/calls46.log
; RUN: FileCheck --check-prefix=UNION46 %s < %t/r46/preprocess_tester.txt
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh \
; RUN:   -poseidon-cache=%t/unset 2>&1 | FileCheck --check-prefix=UNSET %s
; RUN: not %{opt} -poseidon-herbie-binary=%S/../Inputs/herbie_taylor_terms/no-option-herbie.sh -poseidon-herbie-processes=1 \
; RUN:   -poseidon-herbie-num-taylor-terms=4 -poseidon-cache=%t/old 2>&1 | FileCheck --check-prefix=NOOPTION %s

; -poseidon-herbie-num-taylor-terms searches a subgraph once per term count,
; each search passing Herbie --num-taylor-terms and cached under its own key,
; and gives each core the answers of all of them, the fewest terms first and
; each expression once, whatever order the counts are listed in. The stand-in
; Herbie answers B then A at 2 terms, C then B at 4 and C then E at 6 (A, a
; regime split, is dropped in pricing, so the report skips its ordinal). A
; count searched before is read from the cache in any list. Unset, nothing is
; passed and the cache key is the one UNSET pins. A Herbie that does not take
; the option stops the compile.

; SEARCH24: [poseidon] Herbie subgraph 0 of preprocess_tester at 2 Taylor terms: 2 FPCore(s) in 1 process(es) of 2 worker(s), per-core timeout {{.*}}, key [[KEY2:[0-9A-F]+]]
; SEARCH24: Saved Herbie output to cache file {{.*}}cachedHerbieOutput_default_[[KEY2]].txt
; SEARCH24: [poseidon] Herbie subgraph 0 of preprocess_tester at 4 Taylor terms: 2 FPCore(s) in 1 process(es) of 2 worker(s), per-core timeout {{.*}}, key [[KEY4:[0-9A-F]+]]
; SEARCH24: Saved Herbie output to cache file {{.*}}cachedHerbieOutput_default_[[KEY4]].txt

; CALLS24: report {{.*}} --enable reduce:dag-cost --num-taylor-terms 2 {{[^ ]+}} {{[^ ]+$}}
; CALLS24-NEXT: report {{.*}} --enable reduce:dag-cost --num-taylor-terms 4 {{[^ ]+}} {{[^ ]+$}}
; CALLS24-NOT: report

; UNION24: Expression #0: (- (sqrt (+ (* v0 v0) (* v1 v1))) v0)
; UNION24: Candidates: 2
; UNION24-NEXT: [0] (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))
; UNION24-NEXT: [2] (/.f64 (*.f64 v1 v1) (+.f64 v0 (sqrt.f64 (+.f64 (*.f64 v1 v1) (*.f64 v0 v0)))))
; UNION24-NEXT: Expression #1:

; SEARCH4-NOT: FPCore(s) in
; SEARCH4: Using cached Herbie output from {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt
; SEARCH4-NOT: FPCore(s) in

; NOCALL-NOT: report

; UNION4: Candidates: 2
; UNION4-NEXT: [0] (/.f64 (*.f64 v1 v1) (+.f64 v0 (sqrt.f64 (+.f64 (*.f64 v1 v1) (*.f64 v0 v0)))))
; UNION4-NEXT: [1] (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))
; UNION4-NEXT: Expression #1:

; SEARCH46: Using cached Herbie output from
; SEARCH46: [poseidon] Herbie subgraph 0 of preprocess_tester at 6 Taylor terms: 2 FPCore(s) in
; SEARCH46-NOT: at 4 Taylor terms: 2 FPCore(s) in

; CALLS46: report {{.*}} --num-taylor-terms 6 {{[^ ]+}} {{[^ ]+$}}
; CALLS46-NOT: report

; UNION46: Candidates: 3
; UNION46-NEXT: [0] (/.f64 (*.f64 v1 v1) (+.f64 v0 (sqrt.f64 (+.f64 (*.f64 v1 v1) (*.f64 v0 v0)))))
; UNION46-NEXT: [1] (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))
; UNION46-NEXT: [2] (-.f64 (sqrt.f64 (fma.f64 v0 v0 (*.f64 v1 v1))) v0)
; UNION46-NEXT: Expression #1:

; UNSET-NOT: Taylor terms
; UNSET: Saved Herbie output to cache file {{.*}}cachedHerbieOutput_default_D7072D91AD47CF1E.txt

; NOOPTION: error: {{.*}}Poseidon: the cache {{.*}}old holds no Herbie answer for subgraph 0 of preprocess_tester at {{[0-9]+}} Taylor terms, and the Herbie at {{.*}}no-option-herbie.sh cannot run (-poseidon-herbie-num-taylor-terms passes it --num-taylor-terms, which it does not take ('report --help' does not list it); use a Herbie built with the option, or leave the flag out)

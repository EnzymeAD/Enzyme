; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles \
; DEFINE:   -poseidon-enable-rewrite=true -poseidon-enable-precision=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-cost-budget=1e9 -poseidon-print -S -o /dev/null
; DEFINE: %{rewrites} = %python -c "import json, sys; print('\\n'.join(a['id'] + ' ' + a['fpcore'] for u in json.load(open(sys.argv[1]))['subgraphs'] for a in u['alternatives'] if a.get('id', '').startswith('R0_')))"
; RUN: rm -rf %t && mkdir %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/herbie_taylor_terms %{opt} \
; RUN:   -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 \
; RUN:   -poseidon-herbie-num-taylor-terms=4,2 -poseidon-cache=%t/c -poseidon-explain=%t/r24 > %t/out24 2>&1
; RUN: FileCheck --check-prefix=SEARCH24 %s < %t/out24
; RUN: grep '^fake-herbie:' %t/out24 | FileCheck --check-prefix=CALLS24 %s
; RUN: %{rewrites} %t/r24/tester.explain.json | FileCheck --check-prefix=UNION24 %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/herbie_taylor_terms %{opt} \
; RUN:   -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 \
; RUN:   -poseidon-herbie-num-taylor-terms=4 -poseidon-cache=%t/c -poseidon-explain=%t/r4 > %t/out4 2>&1
; RUN: FileCheck --check-prefix=SEARCH4 %s < %t/out4
; RUN: not grep '^fake-herbie:' %t/out4
; RUN: %{rewrites} %t/r4/tester.explain.json | FileCheck --check-prefix=UNION4 %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/herbie_taylor_terms %{opt} \
; RUN:   -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 \
; RUN:   -poseidon-herbie-num-taylor-terms=6,4 -poseidon-cache=%t/c -poseidon-explain=%t/r46 > %t/out46 2>&1
; RUN: FileCheck --check-prefix=SEARCH46 %s < %t/out46
; RUN: grep '^fake-herbie:' %t/out46 | FileCheck --check-prefix=CALLS46 %s
; RUN: %{rewrites} %t/r46/tester.explain.json | FileCheck --check-prefix=UNION46 %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh \
; RUN:   -poseidon-cache=%t/unset 2>&1 | FileCheck --check-prefix=UNSET %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/herbie_taylor_terms/no_option not %{opt} -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 \
; RUN:   -poseidon-herbie-num-taylor-terms=4 -poseidon-cache=%t/old 2>&1 | FileCheck --check-prefix=NOOPTION %s

; -poseidon-herbie-num-taylor-terms searches a subgraph once per term count,
; each search passing Herbie --num-taylor-terms and cached under its own key,
; and gives each core the answers of all of them, the fewest terms first and
; each expression once, whatever order the counts are listed in. The stand-in
; Herbie answers B then A at 2 terms, C then B at 4 and C then E at 6 (A, a
; regime split, is dropped in pricing, so the explanation skips its ordinal). A
; count searched before is read from the cache in any list. Unset, nothing is
; passed and the cache key is the one UNSET pins. A Herbie that does not take
; the option stops the compile.

; SEARCH24: [poseidon] Herbie subgraph 0 of preprocess_tester at 2 Taylor terms: 2 FPCore(s) in 1 process(es) of 1 worker(s), per-core timeout {{.*}}, key [[KEY2:[0-9A-F]+]]
; SEARCH24: [poseidon] Saved Herbie output to cache file {{.*}}herbie_default_[[KEY2]].json
; SEARCH24: [poseidon] Herbie subgraph 0 of preprocess_tester at 4 Taylor terms: 2 FPCore(s) in 1 process(es) of 1 worker(s), per-core timeout {{.*}}, key [[KEY4:[0-9A-F]+]]
; SEARCH24: [poseidon] Saved Herbie output to cache file {{.*}}herbie_default_[[KEY4]].json

; CALLS24: {{^}}fake-herbie: report {{.*}} --enable reduce:dag-cost --num-taylor-terms 2 {{[^ ]+}} {{[^ ]+$}}
; CALLS24-NEXT: {{^}}fake-herbie: report {{.*}} --enable reduce:dag-cost --num-taylor-terms 4 {{[^ ]+}} {{[^ ]+$}}
; CALLS24-NOT: report

; UNION24: R0_0 (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))
; UNION24-NEXT: R0_2 (/.f64 (*.f64 v1 v1) (+.f64 v0 (sqrt.f64 (+.f64 (*.f64 v1 v1) (*.f64 v0 v0)))))
; UNION24-NOT: R0_

; SEARCH4-NOT: FPCore(s) in
; SEARCH4: [poseidon] Using cached Herbie output from {{.*}}herbie_default_{{[0-9A-F]+}}.json
; SEARCH4-NOT: FPCore(s) in

; UNION4: R0_0 (/.f64 (*.f64 v1 v1) (+.f64 v0 (sqrt.f64 (+.f64 (*.f64 v1 v1) (*.f64 v0 v0)))))
; UNION4-NEXT: R0_1 (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))
; UNION4-NOT: R0_

; SEARCH46: [poseidon] Using cached Herbie output from
; SEARCH46: [poseidon] Herbie subgraph 0 of preprocess_tester at 6 Taylor terms: 2 FPCore(s) in
; SEARCH46-NOT: at 4 Taylor terms: 2 FPCore(s) in

; CALLS46: {{^}}fake-herbie: report {{.*}} --num-taylor-terms 6 {{[^ ]+}} {{[^ ]+$}}
; CALLS46-NOT: report

; UNION46: R0_0 (/.f64 (*.f64 v1 v1) (+.f64 v0 (sqrt.f64 (+.f64 (*.f64 v1 v1) (*.f64 v0 v0)))))
; UNION46-NEXT: R0_1 (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))
; UNION46-NEXT: R0_2 (-.f64 (sqrt.f64 (fma.f64 v0 v0 (*.f64 v1 v1))) v0)
; UNION46-NOT: R0_

; UNSET-NOT: Taylor terms
; UNSET: [poseidon] Saved Herbie output to cache file {{.*}}herbie_default_7D7D1B4F7D656892.json

; NOOPTION: error: {{.*}}Poseidon: the cache {{.*}}old holds no Herbie answer for subgraph 0 of preprocess_tester at {{[0-9]+}} Taylor terms, and the Herbie at {{.*}}fake-herbie.sh cannot run (-poseidon-herbie-num-taylor-terms passes it --num-taylor-terms, which it does not take ('report --help' does not list it); use a Herbie built with the option, or leave the flag out)

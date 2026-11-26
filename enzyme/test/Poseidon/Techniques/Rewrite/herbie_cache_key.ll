; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles -poseidon-early-prune \
; DEFINE:   -poseidon-enable-rewrite=true -poseidon-enable-precision=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-print -S -o /dev/null
; DEFINE: %{fake} = env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys
; RUN: rm -rf %t && mkdir %t
; RUN: printf '#!/bin/sh\nexec %S/../../Inputs/fake-herbie.sh "$@"\n' > %t/herbie-a.sh && chmod +x %t/herbie-a.sh
; RUN: printf '#!/bin/sh\n# another build\nexec %S/../../Inputs/fake-herbie.sh "$@"\n' > %t/herbie-b.sh && chmod +x %t/herbie-b.sh

; A Herbie answer is filed under its question, the cores, the platform and the
; search arguments, and records the Herbie that gave it, so a cache directory
; carries answers to a machine without Herbie: Herbie A answers (FRESH), with
; no Herbie the answer comes from the cache (CACHED), another Herbie B runs and
; replaces it under the same name (REPLACED), which then records B (ENTRY).
; RUN: %{fake} %{opt} -poseidon-herbie-processes=1 -poseidon-cache=%t/c -poseidon-herbie-binary=%t/herbie-a.sh > %t/a.log 2>&1
; RUN: FileCheck --check-prefix=FRESH %s < %t/a.log
; RUN: %{opt} -poseidon-herbie-processes=1 -poseidon-cache=%t/c -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=CACHED --implicit-check-not="Executing Herbie" %s
; RUN: %{fake} %{opt} -poseidon-herbie-processes=1 -poseidon-cache=%t/c -poseidon-herbie-binary=%t/herbie-b.sh > %t/b.log 2>&1
; RUN: FileCheck --check-prefix=REPLACED %s < %t/b.log
; RUN: cat %t/a.log %t/b.log %t/c/herbie_*.json | FileCheck --check-prefix=ENTRY %s

; A changed search argument searches again into another DP table (ARGS); going
; back finds the first answers and table (BACK); timeouts and worker count are
; not in the key (HIT); another Herbie binary searches again and its same
; answers read the table back (BINARY). Without -poseidon-herbie-threads a
; search runs one process of one worker (FIRST).
; RUN: %{fake} %{opt} -poseidon-cache=%t/k -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh 2>&1 | FileCheck --check-prefix=FIRST %s
; RUN: %{fake} %{opt} -poseidon-cache=%t/k -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-num-iters=2 2>&1 | FileCheck --check-prefix=ARGS %s
; RUN: %{fake} %{opt} -poseidon-cache=%t/k -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh 2>&1 | FileCheck --check-prefix=BACK %s
; RUN: %{fake} %{opt} -poseidon-cache=%t/k -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-timeout=50 -poseidon-herbie-threads=2 2>&1 | FileCheck --check-prefix=HIT %s
; RUN: %{fake} %{opt} -poseidon-cache=%t/k -poseidon-herbie-binary=%t/herbie-a.sh 2>&1 | FileCheck --check-prefix=BINARY %s

; FRESH: [poseidon] Saved Herbie output to cache file {{.*}}/c/herbie_default_[[KEY:[0-9A-F]+]].json{{$}}
; FRESH: [poseidon] Herbie subgraph 0 of preprocess_tester: completed in {{.*}}, a fresh answer of the Herbie with digest {{[0-9A-F]+$}}

; CACHED: [poseidon] Using cached Herbie output from {{.*}}/c/herbie_default_{{[0-9A-F]+}}.json{{$}}
; CACHED: Applying solution for

; REPLACED-NOT: [poseidon] Using cached Herbie output
; REPLACED: [poseidon] Herbie subgraph 0 of preprocess_tester: the cache file {{.*}}/c/herbie_default_{{[0-9A-F]+}}.json holds the answer of another Herbie; running Herbie to replace it{{$}}
; REPLACED: [poseidon] Saved Herbie output to cache file {{.*}}/c/herbie_default_{{[0-9A-F]+}}.json{{$}}

; ENTRY: [poseidon] Saved Herbie output to cache file {{.*}}/c/herbie_default_[[KEY:[0-9A-F]+]].json{{$}}
; ENTRY: a fresh answer of the Herbie with digest [[A:[0-9A-F]+]]{{$}}
; ENTRY: [poseidon] Saved Herbie output to cache file {{.*}}/c/herbie_default_[[KEY]].json{{$}}
; ENTRY: a fresh answer of the Herbie with digest [[B:[0-9A-F]+]]{{$}}
; ENTRY-NOT: "Herbie binary": "[[A]]"
; ENTRY: "Herbie binary": "[[B]]"

; FIRST: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in 1 process(es) of 1 worker(s),
; FIRST: [poseidon] Saved Herbie output to cache file {{.*}}herbie_default_{{[0-9A-F]+}}.json
; FIRST-NOT: DP cache entry
; FIRST: Solving the DP for preprocess_tester.

; ARGS-NOT: [poseidon] Using cached Herbie output
; ARGS: [poseidon] Saved Herbie output to cache file
; ARGS-NOT: DP cache entry
; ARGS: Solving the DP for preprocess_tester.

; BACK-NOT: [poseidon] Saved Herbie output
; BACK: [poseidon] Using cached Herbie output from {{.*}}herbie_default_{{[0-9A-F]+}}.json
; BACK-NOT: DP cache entry
; BACK: Cache entry for preprocess_tester found. Loaded DP tables from cache.

; HIT-NOT: [poseidon] Saved Herbie output
; HIT: [poseidon] Using cached Herbie output from
; HIT-NOT: DP cache entry
; HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.

; BINARY-NOT: [poseidon] Using cached Herbie output
; BINARY: [poseidon] Saved Herbie output to cache file
; BINARY-NOT: DP cache entry
; BINARY: Cache entry for preprocess_tester found. Loaded DP tables from cache.

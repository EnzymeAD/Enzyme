; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles \
; DEFINE:   -poseidon-enable-rewrite=true -poseidon-enable-precision=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -poseidon-cache=%t/c -poseidon-print -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: printf '#!/bin/sh\nexec %S/../../Inputs/fake-herbie.sh "$@"\n' > %t/other-herbie.sh && chmod +x %t/other-herbie.sh
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh 2>&1 \
; RUN:   | FileCheck --check-prefix=FIRST %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-num-iters=2 2>&1 \
; RUN:   | FileCheck --check-prefix=OTHER %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh 2>&1 \
; RUN:   | FileCheck --check-prefix=BACK %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-timeout=50 -poseidon-herbie-threads=2 2>&1 \
; RUN:   | FileCheck --check-prefix=HIT %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%t/other-herbie.sh 2>&1 | FileCheck --check-prefix=BINARY %s

; A Herbie result is cached under its cores, the search arguments and the
; platform, and records the binary that gave it. Changing a search argument
; searches again and solves into another table; going back finds the first
; search's results and the first table. Herbie's timeouts and worker count
; decide no result, so changing them reads both caches. Another Herbie binary
; searches again, and its answers, the same, read the table back: the table
; depends on the answers, not on which Herbie gave them.

; FIRST: [poseidon] Saved Herbie output to cache file {{.*}}herbie_default_{{[0-9A-F]+}}.json
; FIRST-NOT: DP cache entry
; FIRST: Solving the DP for preprocess_tester.

; OTHER-NOT: [poseidon] Using cached Herbie output
; OTHER: [poseidon] Saved Herbie output to cache file
; OTHER-NOT: DP cache entry
; OTHER: Solving the DP for preprocess_tester.

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

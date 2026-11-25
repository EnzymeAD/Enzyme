; DEFINE: %{opt} = %opt < %S/../Inputs/regime_split/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/regime_split/profiles \
; DEFINE:   -poseidon-enable-herbie=true -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -poseidon-cache=%t/c -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: printf '#!/bin/sh\nexec %S/../Inputs/fake-herbie.sh "$@"\n' > %t/other-herbie.sh && chmod +x %t/other-herbie.sh
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh 2>&1 \
; RUN:   | FileCheck --check-prefix=FIRST %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-num-iters=2 2>&1 \
; RUN:   | FileCheck --check-prefix=OTHER %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh 2>&1 \
; RUN:   | FileCheck --check-prefix=BACK %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-timeout=50 -poseidon-herbie-num-threads=2 2>&1 \
; RUN:   | FileCheck --check-prefix=HIT %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%t/other-herbie.sh 2>&1 | FileCheck --check-prefix=BINARY %s

; A Herbie result is cached under its cores, the binary, the search arguments
; and the platform, and a DP table records them. Changing a search argument
; searches again and re-solves; going back finds the first search's results,
; but the table was last solved under the other argument, so it is solved
; again. Herbie's timeouts and worker count decide no result, so changing them
; reads both caches. Another Herbie binary searches again and re-solves.

; FIRST: Saved Herbie output to cache file {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt
; FIRST-NOT: DP cache entry
; FIRST: Solving the DP for preprocess_tester.

; OTHER-NOT: Using cached Herbie output
; OTHER: Saved Herbie output to cache file
; OTHER: [poseidon] DP cache entry for preprocess_tester was solved for a different -poseidon-herbie-num-iters; re-solving.

; BACK-NOT: Saved Herbie output
; BACK: Using cached Herbie output from {{.*}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt
; BACK: [poseidon] DP cache entry for preprocess_tester was solved for a different -poseidon-herbie-num-iters; re-solving.

; HIT-NOT: Saved Herbie output
; HIT: Using cached Herbie output from
; HIT-NOT: DP cache entry
; HIT: Cache entry for preprocess_tester found. Loaded DP tables from cache.

; BINARY-NOT: Using cached Herbie output
; BINARY: Saved Herbie output to cache file
; BINARY: [poseidon] DP cache entry for preprocess_tester was solved for a different Herbie binary; re-solving.

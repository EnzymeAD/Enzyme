; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles \
; DEFINE:   -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-cost-budget=1e9 -S -o /dev/null
; RUN: rm -rf %t && mkdir -p %t/tmp %t/used %t/none %t/sig && touch %t/file
; RUN: not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/no-herbie -poseidon-cache=%t/c -poseidon-print 2>&1 | FileCheck --check-prefix=ENABLED %s
; RUN: not env TMPDIR=%t/tmp %clang -O1 -c -x ir %S/../../Inputs/piecewise_rewrite/tester.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles -mllvm -poseidon-enable-rewrite=true -mllvm -poseidon-herbie-binary=%t/no-herbie -mllvm -poseidon-enable-precision=false -mllvm -poseidon-cost-model=%cost_models/cm_cpu_native.csv -mllvm -poseidon-cost-budget=1e9 -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print -o %t/site.o 2>&1 | FileCheck --check-prefix=ENABLED --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE %s
; RUN: not ls %t/site.o
; RUN: ls %t/tmp | count 0
; RUN: %{opt} -poseidon-enable-rewrite=false -poseidon-herbie-binary=%t/no-herbie -poseidon-cache=%t/c 2>&1 | count 0

; A subgraph whose Herbie answer is not in the cache needs Herbie: when there is
; no Herbie the compile stops, naming the search, the cache file the answer
; would be in and the binary, under opt and under clang alike: an error at the
; site, exit status 1, no object and none of clang's crash reporting. With
; algebraic rewriting off nothing asks Herbie, and the site is solved without
; it.

; ENABLED-NOT: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in
; ENABLED: error: {{.*}}preprocess_tester{{.*}}: Poseidon: no Herbie answer for subgraph 0 of preprocess_tester is cached in {{.*}}/c/herbie_default_{{[0-9A-F]+}}.json, and there is no Herbie at {{.*}}/no-herbie to compute it; copy here the -poseidon-cache directory of a compile that ran Herbie, or turn algebraic rewriting off with -poseidon-enable-rewrite=0{{$}}


; A Herbie that exits with a nonzero status is not a crash. When it wrote a
; usable results.json, the compile says so once and uses and caches the
; results; when it wrote none, the compile stops.
; RUN: printf '#!/bin/sh\n%S/../../Inputs/fake-herbie.sh "$@"\nexit 1\n' > %t/herbie.sh && chmod +x %t/herbie.sh
; RUN: printf '#!/bin/sh\nexit 3\n' > %t/no-results.sh && chmod +x %t/no-results.sh
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/used -poseidon-print 2>&1 | FileCheck --check-prefix=USED --implicit-check-not=crash %s
; RUN: not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/no-results.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/none 2>&1 | FileCheck --check-prefix=NONE --implicit-check-not=crashed %s

; USED: [poseidon] Saved Herbie output to cache file
; USED: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: Herbie subgraph 0 of preprocess_tester exited with status 1 but wrote a usable results.json, which is used{{$}}
; USED: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)

; NONE: error: {{.*}}Poseidon: the cache {{.*}}none holds no Herbie answer for subgraph 0 of preprocess_tester, and the Herbie at {{.*}}no-results.sh cannot run (it exited with status 3 and wrote no usable results.json (platform {{.*}}); its own diagnostics are on stderr above)

; A Herbie process that dies from a signal (here SIGKILL, as the OOM killer
; sends) well inside its wall budget crashed; it did not time out. The compile
; says so, leaves the subgraph without algebraic candidates and keeps the body
; as written; under -poseidon-strict-mode the crash is fatal. Neither compile
; caches anything, so the next compile runs Herbie again.
; RUN: printf '#!/bin/sh\n[ -z "$FAKE_HERBIE_SIGNAL" ] || kill -KILL $$\nexec %S/../../Inputs/fake-herbie.sh "$@"\n' > %t/signal.sh && chmod +x %t/signal.sh
; RUN: env FAKE_HERBIE_SIGNAL=1 FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/signal.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/sig -poseidon-print 2>&1 | FileCheck --check-prefix=SIGNAL --implicit-check-not="was stopped after" --implicit-check-not="[poseidon] Saved Herbie output" %s
; RUN: env FAKE_HERBIE_SIGNAL=1 FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/signal.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/sig -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=STRICT %s
; RUN: ls %t/sig | FileCheck --allow-empty --check-prefix=FILES %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/signal.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/sig -poseidon-print 2>&1 | FileCheck --check-prefix=RETRY %s

; SIGNAL: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in 1 process(es) of 2 worker(s), per-core timeout 1800 s, wall budget 2040 s, key {{[0-9A-F]+$}}
; SIGNAL-NEXT: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: Herbie subgraph 0 of preprocess_tester crashed (signal: {{.+}}) after {{[0-9.e+-]+}} s, so the subgraph has no algebraic candidates in this compile; nothing is cached, so the next compile runs Herbie again{{$}}
; SIGNAL: DP table contains 1 entries.
; SIGNAL-NOT: Applying solution
; SIGNAL: [poseidon] no rewrite applied for preprocess_tester; calling original body tester (bit-transparent no-op)

; STRICT: error: {{.*}}Poseidon: strict mode: Herbie crashed (signal: {{.+}}) on subgraph 0 of preprocess_tester

; FILES-NOT: cachedHerbie

; RETRY-NOT: timed out before
; RETRY: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in 1 process(es) of 2 worker(s), per-core timeout 1800 s, wall budget 2040 s, key [[KEY:[0-9A-F]+]]{{$}}
; RETRY: [poseidon] Saved Herbie output to cache file {{.*}}herbie_default_[[KEY]].json{{$}}
; RETRY-NEXT: [poseidon] Herbie subgraph 0 of preprocess_tester: completed in {{[0-9.e+-]+}} s, a fresh answer of the Herbie with digest {{[0-9A-F]+}}{{$}}
; RETRY: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; RETRY: [poseidon] Herbie answers: 1 fresh, 0 from the cache{{$}}

; A Herbie answer that cannot be cached, here because the cache directory is
; under a regular file, stops the compile naming the path.
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/file/c 2>&1 | FileCheck --check-prefix=UNWRITABLE %s

; UNWRITABLE: error: {{.*}}preprocess_tester{{.*}}: Poseidon: cannot create the Herbie cache directory {{.*}}file/c: Not a directory. Herbie's answers are kept in the -poseidon-cache directory, which has to be writable.

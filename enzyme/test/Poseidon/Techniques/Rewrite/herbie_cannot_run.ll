; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles -poseidon-early-prune \
; DEFINE:   -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-cost-budget=1e9 -S -o /dev/null
; RUN: rm -rf %t && mkdir -p %t/tmp %t/used %t/none %t/sig && touch %t/file && split-file %s %t/timeout
; RUN: not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/no-herbie -poseidon-cache=%t/c -poseidon-print 2>&1 | FileCheck --check-prefix=ENABLED %s
; RUN: not env TMPDIR=%t/tmp %clang -O1 -c -x ir %S/../../Inputs/piecewise_rewrite/tester.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles -mllvm -poseidon-early-prune -mllvm -poseidon-enable-rewrite=true -mllvm -poseidon-herbie-binary=%t/no-herbie -mllvm -poseidon-enable-precision=false -mllvm -poseidon-cost-model=%cost_models/cm_cpu_native.csv -mllvm -poseidon-cost-budget=1e9 -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print -o %t/site.o 2>&1 | FileCheck --check-prefix=ENABLED --implicit-check-not="error in backend" --implicit-check-not="Stack dump" --implicit-check-not=PLEASE %s
; RUN: not ls %t/site.o
; RUN: ls %t/tmp | count 0
; RUN: %{opt} -poseidon-enable-rewrite=false -poseidon-herbie-binary=%t/no-herbie -poseidon-cache=%t/c 2>&1 | count 0

; An uncached answer with no Herbie stops the compile naming the search, the
; cache file and the binary, under opt and clang alike (exit 1, no object, no
; crash report); with rewriting off nothing asks Herbie.

; ENABLED-NOT: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in
; ENABLED: error: {{.*}}preprocess_tester{{.*}}: Poseidon: no Herbie answer for subgraph 0 of preprocess_tester is cached in {{.*}}/c/herbie_default_{{[0-9A-F]+}}.json, and there is no Herbie at {{.*}}/no-herbie to compute it; copy here the -poseidon-cache directory of a compile that ran Herbie, or turn algebraic rewriting off with -poseidon-enable-rewrite=0{{$}}

; A Herbie exiting nonzero after writing usable results is used, with one
; warning (USED); with no results the compile stops (NONE).
; RUN: printf '#!/bin/sh\n%S/../../Inputs/fake-herbie.sh "$@"\nexit 1\n' > %t/herbie.sh && chmod +x %t/herbie.sh
; RUN: printf '#!/bin/sh\nexit 3\n' > %t/no-results.sh && chmod +x %t/no-results.sh
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/used -poseidon-print 2>&1 | FileCheck --check-prefix=USED --implicit-check-not=crash %s
; RUN: not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/no-results.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/none 2>&1 | FileCheck --check-prefix=NONE --implicit-check-not=crashed %s

; USED: [poseidon] Saved Herbie output to cache file
; USED: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: Herbie subgraph 0 of preprocess_tester exited with status 1 but wrote a usable results.json, which is used{{$}}
; USED: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)

; NONE: error: {{.*}}Poseidon: the cache {{.*}}none holds no Herbie answer for subgraph 0 of preprocess_tester, and the Herbie at {{.*}}no-results.sh cannot run (it exited with status 3 and wrote no usable results.json (platform {{.*}}); its own diagnostics are on stderr above)

; A Herbie killed by a signal inside its wall budget crashed, not timed out:
; no candidates, body kept, nothing cached (SIGNAL, FILES, RETRY); fatal under
; strict mode (STRICT).
; RUN: printf '#!/bin/sh\n[ -z "$FAKE_HERBIE_SIGNAL" ] || kill -KILL $$\nexec %S/../../Inputs/fake-herbie.sh "$@"\n' > %t/signal.sh && chmod +x %t/signal.sh
; RUN: env FAKE_HERBIE_SIGNAL=1 FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/signal.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/sig -poseidon-print 2>&1 | FileCheck --check-prefix=SIGNAL --implicit-check-not="was stopped after" --implicit-check-not="[poseidon] Saved Herbie output" %s
; RUN: env FAKE_HERBIE_SIGNAL=1 FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/signal.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/sig -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=STRICT %s
; RUN: ls %t/sig | FileCheck --allow-empty --check-prefix=FILES %s
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%t/signal.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/sig -poseidon-print 2>&1 | FileCheck --check-prefix=RETRY %s

; SIGNAL: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in 1 process(es) of 1 worker(s), per-core timeout 1800 s, wall budget 3840 s, key {{[0-9A-F]+$}}
; SIGNAL-NEXT: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: Herbie subgraph 0 of preprocess_tester crashed (signal: {{.+}}) after {{[0-9.e+-]+}} s, so the subgraph has no algebraic candidates in this compile; nothing is cached, so the next compile runs Herbie again{{$}}
; SIGNAL: DP table contains 1 entries.
; SIGNAL-NOT: Applying solution
; SIGNAL: [poseidon] no rewrite applied for preprocess_tester; calling original body tester (bit-transparent no-op)

; STRICT: error: {{.*}}Poseidon: strict mode: Herbie crashed (signal: {{.+}}) on subgraph 0 of preprocess_tester

; FILES-NOT: cachedHerbie

; RETRY: [poseidon] Herbie subgraph 0 of preprocess_tester: 2 FPCore(s) in 1 process(es) of 1 worker(s), per-core timeout 1800 s, wall budget 3840 s, key [[KEY:[0-9A-F]+]]{{$}}
; RETRY: [poseidon] Saved Herbie output to cache file {{.*}}herbie_default_[[KEY]].json{{$}}
; RETRY-NEXT: [poseidon] Herbie subgraph 0 of preprocess_tester: completed in {{[0-9.e+-]+}} s, a fresh answer of the Herbie with digest {{[0-9A-F]+}}{{$}}
; RETRY: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; RETRY: [poseidon] Herbie answers: 1 fresh, 0 from the cache{{$}}

; An answer with a core stopped by the per-core --timeout is used but not
; cached; a complete answer is cached and serves the next compile.
; RUN: env FAKE_HERBIE_DIR=%t/timeout %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/to -poseidon-print 2>&1 | FileCheck --check-prefix=CORE-TIMEOUT --implicit-check-not="[poseidon] Saved Herbie output" %s
; RUN: not ls %t/to/herbie_default_*.json
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/to -poseidon-print 2>&1 | FileCheck --check-prefix=COMPLETE --implicit-check-not="timed out" %s
; RUN: env FAKE_HERBIE_DIR=%t/timeout %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/to -poseidon-print 2>&1 | FileCheck --check-prefix=REUSED --implicit-check-not="timed out" --implicit-check-not="[poseidon] Executing Herbie" %s

; CORE-TIMEOUT: [poseidon] Executing Herbie
; CORE-TIMEOUT: warning: {{.*}}preprocess_tester{{.*}}: Poseidon: Herbie subgraph 0 of preprocess_tester timed out on 1 of 2 FPCore(s) at its per-core --timeout of 1800 s, so they have no algebraic candidates in this compile; the answer is not cached, so the next compile runs Herbie again{{$}}
; CORE-TIMEOUT: [poseidon] Herbie core "1" (subgraph 0 of preprocess_tester): status=timeout
; CORE-TIMEOUT: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; CORE-TIMEOUT: [poseidon] Herbie answers: 1 fresh, 0 from the cache{{$}}

; COMPLETE: [poseidon] Executing Herbie
; COMPLETE: [poseidon] Saved Herbie output to cache file {{.*}}herbie_default_{{[0-9A-F]+}}.json{{$}}
; COMPLETE: [poseidon] Herbie answers: 1 fresh, 0 from the cache{{$}}

; REUSED: [poseidon] Using cached Herbie output from {{.*}}herbie_default_{{[0-9A-F]+}}.json{{$}}
; REUSED: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)){{$}}
; REUSED: [poseidon] Herbie answers: 0 fresh, 1 from the cache{{$}}

; An answer that cannot be cached (cache under a regular file) stops the compile.
; RUN: env FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/file/c 2>&1 | FileCheck --check-prefix=UNWRITABLE %s

; UNWRITABLE: error: {{.*}}preprocess_tester{{.*}}: Poseidon: cannot create the Herbie cache directory {{.*}}file/c: Not a directory. Herbie's answers are kept in the -poseidon-cache directory, which has to be writable.

; Herbie's files live in TMPDIR and are removed, so a read-only working
; directory is fine (READONLY); an unusable TMPDIR stops the compile (NOTMP).
; RUN: mkdir %t/ro %t/rotmp && chmod 555 %t/ro
; RUN: cd %t/ro && env TMPDIR=%t/rotmp FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/rocache -poseidon-print 2>&1 | FileCheck --check-prefix=READONLY %s
; RUN: chmod 755 %t/ro && ls -A %t/ro | count 0 && ls -A %t/rotmp | count 0
; RUN: env TMPDIR=%t/file FAKE_HERBIE_DIR=%S/../../Inputs/dp_cache_keys not %{opt} -poseidon-enable-rewrite=true -poseidon-herbie-binary=%S/../../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-cache=%t/tmpfile 2>&1 | FileCheck --check-prefix=NOTMP %s

; READONLY: [poseidon] Saved Herbie output to cache file
; READONLY: Applying solution for (- (sqrt (+ (* v0 v0) (* v1 v1))) v0) --(1)-> (/.f64 (*.f64 v1 v1)

; NOTMP: error: {{.*}}preprocess_tester{{.*}}: Poseidon: cannot create a Herbie input file in the temporary directory {{.*}}file: Not a directory{{$}}

;--- results.json
{"tests":[{"bits":64,"name":"0","status":"imp-start","prec":"binary64","output":"(if.f64 (<=.f64 v1 #s(literal -4999/5000 binary64)) #s(literal 0 binary64) (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)))","cost-accuracy":[[1.0,30.0],[1.5,0.5,"(if.f64 (<=.f64 v1 #s(literal -4999/5000 binary64)) #s(literal 0 binary64) (/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0)))"],[[1.4,0.6,"(/.f64 (*.f64 v1 v1) (+.f64 (sqrt.f64 (+.f64 (*.f64 v0 v0) (*.f64 v1 v1))) v0))"]]]},{"bits":32,"name":"1","status":"timeout","prec":"binary32","output":"#f","time":1800000}]}

; DEFINE: %{opt} = %opt < %S/../Inputs/regime_split/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/regime_split/profiles \
; DEFINE:   -poseidon-enable-herbie=true -poseidon-herbie-processes=1 -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 \
; DEFINE:   -poseidon-cache=%t/c -poseidon-print -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: printf '#!/bin/sh\nexec %S/../Inputs/fake-herbie.sh "$@"\n' > %t/herbie-a.sh && chmod +x %t/herbie-a.sh
; RUN: printf '#!/bin/sh\n# another build\nexec %S/../Inputs/fake-herbie.sh "$@"\n' > %t/herbie-b.sh && chmod +x %t/herbie-b.sh
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%t/herbie-a.sh > %t/a.log 2>&1
; RUN: FileCheck --check-prefix=FRESH %s < %t/a.log
; RUN: %{opt} -poseidon-herbie-binary=%t/no-herbie 2>&1 | FileCheck --check-prefix=CACHED --implicit-check-not="Executing Herbie" %s
; RUN: env FAKE_HERBIE_DIR=%S/../Inputs/dp_cache_keys %{opt} -poseidon-herbie-binary=%t/herbie-b.sh > %t/b.log 2>&1
; RUN: FileCheck --check-prefix=OTHER %s < %t/b.log
; RUN: cat %t/a.log %t/b.log %t/c/cachedHerbieOutput_* | FileCheck --check-prefix=ENTRY %s

; A Herbie answer is filed under its question, the cores, the platform and the
; search arguments, and records the Herbie that gave it, so a cache directory
; carries answers to a machine without Herbie. Herbie A answers (FRESH); with
; no Herbie at the configured path the compile takes A's answer from the cache
; (CACHED); another Herbie B does not take it but runs and replaces it under
; the same file name (OTHER), which then records B (ENTRY). The error a
; question with no cached answer and no Herbie gives is checked in
; herbie_cannot_run.ll.

; FRESH: [poseidon] Saved Herbie output to cache file {{.*}}/c/cachedHerbieOutput_default_[[KEY:[0-9A-F]+]].txt{{$}}
; FRESH: [poseidon] Herbie subgraph 0 of preprocess_tester: completed in {{.*}}, a fresh answer of the Herbie with digest {{[0-9A-F]+$}}

; CACHED: [poseidon] Using cached Herbie output from {{.*}}/c/cachedHerbieOutput_default_{{[0-9A-F]+}}.txt{{$}}
; CACHED: Applying solution for

; OTHER-NOT: [poseidon] Using cached Herbie output
; OTHER: [poseidon] Herbie subgraph 0 of preprocess_tester: the cache file {{.*}}/c/cachedHerbieOutput_default_{{[0-9A-F]+}}.txt holds the answer of another Herbie; running Herbie to replace it{{$}}
; OTHER: [poseidon] Saved Herbie output to cache file {{.*}}/c/cachedHerbieOutput_default_{{[0-9A-F]+}}.txt{{$}}

; ENTRY: [poseidon] Saved Herbie output to cache file {{.*}}/c/cachedHerbieOutput_default_[[KEY:[0-9A-F]+]].txt{{$}}
; ENTRY: a fresh answer of the Herbie with digest [[A:[0-9A-F]+]]{{$}}
; ENTRY: [poseidon] Saved Herbie output to cache file {{.*}}/c/cachedHerbieOutput_default_[[KEY]].txt{{$}}
; ENTRY: a fresh answer of the Herbie with digest [[B:[0-9A-F]+]]{{$}}
; ENTRY-NOT: "Herbie binary": "[[A]]"
; ENTRY: "Herbie binary": "[[B]]"

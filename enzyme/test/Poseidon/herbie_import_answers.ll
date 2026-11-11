; RUN: rm -rf %t && mkdir -p %t && touch %t/other.log
; RUN: printf '#!/bin/sh\nexec %S/Inputs/herbie_taylor_terms/fake-herbie.sh "$@"\n' > %t/other-herbie.sh && chmod +x %t/other-herbie.sh
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/herbie_taylor_terms FAKE_HERBIE_LOG=%t/shipped.log %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/herbie_taylor_terms/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-herbie-num-taylor-terms=2,4 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/shipped -poseidon-report-path=%t/r-shipped -S -o /dev/null > %t/fresh.log 2>&1
; RUN: ls %t/shipped | FileCheck --check-prefix=SHIPPED %s
; RUN: %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/no-herbie -poseidon-herbie-num-taylor-terms=2,4 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/mine -poseidon-import-herbie-answers=%t/shipped -poseidon-report-path=%t/r-mine -S -o /dev/null > %t/replay.log 2>&1
; RUN: cat %t/fresh.log %t/replay.log | FileCheck --check-prefix=REPLAY %s
; RUN: diff %t/r-shipped/preprocess_tester.txt %t/r-mine/preprocess_tester.txt
; RUN: ls %t/mine | FileCheck --check-prefix=MINE %s
; RUN: not %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/no-herbie -poseidon-herbie-num-taylor-terms=2,4 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/mine -S -o /dev/null 2>&1 | FileCheck --check-prefix=NOFLAG %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/herbie_taylor_terms FAKE_HERBIE_LOG=%t/other.log %opt < %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/regime_split/profiles -poseidon-enable-herbie=true -poseidon-herbie-binary=%t/other-herbie.sh -poseidon-herbie-processes=1 -poseidon-herbie-num-taylor-terms=2,4,6 -poseidon-enable-pt=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cost-budget=1e9 -poseidon-cache=%t/other -poseidon-import-herbie-answers=%t/shipped -S -o /dev/null 2>&1 | FileCheck --check-prefix=MIXED %s
; RUN: FileCheck --check-prefix=OTHERCALLS %s < %t/other.log

; -poseidon-import-herbie-answers replays the answers another Herbie binary
; gave: a fresh answer is cached with a record of the question it answers (the
; cores, the platform and the search options, the term count among them, but
; not the binary), and a compile with the flag takes the answer recorded for
; its own question when its cache has none from its own binary, logging the
; binary that gave it, here with no Herbie installed at all. The replay is not
; cached as this binary's answer, so without the flag the same compile has no
; answer and stops. With a Herbie that runs, a question the directory does not
; record (6 terms) is searched fresh and the others are replayed.

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

; SHIPPED: cachedHerbieOutput_default_{{[0-9A-F]+}}.txt
; SHIPPED: cachedHerbieOutput_default_{{[0-9A-F]+}}.txt
; SHIPPED: cachedHerbieQuestion_default_{{[0-9A-F]+}}.json
; SHIPPED: cachedHerbieQuestion_default_{{[0-9A-F]+}}.json

; REPLAY: [poseidon] Herbie subgraph 0 of preprocess_tester at 2 Taylor terms: completed in {{.*}} s, a fresh answer of the Herbie with digest [[DIGEST:[0-9A-F]+]]
; REPLAY: [poseidon] Herbie answers: 2 fresh, 0 from the cache, 0 replayed{{$}}
; REPLAY-NOT: FPCore(s) in
; REPLAY: [poseidon] Herbie subgraph 0 of preprocess_tester at 2 Taylor terms: replayed the answer of the Herbie with digest [[DIGEST]] from {{.*}}shipped{{/|\\}}cachedHerbieOutput_default_{{[0-9A-F]+}}.txt (-poseidon-import-herbie-answers)
; REPLAY: [poseidon] Herbie subgraph 0 of preprocess_tester at 4 Taylor terms: replayed the answer of the Herbie with digest [[DIGEST]] from
; REPLAY-NOT: FPCore(s) in
; REPLAY: [poseidon] Herbie answers: 0 fresh, 0 from the cache, 2 replayed from {{.*}}shipped{{$}}

; MINE-NOT: cachedHerbie

; NOFLAG: LLVM ERROR: Poseidon: the cache {{.*}}mine holds no Herbie answer for subgraph 0 of preprocess_tester at 2 Taylor terms, and the Herbie at {{.*}}no-herbie cannot run (

; MIXED: at 2 Taylor terms: replayed the answer of the Herbie with digest
; MIXED: at 4 Taylor terms: replayed the answer of the Herbie with digest
; MIXED: [poseidon] Herbie subgraph 0 of preprocess_tester at 6 Taylor terms: 2 FPCore(s) in
; MIXED: [poseidon] Herbie answers: 1 fresh, 0 from the cache, 2 replayed from

; OTHERCALLS: report {{.*}} --num-taylor-terms 6 {{[^ ]+}} {{[^ ]+$}}
; OTHERCALLS-NOT: report

; UNSUPPORTED: herbie
; DEFINE: %{opt} = %opt < %S/../../Inputs/piecewise_rewrite/tester.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/piecewise_rewrite/profiles \
; DEFINE:   -poseidon-enable-precision=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-cost-budget=1e9 -disable-output
; RUN: rm -rf %t
; RUN: %{opt} -poseidon-cache=%t/c -pass-remarks-missed=poseidon 2>&1 | FileCheck --check-prefix=DEFAULT --implicit-check-not=warning: --implicit-check-not=error: %s
; RUN: not %{opt} -poseidon-enable-rewrite -poseidon-cache=%t/c 2>&1 | FileCheck --check-prefix=ASKED %s

; A build configured without Herbie: left to its default, algebraic rewriting
; is off, and a remark says so; asked for, it is an error at the first search
; whose answer is not cached. Both name the configure options that give the
; build a Herbie.

; DEFAULT: remark: {{.*}}Poseidon proposes no algebraic rewrites: this Poseidon was built without Herbie (configure it with -DPOSEIDON_HERBIE_BINARY=<herbie> or -DPOSEIDON_BUILD_HERBIE=ON){{$}}

; ASKED: error: {{.*}}Poseidon: no Herbie answer for subgraph 0 of preprocess_tester is cached in {{.*}}/c/herbie_default_{{[0-9A-F]+}}.json, and this Poseidon was built without Herbie (configure it with -DPOSEIDON_HERBIE_BINARY=<herbie> or -DPOSEIDON_BUILD_HERBIE=ON). Compute the answer on a machine with Herbie and copy its -poseidon-cache directory here, or turn algebraic rewriting off with -poseidon-enable-rewrite=0.{{$}}

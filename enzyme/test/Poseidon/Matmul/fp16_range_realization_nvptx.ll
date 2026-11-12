; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/fp16_range/matmul_tiny -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-6 -poseidon-print -poseidon-cache=%t/tau -S -o /dev/null 2>&1 | FileCheck --check-prefixes=TAU,%if tcec-cumpsgemm %{TAU-CUMPSGEMM%} %else %{TAU-BUILTIN%} %s
; RUN: grep -h '"TCEC backend"' %t/tau/matmulacc_*.json | FileCheck --check-prefixes=KEY,%if tcec-cumpsgemm %{KEY-CUMPSGEMM%} %else %{KEY-BUILTIN%} %s
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/fp16_range/matmul_tiny -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cost-budget=0 -poseidon-cache=%t/budget -S -o /dev/null
; RUN: grep -h '"TCEC backend"' %t/budget/table.json | FileCheck --check-prefixes=KEY,%if tcec-cumpsgemm %{KEY-CUMPSGEMM%} %else %{KEY-BUILTIN%} %s
; RUN: sed -i 's/"TCEC backend": "[^"]*"/"TCEC backend": "another"/' %t/budget/table.json %t/budget/matmulacc_*.json
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/fp16_range/matmul_tiny -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cost-budget=0 -poseidon-cache=%t/budget -S -o /dev/null 2>&1 | FileCheck --check-prefix=ANOTHER %s

; Which realizations of a matrix product the FP16 range rule refuses. The
; product of ozaki_dispatch_nvptx.ll profiled with A at 1e-9 (Inputs/
; fp16_range/matmul_tiny), below FP16's smallest subnormal 5.96e-8. The
; in-kernel direct raise converts each FP64 operand to FP16 as it is, the
; in-kernel splits narrow it to FP32 and split it without scaling, and the
; direct library call (__poseidon_direct_dgemm) rounds it once without scaling:
; all of them are refused. The split library call (__poseidon_tcec_dgemm)
; first scales each operand by the power of two that brings its largest
; magnitude to [2^10, 2^11], so it is simulated on the backend the runtime is
; built with instead, and at tolerance 1e-6 it is the cheapest candidate that
; clears it. Its result is filed under that backend, in its accuracy memo entry
; and in the DP table's key; a table or entry of another backend is solved and
; computed again.

; TAU-BUILTIN: DOMERR tcec library backend=built-in p95_relErr=1.077071e-07
; TAU-CUMPSGEMM: DOMERR tcec library backend=cuMpSGEMM p95_relErr=1.106362e-07
; TAU: #0 wmma m16n16k16 f16/f32  {{.*}}  domainError=INF
; TAU-NEXT: #1 tcec n=2 wmma m16n16k16 f16/f32  {{.*}}  domainError=INF
; TAU-NEXT: #2 tcec n=3 wmma m16n16k16 f16/f32  {{.*}}  domainError=INF
; TAU: #10 tcec-dispatch wmma m16n16k16 f16/f32  compCost/MAC=1.400000e+01  rel=2.734375e-02  domainError=[[ERR:[0-9.e+-]+]]
; TAU-NEXT: #11 direct-dispatch cublas f16/f32  {{.*}}  domainError=INF
; TAU: Poseidon error-budget: matmul -> tcec-dispatch wmma m16n16k16 f16/f32 (domain err [[ERR]] <= 1.000000e-06
; TAU: Applying solution for matmul[0] -> tcec-dispatch wmma m16n16k16 f16/f32 (#10)

; KEY-BUILTIN: "TCEC backend": "built-in",
; KEY-CUMPSGEMM: "TCEC backend": "cuMpSGEMM",
; KEY-NOT: TCEC backend

; ANOTHER-DAG: [poseidon] accuracy memo {{.*}}matmulacc_{{[0-9A-F]+}}.json was computed for a different TCEC backend; recomputing it.
; ANOTHER-DAG: [poseidon] DP cache entry for preprocess_matmul_body was solved for a different TCEC backend; re-solving.

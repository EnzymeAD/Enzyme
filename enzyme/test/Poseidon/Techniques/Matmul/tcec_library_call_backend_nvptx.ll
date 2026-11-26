; DEFINE: %{opt} = %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/fp16_range/matmul_tiny -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_tcec.csv -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: %if tcec-cumpsgemm %{ %{opt} -poseidon-tau=1e-6 -poseidon-print -poseidon-cache=%t/tau 2>&1 | FileCheck --check-prefix=TAU --implicit-check-not="TCEC library call not proposed" %s %}
; RUN: %if tcec-cumpsgemm %{ cat %t/tau/matmul_accuracy_*.json | FileCheck --check-prefix=KEY %s %}
; RUN: %if tcec-cumpsgemm %{ %{opt} -poseidon-cost-budget=0 -poseidon-cache=%t/budget %}
; RUN: %if tcec-cumpsgemm %{ cat %t/budget/dp_*.json | FileCheck --check-prefix=KEY %s %}
; RUN: %if tcec-cumpsgemm %{ sed -i 's/"TCEC backend": "[^"]*"/"TCEC backend": "another"/' %t/budget/dp_*.json %t/budget/matmul_accuracy_*.json %}
; RUN: %if tcec-cumpsgemm %{ %{opt} -poseidon-cost-budget=0 -poseidon-cache=%t/budget -poseidon-print 2>&1 | FileCheck --check-prefix=ANOTHER %s %}
; RUN: %if !tcec-cumpsgemm %{ %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_tcec.csv -poseidon-cost-budget=0 -poseidon-print -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=PRICED --implicit-check-not=tcec-dispatch %s %}
; RUN: %if !tcec-cumpsgemm %{ cat %t/c/dp_*.json | FileCheck --check-prefix=NONE %s %}
; RUN: %if !tcec-cumpsgemm %{ not %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_tcec.csv -poseidon-cache=%t/pin "-poseidon-pin=M0=tcec-dispatch wmma m16n16k16 f16/f32" -S -o /dev/null 2>&1 | FileCheck --check-prefix=PIN %s %}

; With cuMpSGEMM, the split library call of a product whose operand A is
; profiled at 1e-9 (Inputs/fp16_range/matmul_tiny) scales its operands, clears
; 1e-6 where no unscaled FP16 realization is offered, and is keyed by its TCEC
; backend in the memo and the DP table. Without it no such call is offered,
; though priced; the DP key, the print and a pin naming it say so.

; TAU: [poseidon] domain error tcec library p95_relErr=1.106362e-07
; TAU: #7 tcec-dispatch wmma m16n16k16 f16/f32  compCost/MAC=1.400000e+01  rel=2.734375e-02  domainError=[[ERR:[0-9.e+-]+]]
; TAU: not offered: direct-dispatch cublas f16/f32: f16 flushes the smallest profiled operand magnitude, 1e-09, to zero
; TAU: tau=1.000000e-06 {{.*}} rel=[[ERR]]{{$}}
; TAU: Applying solution for matmul[0] -> tcec-dispatch wmma m16n16k16 f16/f32 (#7)

; KEY: "TCEC backend": "cuMpSGEMM",
; KEY-NOT: TCEC backend

; ANOTHER: [poseidon] accuracy memo {{.*}}matmul_accuracy_{{[0-9A-F]+}}.json was computed for a different TCEC backend; recomputing it
; ANOTHER: [poseidon] DP cache entry for preprocess_matmul_body was solved for a different TCEC backend; re-solving.

; PRICED: [poseidon] matmul[0] {{[0-9]+x[0-9]+x[0-9]+}}: TCEC library call not proposed: this build has no TCEC runtime; configure it with -DPOSEIDON_CUMPSGEMM_ROOT=<cuMpSGEMM checkout>{{$}}
; PRICED-NOT: TCEC library call not proposed
; PRICED: #9 ozaki-ii nm=14 wmma m16n16k16 s8/s32
; PRICED-NEXT: #10 direct-dispatch cublas f16/f32
; PRICED: [poseidon]   not offered: tcec-dispatch wmma m16n16k16 f16/f32: this compiler was built without cuMpSGEMM, on which the error-corrected library call runs; configure it with -DPOSEIDON_CUMPSGEMM_ROOT=<cuMpSGEMM checkout>{{$}}
; NONE: "TCEC backend": "none",
; PIN: error: {{.*}}Poseidon: -poseidon-pin: matmul_body: M0 has no candidate named 'tcec-dispatch wmma m16n16k16 f16/f32': pricing dropped it, because this compiler was built without cuMpSGEMM, on which the error-corrected library call runs; configure it with -DPOSEIDON_CUMPSGEMM_ROOT=<cuMpSGEMM checkout>

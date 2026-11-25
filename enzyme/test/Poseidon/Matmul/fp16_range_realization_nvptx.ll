; REQUIRES: tcec-cumpsgemm
; DEFINE: %{opt} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/fp16_range/matmul_tiny \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_tcec.csv -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -poseidon-tau=1e-6 -poseidon-print -poseidon-cache=%t/tau 2>&1 | FileCheck --check-prefix=TAU --implicit-check-not="TCEC dispatch not proposed" %s
; RUN: cat %t/tau/matmulacc_*.json | FileCheck --check-prefix=KEY %s
; RUN: %{opt} -poseidon-cost-budget=0 -poseidon-cache=%t/budget
; RUN: cat %t/budget/dp_*.json | FileCheck --check-prefix=KEY %s
; RUN: sed -i 's/"TCEC backend": "[^"]*"/"TCEC backend": "another"/' %t/budget/dp_*.json %t/budget/matmulacc_*.json
; RUN: %{opt} -poseidon-cost-budget=0 -poseidon-cache=%t/budget -poseidon-print 2>&1 | FileCheck --check-prefix=ANOTHER %s

; Operand A of the product of ozaki_dispatch_nvptx.ll is profiled at 1e-9
; (Inputs/fp16_range/matmul_tiny), below FP16's smallest subnormal: every
; realization that converts it to FP16 unscaled has domainError=INF (the
; exponent-range check in Techniques/Matmul/MatmulAccuracy.cpp). The split
; library call scales its operands first, clears tolerance 1e-6, and is keyed
; by its TCEC backend in the accuracy memo and the DP table.

; TAU: [poseidon] domain error tcec library p95_relErr=1.106362e-07
; TAU: #10 tcec-dispatch wmma m16n16k16 f16/f32  compCost/MAC=1.400000e+01  rel=2.734375e-02  domainError=[[ERR:[0-9.e+-]+]]
; TAU-NEXT: #11 direct-dispatch cublas f16/f32  {{.*}}  domainError=INF
; TAU: [poseidon] error budget: matmul -> tcec-dispatch wmma m16n16k16 f16/f32 (domain err [[ERR]] <= 1.000000e-06
; TAU: Applying solution for matmul[0] -> tcec-dispatch wmma m16n16k16 f16/f32 (#10)

; KEY: "TCEC backend": "cuMpSGEMM",
; KEY-NOT: TCEC backend

; ANOTHER: [poseidon] accuracy memo {{.*}}matmulacc_{{[0-9A-F]+}}.json was computed for a different TCEC backend; recomputing it
; ANOTHER: [poseidon] DP cache entry for preprocess_matmul_body was solved for a different TCEC backend; re-solving.

; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t/prof %t/seeded
; RUN: sed '1i CanonicalHash = fea337c617ef94c9' %S/../Inputs/raise_tcec/preprocess_matmul_body.fpprofile > %t/prof/preprocess_matmul_body.fpprofile
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/first -S -o /dev/null 2>&1 | FileCheck --check-prefixes=FIRST,PRICED %s
; RUN: ls %t/first | FileCheck --check-prefix=MEMO %s
; RUN: for f in %t/first/matmulacc_*; do echo '0 0' > %t/seeded/$(basename $f); done
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/seeded -S -o /dev/null 2>&1 | FileCheck --check-prefix=PRICED %s

; Every accuracy memo the first compile wrote is planted in an empty cache
; under the same name with the numbers 0 0 (an FP16 product accurate to every
; tolerance). A memo that does not carry its full key is a miss, so the second
; compile computes the same domain errors and pick as the first.

; FIRST: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04

; MEMO: matmulacc_

; PRICED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=3.137966e-04
; PRICED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; PRICED: #10 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=3.137966e-04
; PRICED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err 1.840864e-13 <= 1.000000e-12 at confidence 9.500000e-01)

; RUN: rm -rf %t && mkdir -p %t/prof %t/seeded
; RUN: sed '1i CanonicalHash = fea337c617ef94c9' %S/Inputs/raise_tcec/preprocess_matmul_body.fpprofile > %t/prof/preprocess_matmul_body.fpprofile
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/first -S -o /dev/null 2>&1 | FileCheck --check-prefixes=FIRST,PRICED %s
; RUN: ls %t/first | FileCheck --check-prefix=MEMO %s
; RUN: for f in %t/first/matmulacc_*; do echo '0 0' > %t/seeded/$(basename $f); done
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%t/prof -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/seeded -S -o /dev/null 2>&1 | FileCheck --check-prefix=PRICED %s
; REQUIRES: poseidon, scev-trunc-iv-nowrap

; Every accuracy memo the first compile wrote is planted in an empty cache
; under the same name with the numbers 0 0 (an FP16 product accurate to every
; tolerance). A memo that does not carry its full key is a miss, so the second
; compile computes the same domain errors and pick as the first.

; FIRST: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04

; MEMO: matmulacc_

; PRICED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.385254e+03  rel=6.786400e-02  domainError=3.137966e-04
; PRICED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.216647e+03  rel=4.443700e-02  domainError=3.148603e-13
; PRICED: #11 direct-dispatch cublas f16/f32  compCost/MAC=4.473499e+02  rel=8.968000e-03  domainError=3.137966e-04
; PRICED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err 3.148603e-13 <= 1.000000e-12 at confidence 9.500000e-01)

; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t/old
; RUN: for h in 3A0C4F59C6907801 3A0C4F59CB4F208F 3A0C4F59CB4F20B1 83F790B2E595C48 83F790B2E595C4A 83F790B2E595C4C 83F790B2E595C50 83F790B2E595C54 83F790B2E595C58 83F790B2E595C5C D060729D8D884DBF D060729D8D8C25C2 D060729D8D92C0C4; do echo '0 0' > %t/old/matmulacc_$h.txt; done
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/old -S -o /dev/null 2>&1 | FileCheck --check-prefix=COMPUTED --implicit-check-not=imported %s
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/new -poseidon-import-accuracy-memo=%t/old -S -o /dev/null > %t/import.log 2>&1
; RUN: FileCheck --check-prefixes=IMPORT,PLANTED --implicit-check-not=DOMERR %s < %t/import.log
; RUN: grep -c 'imported accuracy memo' %t/import.log | FileCheck --check-prefix=COUNT %s
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/new -S -o /dev/null 2>&1 | FileCheck --check-prefix=PLANTED --implicit-check-not=DOMERR --implicit-check-not=imported --implicit-check-not=recomputing %s

; The 13 accuracy results of this compile as the memo filed them before it
; stored its key (matmulacc_<hash of the inputs>.txt), each planted with the
; numbers 0 0 (an FP16 product accurate to every tolerance). In the cache
; directory they are never read: the first compile computes every result.
; Named by -poseidon-import-accuracy-memo, every one is taken over, so the FP16
; dispatch is picked, and stored under its key: the compile after it, without
; the flag, finds all 13 in the cache and computes none.

; COMPUTED: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04
; COMPUTED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=3.137966e-04
; COMPUTED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=3.148603e-13
; COMPUTED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32

; IMPORT: [poseidon] imported accuracy memo {{.*}}old{{/|\\}}matmulacc_{{[0-9A-F]+}}.txt as {{.*}}new{{/|\\}}matmulacc_{{[0-9A-F]+}}.json

; COUNT: {{^13$}}

; PLANTED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=0.000000e+00
; PLANTED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=0.000000e+00
; PLANTED: #11 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=0.000000e+00
; PLANTED: Poseidon error-budget: matmul -> direct-dispatch cublas f16/f32

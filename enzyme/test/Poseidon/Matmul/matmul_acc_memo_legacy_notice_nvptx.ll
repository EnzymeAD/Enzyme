; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t/legacy %t/flagged %t/clean
; RUN: echo '0 0' > %t/legacy/matmulacc_0123456789ABCDEF.txt && cp %t/legacy/matmulacc_0123456789ABCDEF.txt %t/flagged/
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/legacy -S -o /dev/null > %t/legacy.log 2>&1
; RUN: FileCheck --check-prefixes=NOTE,COMPUTED %s < %t/legacy.log
; RUN: grep -c 'matmulacc_\*.txt' %t/legacy.log | FileCheck --check-prefix=ONCE %s
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/flagged -poseidon-import-accuracy-memo=%t/flagged -S -o /dev/null 2>&1 | FileCheck --check-prefix=COMPUTED --implicit-check-not='matmulacc_*.txt' %s
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/clean -S -o /dev/null 2>&1 | FileCheck --check-prefix=COMPUTED --implicit-check-not='matmulacc_*.txt' %s

; A cache directory holding results in the format before the memo stored its
; keys is not read without -poseidon-import-accuracy-memo, so a compile that
; computes in front of it says so, once, naming the directory and the flag.
; With the flag given, or with no such file, it says nothing.

; NOTE: [poseidon] {{.*}}legacy holds matrix accuracy results stored before the memo kept their keys (matmulacc_*.txt), which are not read; computing them again. -poseidon-import-accuracy-memo={{.*}}legacy takes them over instead.

; ONCE: {{^1$}}

; COMPUTED: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04
; COMPUTED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32

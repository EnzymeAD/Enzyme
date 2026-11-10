; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefixes=COMPUTE,PRICED --implicit-check-not=recomputing %s
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=PRICED --implicit-check-not=DOMERR --implicit-check-not=recomputing %s
; RUN: cp -r %t/c %t/seed && sed -i 's/^\( *"seed": "\)[0-9]*"/\11"/' %t/seed/matmulacc_*.json
; RUN: grep -h '"seed"' %t/seed/matmulacc_*.json | sort -u | FileCheck --check-prefix=EDITED-SEED %s
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/seed -S -o /dev/null > %t/seed.log 2>&1
; RUN: FileCheck --check-prefixes=SEED,COMPUTE,PRICED %s < %t/seed.log
; RUN: grep -c 'was computed for a different seed' %t/seed.log | FileCheck --check-prefix=ALL %s
; RUN: cp -r %t/c %t/model && sed -i 's/^\( *"accuracy model": "\)[0-9A-F]*"/\10123456789ABCDEF"/' %t/model/matmulacc_*.json
; RUN: grep -h '"accuracy model"' %t/model/matmulacc_*.json | sort -u | FileCheck --check-prefix=EDITED-MODEL %s
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_dispatch_x1e6.csv -poseidon-tau=1e-12 -poseidon-print -poseidon-cache=%t/model -S -o /dev/null > %t/model.log 2>&1
; RUN: FileCheck --check-prefixes=MODEL,COMPUTE,PRICED %s < %t/model.log
; RUN: grep -c 'was computed for a different accuracy model' %t/model.log | FileCheck --check-prefix=ALL %s

; An accuracy memo entry holds its key's parts. The compile after the first
; finds all 13 entries under the same key and computes nothing. An entry whose
; stored key differs from the compile's in one part is recomputed, and the
; compile says which part: here the sampling seed, and the identity of the
; accuracy model, which no earlier model listed as equivalent to this one has.

; ALL: {{^13$}}

; EDITED-SEED: "seed": "1"
; EDITED-SEED-NOT: seed
; EDITED-MODEL: "accuracy model": "0123456789ABCDEF",
; EDITED-MODEL-NOT: accuracy model

; SEED: [poseidon] accuracy memo {{.*}}matmulacc_{{[0-9A-F]+}}.json was computed for a different seed; recomputing it.
; MODEL: [poseidon] accuracy memo {{.*}}matmulacc_{{[0-9A-F]+}}.json was computed for a different accuracy model; recomputing it.

; COMPUTE: DOMERR matmul in=f16 acc=f32 p95_relErr=3.137966e-04

; PRICED: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.385254e+03  rel=6.786400e-02  domainError=3.137966e-04
; PRICED: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.216647e+03  rel=4.443700e-02  domainError=3.148603e-13
; PRICED: #11 direct-dispatch cublas f16/f32  compCost/MAC=4.473499e+02  rel=8.968000e-03  domainError=3.137966e-04
; PRICED: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err 3.148603e-13 <= 1.000000e-12 at confidence 9.500000e-01)

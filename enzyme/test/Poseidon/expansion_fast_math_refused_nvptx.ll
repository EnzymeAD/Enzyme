; REQUIRES: global-fp-contract
; RUN: rm -rf %t && mkdir -p %t
; RUN: sed 's/tail call double @llvm.fmuladd/tail call contract double @llvm.fmuladd/' %S/Inputs/ds_expansion/ds_fms.ll > %t/contract.ll
; RUN: sed 's/tail call double @llvm.fmuladd/tail call fast double @llvm.fmuladd/; s/= fneg double/= fneg fast double/' %S/Inputs/ds_expansion/ds_fms.ll > %t/fast.ll
; RUN: %opt %t/contract.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-comp-cost-budget=-1 -poseidon-cache=%t/contract -S -o /dev/null 2>&1 | FileCheck --check-prefix=REFUSED %s
; RUN: %opt %t/fast.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-comp-cost-budget=-1 -poseidon-cache=%t/fast -S -o /dev/null 2>&1 | FileCheck --check-prefix=REFUSED %s
; RUN: not --crash %opt %t/contract.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-comp-cost-budget=-1 -poseidon-cache=%t/strict -poseidon-strict-mode -S -o /dev/null 2>&1 | FileCheck --check-prefix=STRICT %s

; Up to LLVM 23 the code generator of a -ffp-contract=fast unit fuses
; multiply-add pairs that carry no 'contract' flag as well, which breaks the
; error-free transformations an expansion is built from. A site whose
; operations carry 'contract' (the mark -ffp-contract=fast and -ffast-math
; leave) therefore gets no expansion candidate, with one warning that needs no
; -poseidon-print, and the solve picks among the rest; under
; -poseidon-strict-mode the refusal is fatal. (The fast-math function
; attributes are refused, in addition, up to the release that still reads each:
; LLVM 21 for unsafe-fp-math and approx-func-fp-math, LLVM 22 for
; no-nans-fp-math and no-infs-fp-math.)

; REFUSED: [poseidon] WARNING: preprocess_fms_body: floating-point expansion candidates NOT PROPOSED -- its floating-point operations carry 'contract' (the first llvm.fmuladd.f64), as under -ffp-contract=fast or -ffast-math, with which this LLVM's code generator also fuses multiply-add pairs that carry no flag.
; REFUSED-SAME: it needs -ffp-contract=on or -ffp-contract=off, without -ffast-math.
; REFUSED-NOT: NOT PROPOSED
; REFUSED-NOT: Expansion
; REFUSED: Applying solution for CS: All FP64(0%) + FP32(100%)

; STRICT: Poseidon strict mode: site 'preprocess_fms_body': its floating-point operations carry 'contract' (the first llvm.fmuladd.f64), as under -ffp-contract=fast or -ffast-math, with which this LLVM's code generator also fuses multiply-add pairs that carry no flag; floating-point expansions need -ffp-contract=on or -ffp-contract=off

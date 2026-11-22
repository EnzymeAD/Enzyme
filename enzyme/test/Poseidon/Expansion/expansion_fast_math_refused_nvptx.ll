; REQUIRES: global-fp-contract
; DEFINE: %{opt} = %opt %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/ds_expansion -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv \
; DEFINE:   -poseidon-cost-budget=-1e-9 -S -o /dev/null
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/tail call double @llvm.fmuladd/tail call contract double @llvm.fmuladd/' %S/../Inputs/ds_expansion/ds_fms.ll > %t/contract.ll
; RUN: sed 's/tail call double @llvm.fmuladd/tail call fast double @llvm.fmuladd/; s/= fneg double/= fneg fast double/' %S/../Inputs/ds_expansion/ds_fms.ll > %t/fast.ll
; RUN: %{opt} %t/contract.ll -poseidon-cache=%t/contract 2>&1 | FileCheck --check-prefix=REFUSED %s
; RUN: %{opt} %t/fast.ll -poseidon-cache=%t/fast 2>&1 | FileCheck --check-prefix=REFUSED %s
; RUN: not %{opt} %t/contract.ll -poseidon-cache=%t/strict -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=STRICT %s

; Up to LLVM 23 the code generator of a -ffp-contract=fast unit also fuses
; multiply-add pairs without a 'contract' flag, which breaks the error-free
; transformations of an expansion. A site whose operations carry 'contract'
; (the mark of -ffp-contract=fast and -ffast-math) gets no expansion candidate,
; with one warning that needs no -poseidon-print, and the solve picks among the
; rest; under -poseidon-strict-mode the refusal is fatal. (The fast-math function
; attributes are refused, in addition, up to the release that still reads each:
; LLVM 21 for unsafe-fp-math and approx-func-fp-math, LLVM 22 for
; no-nans-fp-math and no-infs-fp-math.)

; REFUSED: [poseidon] WARNING: preprocess_fms_body: floating-point expansion candidates NOT PROPOSED -- its floating-point operations carry 'contract' (the first llvm.fmuladd.f64), as under -ffp-contract=fast or -ffast-math, with which this LLVM's code generator also fuses multiply-add pairs that carry no flag.
; REFUSED-SAME: it needs -ffp-contract=on or -ffp-contract=off, without -ffast-math.
; REFUSED-NOT: NOT PROPOSED
; REFUSED-NOT: Expansion
; REFUSED: Applying solution for CS: All FP64(0%) + FP32(100%)

; STRICT: error: {{.*}}Poseidon: strict mode: site 'preprocess_fms_body': its floating-point operations carry 'contract' (the first llvm.fmuladd.f64), as under -ffp-contract=fast or -ffast-math, with which this LLVM's code generator also fuses multiply-add pairs that carry no flag; floating-point expansions need -ffp-contract=on or -ffp-contract=off

; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %S/ds_expansion_nvptx.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/c -poseidon-cost-budget=-1e-9 -poseidon-print -S -o %t/fresh.ll 2> %t/fresh.log
; RUN: %opt %S/ds_expansion_nvptx.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,%simplifycfg)" -poseidon-profile-use=%S/Inputs/ds_expansion -poseidon-enable-herbie=false -poseidon-enable-pt=true -poseidon-enable-multifloat=true -poseidon-two-tier-step=100 -poseidon-enable-three-tier=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/c -poseidon-cost-budget=-1e-9 -poseidon-print -S -o %t/replay.ll 2> %t/replay.log
; RUN: diff %t/fresh.ll %t/replay.ll
; RUN: FileCheck --check-prefix=FRESH %s < %t/fresh.log
; RUN: FileCheck --check-prefix=REPLAY --implicit-check-not="AccuracyCost: nan" --implicit-check-not=unpriced %s < %t/replay.log
; RUN: grep -E 'budget used|Applying solution' %t/fresh.log > %t/fresh.picked
; RUN: grep -E 'budget used|Applying solution' %t/replay.log > %t/replay.picked
; RUN: diff %t/fresh.picked %t/replay.picked

; A compile that reads its DP table from the cache emits the module the solve
; that wrote the table emitted, byte for byte: pricing, which the replay skips,
; leaves no declaration of its own in the module (here an FP32 fmuladd only a
; priced candidate used). The replay prints no price it did not compute, and
; the numbers both print are the same.

; FRESH: DP table for preprocess_fms_body cached to file.
; FRESH: Computation cost budget used:

; REPLAY: [poseidon] preprocess_fms_body: candidates not priced; the DP table read from the cache holds the prices its solve used
; REPLAY: Cache entry for preprocess_fms_body found. Loaded DP tables from cache.
; REPLAY: Computation cost budget used:

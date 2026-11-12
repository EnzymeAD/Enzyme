; REQUIRES: scev-trunc-iv-nowrap
; UNSUPPORTED: tcec-cumpsgemm
; RUN: rm -rf %t && mkdir -p %t
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_tcec.csv -poseidon-cost-budget=0 -poseidon-print -poseidon-cache=%t/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=PRICED --implicit-check-not=tcec-dispatch %s
; RUN: grep -h '"TCEC backend"' %t/c/table.json | FileCheck --check-prefix=KEY %s
; RUN: not %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_tcec.csv -poseidon-cache=%t/pin "-poseidon-apply-rewrites=M0=tcec-dispatch wmma m16n16k16 f16/f32" -S -o /dev/null 2>&1 | FileCheck --check-prefix=PIN %s

; A compiler built without cuMpSGEMM has no error-corrected GEMM dispatch. The
; product of Inputs/stale_cache/matmul_device.ll is not offered it although the
; cost model prices it, the DP table says so in its key, and a pin that names
; it says how to build a compiler that has it.

; PRICED: #9 ozaki-ii nm=14 wmma m16n16k16 s8/s32
; PRICED-NEXT: #10 direct-dispatch cublas f16/f32
; KEY: "TCEC backend": "none",
; PIN: error: {{.*}}Poseidon: -poseidon-apply-rewrites: matmul_body: M0 has no candidate named 'tcec-dispatch wmma m16n16k16 f16/f32': pricing dropped it, because this compiler was built without cuMpSGEMM, on which the error-corrected dispatch runs; configure it with -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON -DPOSEIDON_CUMPSGEMM_ROOT=<cuMpSGEMM checkout>

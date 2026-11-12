; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: grep -v '^tcec_dispatch_rel,' %cost_models/cm_sm_120_synthetic_dispatch.csv > %t/cm_shifted.csv
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/full "-poseidon-apply-rewrites=M0=native-dgemm cublas f64/f64" -S -o /dev/null 2>&1 | FileCheck --check-prefixes=NATIVE,FULL %s
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%t/cm_shifted.csv -poseidon-cache=%t/shifted "-poseidon-apply-rewrites=M0=native-dgemm cublas f64/f64" -S -o /dev/null 2>&1 | FileCheck --check-prefixes=NATIVE,SHIFTED %s

; The value a tolerance-selected product prints applies the same candidate,
; and the unit's dispatch manifest is the same, without the tolerance.
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/tau -poseidon-unit=tcec -poseidon-tau=1e-7 -S -o %t/tau.ll 2> %t/tau.err
; RUN: FileCheck --check-prefix=TAU %s < %t/tau.err
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/pinned -poseidon-unit=tcec "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/tau.err | paste -sd, -)" -S -o %t/pinned.ll 2> %t/pinned.err
; RUN: diff %t/tau.ll %t/pinned.ll
; RUN: diff %t/tau/launches_tcec_matmul.cu_*.json %t/pinned/launches_tcec_matmul.cu_*.json
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: |wrote descriptor' %t/tau.err > %t/tau.picks
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: |wrote descriptor' %t/pinned.err > %t/pinned.picks
; RUN: diff %t/tau.picks %t/pinned.picks

; -poseidon-apply-rewrites names a matrix-product candidate by its label. The
; native DGEMM is candidate 14 of the product of Inputs/stale_cache/
; matmul_device.ll under the dispatch fixture cost model and candidate 13 once
; the model carries no tcec_dispatch_rel row; its label names it under both.

; NATIVE: [poseidon] Selecting M0=native-dgemm cublas f64/f64: matmul[0] -> native-dgemm cublas f64/f64
; FULL: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#14)
; SHIFTED: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#13)
; NATIVE: [poseidon] pin: matmul_body:M0=native-dgemm cublas f64/f64{{$}}
; NATIVE: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=0 (standalone)

; TAU: Applying solution for matmul[0] -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (#3)
; TAU: [poseidon] pin: matmul_body:M0=ozaki-ii nm=8 wmma m16n16k16 s8/s32{{$}}

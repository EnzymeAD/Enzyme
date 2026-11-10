; REQUIRES: enzyme, scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t && echo > %t/file
; RUN: not %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/file/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=DISPATCH -DCACHE=%t/file/c %s
; RUN: not %opt %S/Inputs/stale_cache/matmul_annotated_device.ll %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,globaldce" -enzyme-preopt=false -poseidon-profile-generate -poseidon-cache=%t/file/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=PROFILE -DCACHE=%t/file/c %s
; RUN: %opt %S/Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/file/c -S -o /dev/null 2>&1 | FileCheck --check-prefix=NONE %s

; A device compile whose descriptors cannot be written stops with the path and
; the system's reason, for a dispatched product and for a profiled kernel: its
; device code already relies on the host half rewriting the launch stubs. The
; cache path here lies under a regular file. A compile that records no
; descriptor writes nothing and succeeds against the same path.

; DISPATCH: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32
; DISPATCH-NOT: wrote descriptor
; DISPATCH: LLVM ERROR: Poseidon: cannot write the launch manifest [[CACHE]]/launches_tcec_matmul.cu_{{[0-9A-F]+}}.json: Not a directory
; DISPATCH-NOT: wrote descriptor

; PROFILE-NOT: [poseidon-profgen] _Z13matmul_kernelPKdS0_PdS1_: site
; PROFILE: LLVM ERROR: Poseidon: cannot write the launch manifest [[CACHE]]/launches_tcec_matmul.cu_{{[0-9A-F]+}}.json: Not a directory
; PROFILE-NOT: [poseidon-profgen] _Z13matmul_kernelPKdS0_PdS1_: site

; NONE: [poseidon] no rewrite applied for preprocess_matmul_body
; NONE-NOT: launch manifest

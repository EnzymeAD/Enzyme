; REQUIRES: scev-trunc-iv-nowrap
; RUN: rm -rf %t && mkdir -p %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/host.ll
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/c -poseidon-unit=u -S -o /dev/null 2>&1 | FileCheck --check-prefix=WROTE %s
; RUN: ls %t/c | grep launches_ > %t/files
; RUN: FileCheck --check-prefix=FILES %s < %t/files
; RUN: count 1 < %t/files
; RUN: cat %t/c/launches_*.json | FileCheck --check-prefix=FULL %s
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/c -poseidon-unit=u -S -o /dev/null 2>&1 | FileCheck --check-prefix=NONE %s
; RUN: ls %t/c > %t/after
; RUN: not grep launches_ %t/after
; RUN: grep -v __poseidon_fp_optimize %S/../Inputs/stale_cache/matmul_device.ll > %t/siteless.ll
; RUN: %opt %t/siteless.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/fresh -S -o /dev/null
; RUN: not ls %t/fresh
; RUN: %opt %t/siteless.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-generate -poseidon-cache=%t/freshgen -S -o /dev/null
; RUN: not ls %t/freshgen
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/none -mllvm -poseidon-unit=u -S -emit-llvm -o %t/use.ll 2> %t/use.err
; RUN: count 0 < %t/use.err
; RUN: FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon %s < %t/use.ll
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/none -mllvm -poseidon-unit=u -S -emit-llvm -o %t/gen.ll 2> %t/gen.err
; RUN: count 0 < %t/gen.err
; RUN: FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon %s < %t/gen.ll
; RUN: not ls %t/none

; The device compile of tcec_matmul.cu as unit u writes the manifest of the
; unit, named after the source, with the descriptor of the dispatched product.
; A later device compile of the unit that dispatches nothing (raising off)
; removes it, and a device compile of a module without a site, under profile
; use and under profile generation, creates neither a manifest nor the cache.
; A host compile whose unit has no manifest in the cache, under profile use and
; under profile generation, says nothing, keeps the launch and creates no
; cache.

; WROTE: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; FILES: {{^}}launches_tcec_matmul.cu_{{[0-9A-F]+}}.json{{$}}

; FULL: "key": {
; FULL-NEXT: "format": "1",
; FULL-NEXT: "source file": "tcec_matmul.cu",
; FULL-NEXT: "unit": "u"
; FULL-NEXT: },
; FULL-NEXT: "value": {
; FULL-NEXT: "host dispatch": [
; FULL-NEXT: "_Z13matmul_kernelPKdS0_PdS1_ 2 0 1 16 1 12 16 16 16 16 16 16 0 0 0 0"
; FULL-NEXT: ],
; FULL-NEXT: "profile generation": []

; NONE-NOT: wrote descriptor
; NONE: [poseidon] no rewrite applied for preprocess_matmul_body
; NONE-NOT: wrote descriptor

; LAUNCH: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; LAUNCH: call {{.*}} @cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernelPKdS0_PdS1_

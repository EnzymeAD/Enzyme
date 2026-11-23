; REQUIRES: scev-trunc-iv-nowrap
; DEFINE: %{reprice} = env FAKE_HERBIE_DIR=%S/../Inputs/exact_dp/reprice_herbie \
; DEFINE:   %opt %S/../Inputs/exact_dp/reprice.ll %loadPoseidon -passes=poseidon \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/exact_dp/reprice -poseidon-enable-herbie=true \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 \
; DEFINE:   -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv
; DEFINE: %{dev} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/raise_tcec \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma
; RUN: rm -rf %t && mkdir %t

; -poseidon-apply-rewrites over sites with products.

; A site with algebraic, precision and product units
; (Inputs/exact_dp/reprice.ll): one value pins a unit of each kind, by ordinal
; and again by the names the pin line prints, which applies the same steps; and
; the errors of each kind. PT1's subgraph holds the product's reduction FMA,
; which M0's raise replaces, so the two are never taken together. The poseidon
; pass runs alone: the precision change calls libdevice float functions the
; module does not define, which poseidon-finalize refuses.
; RUN: mkdir %t/kinds
; RUN: %{reprice} -poseidon-cache=%t/kinds/ord -poseidon-explain=%t/kinds/ex -poseidon-apply-rewrites='R0_1,PT0_0,M0_2' -S -o %t/kinds/ord.ll > %t/kinds/ord.log 2>&1
; RUN: FileCheck --check-prefix=KINDS-ORD %s < %t/kinds/ord.log
; RUN: cat %t/kinds/ex/preprocess_matmul_body.explain.json | FileCheck --check-prefix=KINDS-EXPLAIN %s
; RUN: %{reprice} -poseidon-cache=%t/kinds/name "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/kinds/ord.log)" -S -o %t/kinds/name.ll 2>&1 | FileCheck --check-prefix=KINDS-NAME %s
; RUN: diff %t/kinds/ord.ll %t/kinds/name.ll
; RUN: not %{reprice} -poseidon-cache=%t/kinds/range_r -poseidon-apply-rewrites='R9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-RANGE-R %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/range_pt -poseidon-apply-rewrites='PT9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-RANGE-PT %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/range_m -poseidon-apply-rewrites='M9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-RANGE-M %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/ordinal_r -poseidon-apply-rewrites='R0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-ORDINAL-R %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/ordinal_pt -poseidon-apply-rewrites='PT0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-ORDINAL-PT %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/ordinal_m -poseidon-apply-rewrites='M0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-ORDINAL-M %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/dup_r -poseidon-apply-rewrites='R0_0,R0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-DUP-R %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/dup_pt -poseidon-apply-rewrites='PT0_0,PT0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-DUP-PT %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/dup_m -poseidon-apply-rewrites='M0_0,M0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-DUP-M %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/overlap -poseidon-apply-rewrites='PT1_0,M0_2' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-OVERLAP %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/name_m -poseidon-apply-rewrites='M0=nothing' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-NAME-M %s

; KINDS-ORD: [poseidon] Selecting R0_1: (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) -> (+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))){{$}}
; KINDS-ORD-NEXT: [poseidon] Selecting PT0_0: All FP64(0%) + FP32(100%)
; KINDS-ORD-NEXT: [poseidon] Selecting M0_2: matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32
; KINDS-ORD: Applying solution for (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)-> (+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))){{$}}
; KINDS-ORD-NEXT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; KINDS-ORD-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; KINDS-ORD: [poseidon] pin: matmul_body:R0=(+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))),matmul_body:PT0=All FP64(0%) + FP32(100%),matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32{{$}}
; KINDS-EXPLAIN: "result": {
; KINDS-EXPLAIN-NEXT: "applied": [
; KINDS-EXPLAIN-NEXT: "R0_1",
; KINDS-EXPLAIN-NEXT: "PT0_0",
; KINDS-EXPLAIN-NEXT: "M0_2"
; KINDS-EXPLAIN-NEXT: ]
; KINDS-EXPLAIN: "selector": "manual"
; KINDS-NAME: [poseidon] Selecting matmul_body:R0=(+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))): (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) -> (+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))){{$}}
; KINDS-NAME-NEXT: [poseidon] Selecting matmul_body:PT0=All FP64(0%) + FP32(100%): All FP64(0%) + FP32(100%)
; KINDS-NAME-NEXT: [poseidon] Selecting matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32: matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32
; KINDS-RANGE-R: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'R9_0': CO index 9 out of range (4 COs)
; KINDS-RANGE-PT: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'PT9_0': CS index 9 out of range (2 CSs)
; KINDS-RANGE-M: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'M9_0': CM index 9 out of range (1 CMs)
; KINDS-ORDINAL-R: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'R0_999': CO 0 has no candidate 999 that pricing kept
; KINDS-ORDINAL-PT: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'PT0_999': CS 0 has no candidate 999 that pricing kept
; KINDS-ORDINAL-M: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'M0_999': candidate index 999 out of range for CM 0 (3 candidates)
; KINDS-DUP-R: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'R0_1': CO 0 already has a rewrite applied; give one item per unit
; KINDS-DUP-PT: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'PT0_1': CS 0 already has a PT applied; give one item per unit
; KINDS-DUP-M: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'M0_1': CM 0 already has a rewrite applied; give one item per unit
; KINDS-OVERLAP: error: {{.*}}Poseidon: -poseidon-apply-rewrites item 'M0_2': CM 0 and CS 1 change the same instructions, and one of them replaces them; give one of the two
; KINDS-NAME-M: error: {{.*}}Poseidon: -poseidon-apply-rewrites: matmul_body: M0 has no candidate named 'nothing'. Its candidates are:
; KINDS-NAME-M-NEXT: {{^}}  wmma m16n16k16 f16/f32{{$}}
; KINDS-NAME-M-NEXT: {{^}}  tcec n=2 wmma m16n16k16 f16/f32{{$}}
; KINDS-NAME-M-NEXT: {{^}}  tcec n=3 wmma m16n16k16 f16/f32{{$}}
; KINDS-NAME-M-EMPTY:

; A matrix-product candidate is named by its label. The native DGEMM is
; candidate 13 of the product of Inputs/stale_cache/matmul_device.ll under the
; dispatch fixture cost model and candidate 12 once the model has no
; direct_dispatch_rel,bf16_f32 row; its label names it under both.
; RUN: mkdir %t/label
; RUN: grep -v '^direct_dispatch_rel,bf16_f32,' %cost_models/cm_sm_120_synthetic_dispatch.csv > %t/label/cm_shifted.csv
; RUN: %{dev} -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/label/full "-poseidon-apply-rewrites=M0=native-dgemm cublas f64/f64" -S -o /dev/null 2>&1 | FileCheck --check-prefixes=LABEL-NATIVE,LABEL-FULL %s
; RUN: %{dev} -poseidon-cost-model=%t/label/cm_shifted.csv -poseidon-cache=%t/label/shifted "-poseidon-apply-rewrites=M0=native-dgemm cublas f64/f64" -S -o /dev/null 2>&1 | FileCheck --check-prefixes=LABEL-NATIVE,LABEL-SHIFTED %s

; LABEL-NATIVE: [poseidon] Selecting M0=native-dgemm cublas f64/f64: matmul[0] -> native-dgemm cublas f64/f64
; LABEL-FULL: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#13)
; LABEL-SHIFTED: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#12)
; LABEL-NATIVE: [poseidon] pin: matmul_body:M0=native-dgemm cublas f64/f64{{$}}
; LABEL-NATIVE: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=0 (standalone)

; The value a tolerance-selected product prints applies the same candidate,
; and the unit's dispatch manifest is the same, without the tolerance.
; RUN: %{dev} -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/label/tau -poseidon-unit=tcec -poseidon-tau=1e-7 -S -o %t/label/tau.ll 2> %t/label/tau.err
; RUN: FileCheck --check-prefix=LABEL-TAU %s < %t/label/tau.err
; RUN: %{dev} -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/label/pinned -poseidon-unit=tcec "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/label/tau.err | paste -sd, -)" -S -o %t/label/pinned.ll 2> %t/label/pinned.err
; RUN: diff %t/label/tau.ll %t/label/pinned.ll
; RUN: diff %t/label/tau/launches_tcec_matmul.cu_*.json %t/label/pinned/launches_tcec_matmul.cu_*.json
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: |wrote descriptor' %t/label/tau.err > %t/label/tau.picks
; RUN: grep -E '^Applying solution for|^\[poseidon\] pin: |wrote descriptor' %t/label/pinned.err > %t/label/pinned.picks
; RUN: diff %t/label/tau.picks %t/label/pinned.picks

; LABEL-TAU: Applying solution for matmul[0] -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (#3)
; LABEL-TAU: [poseidon] pin: matmul_body:M0=ozaki-ii nm=8 wmma m16n16k16 s8/s32{{$}}

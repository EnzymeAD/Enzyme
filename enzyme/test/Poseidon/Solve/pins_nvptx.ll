; DEFINE: %{reprice} = env FAKE_HERBIE_DIR=%S/../Inputs/no_early_prune/reprice_herbie \
; DEFINE:   %opt %S/../Inputs/no_early_prune/reprice.ll %loadPoseidon -passes=poseidon \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/no_early_prune/reprice -poseidon-enable-rewrite=true \
; DEFINE:   -poseidon-herbie-binary=%S/../Inputs/fake-herbie.sh -poseidon-herbie-processes=1 \
; DEFINE:   -poseidon-enable-precision=true -poseidon-enable-matmul -poseidon-matmul-library-calls=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv
; DEFINE: %{dev} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../Inputs/matmul_site_profile \
; DEFINE:   -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul
; RUN: rm -rf %t && mkdir %t

; -poseidon-pin over sites with products.

; A site with algebraic, precision and product units
; (Inputs/no_early_prune/reprice.ll): one value pins a unit of each kind, by ordinal
; and again by the names the pin line prints, which applies the same steps; and
; the errors of each kind. The steps apply in the order of the pruned solve
; (algebraic, product, precision), not the order of the items. P1's subgraph holds the product's reduction FMA,
; which M0's raise replaces, so the two are never taken together. The poseidon
; pass runs alone: the precision change calls libdevice float functions the
; module does not define, which poseidon-finalize refuses.
; RUN: mkdir %t/kinds
; RUN: %{reprice} -poseidon-cache=%t/kinds/ord -poseidon-explain=%t/kinds/ex -poseidon-pin='R0_1,P0_0,M0_2' -S -poseidon-print -o %t/kinds/ord.ll > %t/kinds/ord.log 2>&1
; RUN: FileCheck --check-prefix=KINDS-ORD %s < %t/kinds/ord.log
; RUN: cat %t/kinds/ex/matmul_body.explain.json | FileCheck --check-prefix=KINDS-EXPLAIN %s
; RUN: %{reprice} -poseidon-cache=%t/kinds/name "-poseidon-pin=$(sed -n 's/^\[poseidon\] pin: //p' %t/kinds/ord.log)" -S -poseidon-print -o %t/kinds/name.ll 2>&1 | FileCheck --check-prefix=KINDS-NAME %s
; RUN: diff %t/kinds/ord.ll %t/kinds/name.ll
; RUN: not %{reprice} -poseidon-cache=%t/kinds/range_r -poseidon-pin='R9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-RANGE-R %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/range_pt -poseidon-pin='P9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-RANGE-PT %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/range_m -poseidon-pin='M9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-RANGE-M %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/ordinal_r -poseidon-pin='R0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-ORDINAL-R %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/ordinal_pt -poseidon-pin='P0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-ORDINAL-PT %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/ordinal_m -poseidon-pin='M0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-ORDINAL-M %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/dup_r -poseidon-pin='R0_0,R0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-DUP-R %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/dup_pt -poseidon-pin='P0_0,P0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-DUP-PT %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/dup_m -poseidon-pin='M0_0,M0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-DUP-M %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/overlap -poseidon-pin='P1_0,M0_2' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-OVERLAP %s
; RUN: not %{reprice} -poseidon-cache=%t/kinds/name_m -poseidon-pin='M0=nothing' -S -o /dev/null 2>&1 | FileCheck --check-prefix=KINDS-NAME-M %s

; A pin builds the module of the solve that printed it, whatever the order of
; its items: the exact solve at the cheapest point changes the precision of P0
; and raises M0, and its pin given with the items reversed applies them in that
; order again.
; RUN: %{reprice} -poseidon-cache=%t/kinds/solve -poseidon-early-prune=false -poseidon-cost-budget=-0.952 -S -poseidon-print -o %t/kinds/solve.ll 2>&1 | tee %t/kinds/solve.log | FileCheck --check-prefix=KINDS-SOLVE %s
; RUN: %{reprice} -poseidon-cache=%t/kinds/repin -poseidon-early-prune=false -poseidon-print "-poseidon-pin=$(sed -n 's/^\[poseidon\] pin: //p' %t/kinds/solve.log | awk -F, '{ for (i = NF; i > 1; --i) printf "%%s,", $i; print $1 }')" -S -o %t/kinds/repin.ll 2>&1 | FileCheck --check-prefix=KINDS-REPIN %s
; RUN: diff %t/kinds/solve.ll %t/kinds/repin.ll

; KINDS-ORD: [poseidon] Selecting R0_1: (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) -> (+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))){{$}}
; KINDS-ORD-NEXT: [poseidon] Selecting P0_0: All FP64(0%) + FP32(100%)
; KINDS-ORD-NEXT: [poseidon] Selecting M0_2: matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32
; KINDS-ORD: Applying solution for (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)-> (+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))){{$}}
; KINDS-ORD-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; KINDS-ORD-NEXT: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; KINDS-ORD: [poseidon] pin: matmul_body:R0=(+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))),matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32,matmul_body:P0=All FP64(0%) + FP32(100%){{$}}
; KINDS-EXPLAIN: "result": {
; KINDS-EXPLAIN-NEXT: "applied": [
; KINDS-EXPLAIN-NEXT: "R0_1",
; KINDS-EXPLAIN-NEXT: "M0_2",
; KINDS-EXPLAIN-NEXT: "P0_0"
; KINDS-EXPLAIN-NEXT: ]
; KINDS-EXPLAIN: "selector": "pinned"
; KINDS-NAME: [poseidon] Selecting matmul_body:R0=(+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))): (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) -> (+.f64 (-.f64 (sqrt.f64 (+.f64 v0 #s(literal 1 binary64))) (sqrt.f64 v0)) (binary32->binary64 (+.f32 (+.f32 (+.f32 (sinh.f32 (binary64->binary32 v1)) (asinh.f32 (binary64->binary32 v1))) (tan.f32 (binary64->binary32 v1))) (asin.f32 (binary64->binary32 v1))))){{$}}
; KINDS-NAME-NEXT: [poseidon] Selecting matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32: matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32
; KINDS-NAME-NEXT: [poseidon] Selecting matmul_body:P0=All FP64(0%) + FP32(100%): All FP64(0%) + FP32(100%)
; KINDS-SOLVE: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; KINDS-SOLVE-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; KINDS-SOLVE: [poseidon] pin: matmul_body:P0=All FP64(0%) + FP32(100%),matmul_body:M0=wmma m16n16k16 f16/f32{{$}}
; KINDS-REPIN: [poseidon] Selecting matmul_body:M0=wmma m16n16k16 f16/f32:
; KINDS-REPIN-NEXT: [poseidon] Selecting matmul_body:P0=All FP64(0%) + FP32(100%):
; KINDS-REPIN: Applying solution for precision change All FP64(0%) + FP32(100%) (#0)
; KINDS-REPIN-NEXT: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)
; KINDS-RANGE-R: error: {{.*}}Poseidon: -poseidon-pin item 'R9_0': there is no R9; the site has R0 to R3{{$}}
; KINDS-RANGE-PT: error: {{.*}}Poseidon: -poseidon-pin item 'P9_0': there is no P9; the site has P0 to P1{{$}}
; KINDS-RANGE-M: error: {{.*}}Poseidon: -poseidon-pin item 'M9_0': there is no M9; the site has M0{{$}}
; KINDS-ORDINAL-R: error: {{.*}}Poseidon: -poseidon-pin item 'R0_999': R0 has no candidate 999 that pricing kept
; KINDS-ORDINAL-PT: error: {{.*}}Poseidon: -poseidon-pin item 'P0_999': P0 has no candidate 999 that pricing kept
; KINDS-ORDINAL-M: error: {{.*}}Poseidon: -poseidon-pin item 'M0_999': M0 has no candidate 999; it has 3
; KINDS-DUP-R: error: {{.*}}Poseidon: -poseidon-pin item 'R0_1': R0 already has a rewrite applied; give one item per unit
; KINDS-DUP-PT: error: {{.*}}Poseidon: -poseidon-pin item 'P0_1': P0 already has a precision change applied; give one item per unit
; KINDS-DUP-M: error: {{.*}}Poseidon: -poseidon-pin item 'M0_1': M0 already has a candidate applied; give one item per unit
; KINDS-OVERLAP: error: {{.*}}Poseidon: -poseidon-pin item 'M0_2': M0 and P1 change the same instructions, and one of them replaces them; give one of the two
; KINDS-NAME-M: error: {{.*}}Poseidon: -poseidon-pin: matmul_body: M0 has no candidate named 'nothing'; its candidates are:
; KINDS-NAME-M-NEXT: {{^}}  wmma m16n16k16 f16/f32{{$}}
; KINDS-NAME-M-NEXT: {{^}}  tcec n=2 wmma m16n16k16 f16/f32{{$}}
; KINDS-NAME-M-NEXT: {{^}}  tcec n=3 wmma m16n16k16 f16/f32{{$}}
; KINDS-NAME-M-EMPTY:

; A matrix-product candidate is named by its label. The native DGEMM is
; candidate 13 of the product of Inputs/stale_cache/matmul_device.ll under the
; dispatch fixture cost model and candidate 12 once the model has no
; library_direct_rel,bf16_f32 row; its label names it under both.
; RUN: mkdir %t/label
; RUN: grep -v '^library_direct_rel,bf16_f32,' %cost_models/cm_sm_120_synthetic_dispatch.csv > %t/label/cm_shifted.csv
; RUN: %{dev} -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/label/full "-poseidon-pin=M0=native-dgemm cublas f64/f64" -S -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefixes=LABEL-NATIVE,LABEL-FULL %s
; RUN: %{dev} -poseidon-cost-model=%t/label/cm_shifted.csv -poseidon-cache=%t/label/shifted "-poseidon-pin=M0=native-dgemm cublas f64/f64" -S -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefixes=LABEL-NATIVE,LABEL-SHIFTED %s

; LABEL-NATIVE: [poseidon] Selecting M0=native-dgemm cublas f64/f64: matmul[0] -> native-dgemm cublas f64/f64
; LABEL-FULL: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#13)
; LABEL-SHIFTED: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#12)
; LABEL-NATIVE: [poseidon] pin: matmul_body:M0=native-dgemm cublas f64/f64{{$}}
; LABEL-NATIVE: [poseidon] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=0 (standalone)

; The value a tolerance-selected product prints applies the same candidate,
; and the unit's dispatch manifest is the same, without the tolerance.
; RUN: %{dev} -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/label/tau -poseidon-compilation-id=tcec -poseidon-tau=1e-7 -S -poseidon-print -o %t/label/tau.ll 2> %t/label/tau.err
; RUN: FileCheck --check-prefix=LABEL-TAU %s < %t/label/tau.err
; RUN: %{dev} -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/label/pinned -poseidon-compilation-id=tcec -poseidon-print "-poseidon-pin=$(sed -n 's/^\[poseidon\] pin: //p' %t/label/tau.err | paste -sd, -)" -S -o %t/label/pinned.ll 2> %t/label/pinned.err
; RUN: diff %t/label/tau.ll %t/label/pinned.ll
; RUN: diff %t/label/tau/launches_tcec_matmul.cu_*.json %t/label/pinned/launches_tcec_matmul.cu_*.json
; RUN: grep -E '^\[poseidon\] Applying solution for|^\[poseidon\] pin: |wrote descriptor' %t/label/tau.err > %t/label/tau.picks
; RUN: grep -E '^\[poseidon\] Applying solution for|^\[poseidon\] pin: |wrote descriptor' %t/label/pinned.err > %t/label/pinned.picks
; RUN: diff %t/label/tau.picks %t/label/pinned.picks

; LABEL-TAU: Applying solution for matmul[0] -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (#3)
; LABEL-TAU: [poseidon] pin: matmul_body:M0=ozaki-ii nm=8 wmma m16n16k16 s8/s32{{$}}

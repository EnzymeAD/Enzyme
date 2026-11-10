; RUN: rm -rf %t && mkdir -p %t
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/ord -poseidon-explain=%t/ex -poseidon-apply-rewrites='R0_1,PT1_0,M0_2' -S -o %t/ord.ll > %t/ord.log 2>&1
; RUN: FileCheck --check-prefix=ORD %s < %t/ord.log
; RUN: cat %t/ex/preprocess_matmul_body.explain.json | FileCheck --check-prefix=EXPLAIN %s
; RUN: env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/name "-poseidon-apply-rewrites=$(sed -n 's/^\[poseidon\] pin: //p' %t/ord.log)" -S -o %t/name.ll > %t/name.log 2>&1
; RUN: FileCheck --check-prefix=NAME %s < %t/name.log
; RUN: diff %t/ord.ll %t/name.ll
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/range_r -poseidon-apply-rewrites='R9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=RANGE_R %s
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/range_pt -poseidon-apply-rewrites='PT9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=RANGE_PT %s
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/range_m -poseidon-apply-rewrites='M9_0' -S -o /dev/null 2>&1 | FileCheck --check-prefix=RANGE_M %s
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/ordinal_r -poseidon-apply-rewrites='R0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=ORDINAL_R %s
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/ordinal_pt -poseidon-apply-rewrites='PT0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=ORDINAL_PT %s
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/ordinal_m -poseidon-apply-rewrites='M0_999' -S -o /dev/null 2>&1 | FileCheck --check-prefix=ORDINAL_M %s
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/dup_r -poseidon-apply-rewrites='R0_0,R0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=DUP_R %s
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/dup_pt -poseidon-apply-rewrites='PT0_0,PT0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=DUP_PT %s
; RUN: not --crash env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/dup_m -poseidon-apply-rewrites='M0_0,M0_1' -S -o /dev/null 2>&1 | FileCheck --check-prefix=DUP_M %s
; RUN: not env FAKE_HERBIE_DIR=%S/Inputs/exact_dp/reprice_herbie %opt %S/exact_dp_reprice_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/Inputs/exact_dp/reprice -poseidon-enable-herbie=true -poseidon-herbie-binary=%S/Inputs/exact_dp/fake-herbie.sh -poseidon-herbie-processes=1 -poseidon-enable-pt=true -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%S/Inputs/cm_gpu_fixture_x1e6.csv -poseidon-cache=%t/name_m -poseidon-apply-rewrites='M0=nothing' -S -o /dev/null 2>&1 | FileCheck --check-prefix=NAME_M %s
; REQUIRES: poseidon, scev-trunc-iv-nowrap

; -poseidon-apply-rewrites over a site with algebraic, precision and product
; units (exact_dp_reprice_nvptx.ll): one value pins a unit of each kind, by
; ordinal and again by the names the pin line prints, which applies the same
; steps; and the errors of each kind. The poseidon pass runs alone, as in
; dp_cache_replay_all_kinds_nvptx.ll.

; ORD: [poseidon] Selecting R0_1: (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) -> (+.f64 (-.f64 (sqrt.f64
; ORD-NEXT: [poseidon] Selecting PT1_0: All FP64(0%) + FP32(100%)
; ORD-NEXT: [poseidon] Selecting M0_2: matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32
; ORD: Applying solution for (+ (- (sqrt (+ v0 1)) (sqrt v0)) (+ (+ (+ (sinh v1) (asinh v1)) (tan v1)) (asin v1))) --(1)->
; ORD-NEXT: Applying solution for CS: All FP64(0%) + FP32(100%) (#0)
; ORD-NEXT: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)
; ORD: [poseidon] pin: matmul_body:R0=(+.f64 (-.f64 (sqrt.f64 {{.*}}),matmul_body:PT1=All FP64(0%) + FP32(100%),matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32

; EXPLAIN: "result": {
; EXPLAIN-NEXT: "applied": [
; EXPLAIN-NEXT: "R0_1",
; EXPLAIN-NEXT: "PT1_0",
; EXPLAIN-NEXT: "M0_2"
; EXPLAIN-NEXT: ]
; EXPLAIN: "selector": "manual"

; NAME: [poseidon] Selecting matmul_body:R0=
; NAME-NEXT: [poseidon] Selecting matmul_body:PT1=All FP64(0%) + FP32(100%): All FP64(0%) + FP32(100%)
; NAME-NEXT: [poseidon] Selecting matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32: matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32

; RANGE_R: LLVM ERROR: [poseidon] CO index 9 out of range (4 COs)
; RANGE_PT: LLVM ERROR: [poseidon] CS index 9 out of range (2 CSs)
; RANGE_M: LLVM ERROR: [poseidon] CM index 9 out of range (1 CMs)
; ORDINAL_R: LLVM ERROR: [poseidon] CO 0 has no candidate 999 that pricing kept ('R0_999')
; ORDINAL_PT: LLVM ERROR: [poseidon] CS 0 has no candidate 999 that pricing kept ('PT0_999')
; ORDINAL_M: LLVM ERROR: [poseidon] candidate index 999 out of range for CM 0 (3 candidates)
; DUP_R: LLVM ERROR: [poseidon] CO 0 already has a rewrite applied (duplicate ID 'R0_1')
; DUP_PT: LLVM ERROR: [poseidon] CS 0 already has a PT applied (duplicate ID 'PT0_1')
; DUP_M: LLVM ERROR: [poseidon] CM 0 already has a rewrite applied (duplicate ID 'M0_1')
; NAME_M: LLVM ERROR: [poseidon] matmul_body: M0 has no candidate named 'nothing'. Its candidates are:
; NAME_M-NEXT: wmma m16n16k16 f16/f32
; NAME_M-NEXT: tcec n=2 wmma m16n16k16 f16/f32
; NAME_M-NEXT: tcec n=3 wmma m16n16k16 f16/f32

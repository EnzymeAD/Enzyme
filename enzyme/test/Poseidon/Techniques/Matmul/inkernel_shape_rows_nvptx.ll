; DEFINE: %{opt} = %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-matmul-library-calls=false -poseidon-tau=1e-14 -S \
; DEFINE:   -o /dev/null -poseidon-print
; RUN: rm -rf %t && mkdir -p %t/scaled
; RUN: sed -e 's/^MaxGridDims = 1 1 1$/MaxGridDims = 1 8 1/' -e 's/^LaunchCount = 2$/LaunchCount = 2\nReductionTrip = 0 8/' %S/../../Inputs/matmul_site_profile/matmul_body.fpprofile > %t/scaled/matmul_body.fpprofile
; RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/one.csv && echo 'inkernel_rel,f16_f32_m16n16k16_browmajor,0.5' >> %t/one.csv
; RUN: cp %t/one.csv %t/all.csv && echo 'inkernel_rel,tcec_n2_f16_f32_m16n16k16_browmajor,0.5' >> %t/all.csv && echo 'inkernel_rel,tcec_n3_f16_f32_m16n16k16_browmajor,0.5' >> %t/all.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/col.csv && echo 'inkernel_rel,f16_f32_m16n16k16_bcolmajor,0.5' >> %t/col.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/ext.csv && echo 'inkernel_rel,f16_f32_m16n8k16_browmajor,0.5' >> %t/ext.csv
; RUN: %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/c_class 2>&1 \
; RUN:   | FileCheck --check-prefixes=ALL,CLASS %s
; RUN: %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-cost-model=%t/col.csv -poseidon-cache=%t/c_col 2>&1 \
; RUN:   | FileCheck --check-prefixes=ALL,CLASS %s
; RUN: %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-cost-model=%t/ext.csv -poseidon-cache=%t/c_ext 2>&1 \
; RUN:   | FileCheck --check-prefixes=ALL,CLASS %s
; RUN: %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-cost-model=%t/one.csv -poseidon-cache=%t/c_one 2>&1 \
; RUN:   | FileCheck --check-prefixes=TCEC,ONE %s
; RUN: %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-cost-model=%t/all.csv -poseidon-cache=%t/c_all 2>&1 \
; RUN:   | FileCheck --check-prefix=EXACT --implicit-check-not='in-kernel raise as' %s
; RUN: %{opt} -poseidon-profile-use=%t/scaled -poseidon-cost-model=%t/all.csv -poseidon-cache=%t/c_scaled 2>&1 \
; RUN:   | FileCheck --check-prefix=CLASS --implicit-check-not='in-kernel raise as' %s
; RUN: cp %t/one.csv %t/strict.csv && echo 'wmma_mma_m8n8k4,f64_f64,64' >> %t/strict.csv
; RUN: for c in f16_f16 bf16_f32 tf32_f32 f64_f64 tcec_n5_f16_f32 tcec_n2_bf16_f32 tcec_n2_tf32_f32 tcec_n3_tf32_f32 tcec_n5_tf32_f32; do \
; RUN:   echo "inkernel_rel,$c,0.5"; echo "inkernel_rel,${c}_m16n16k16_browmajor,0.5"; done >> %t/strict.csv
; RUN: not %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-cost-model=%t/strict.csv -poseidon-cache=%t/c_strict -poseidon-strict-mode 2>&1 \
; RUN:   | FileCheck --check-prefix=STRICT %s

; The in-kernel raises of the product of Inputs/stale_cache/matmul_device.ll,
; whose extents are constants of the compile: profiled at its reduction length
; over one 16 x 16 block, it runs as 16 x 16 x 16 with B strided along K, and
; each class is priced from its row at that shape. A class without one is
; priced from its class row, with one warning naming the classes, the row and
; the command that measures it; a row of another layout or of other extents is
; not read, and strict mode refuses the estimate. Profiled over eight blocks
; at a reduction of 8, the product's output extents at its reduction of 16 are
; run-time values (no guard bounds them), so the class rows price it with no
; warning. native_dgemm_shape_rows_nvptx.ll covers the rows of its library
; calls.

; ALL: Poseidon: matrix product 0, 16x16x16: in-kernel raise as f16_f32, tcec_n2_f16_f32, tcec_n3_f16_f32 is priced from the row of its class: this device's cost model carries no measured 'inkernel_rel,<class>_m16n16k16_browmajor' row; poseidon-calibrate --gpu <id> --product 16x16x16:browmajor measures it{{$}}
; ONE: Poseidon: matrix product 0, 16x16x16: in-kernel raise as tcec_n2_f16_f32, tcec_n3_f16_f32 is priced from the row of its class: this device's cost model carries no measured 'inkernel_rel,<class>_m16n16k16_browmajor' row; poseidon-calibrate --gpu <id> --product 16x16x16:browmajor measures it{{$}}

; CLASS: matmul[0] pricing: baseline/MAC=5.120000e+02
; CLASS-NEXT: {{#}}0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02
; CLASS-NEXT: {{#}}1 tcec n=2 wmma m16n16k16 f16/f32  compCost/MAC=4.000000e+01  rel=7.812500e-02
; CLASS-NEXT: {{#}}2 tcec n=3 wmma m16n16k16 f16/f32  compCost/MAC=4.800000e+01  rel=9.375000e-02

; TCEC: matmul[0] pricing: baseline/MAC=5.120000e+02
; TCEC-NEXT: {{#}}0 wmma m16n16k16 f16/f32  compCost/MAC=2.560000e+02  rel=5.000000e-01
; TCEC-NEXT: {{#}}1 tcec n=2 wmma m16n16k16 f16/f32  compCost/MAC=4.000000e+01  rel=7.812500e-02

; EXACT: matmul[0] pricing: baseline/MAC=5.120000e+02
; EXACT-NEXT: {{#}}0 wmma m16n16k16 f16/f32  compCost/MAC=2.560000e+02  rel=5.000000e-01
; EXACT-NEXT: {{#}}1 tcec n=2 wmma m16n16k16 f16/f32  compCost/MAC=2.560000e+02  rel=5.000000e-01
; EXACT-NEXT: {{#}}2 tcec n=3 wmma m16n16k16 f16/f32  compCost/MAC=2.560000e+02  rel=5.000000e-01

; STRICT: error: {{.*}}strict mode: in-kernel raise as tcec_n2_f16_f32, tcec_n3_f16_f32 has no measured 'inkernel_rel,<class>_m16n16k16_browmajor' row in {{.*}}strict.csv; poseidon-calibrate --gpu <id> --product 16x16x16:browmajor measures it

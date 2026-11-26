; DEFINE: %{opt} = %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-enable-rewrite=false -poseidon-enable-precision=false \
; DEFINE:   -poseidon-enable-matmul -poseidon-tau=1e-14 -S -o /dev/null
; DEFINE: %{raise} = %{opt} -poseidon-matmul-library-calls=false -poseidon-print
; RUN: rm -rf %t && mkdir -p %t/scaled %t/rect
; RUN: sed -e 's/^MaxGridDims = 1 1 1$/MaxGridDims = 1 8 1/' -e 's/^LaunchCount = 2$/LaunchCount = 2\nReductionTrip = 0 8/' %S/../../Inputs/matmul_site_profile/matmul_body.fpprofile > %t/scaled/matmul_body.fpprofile
; RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/one.csv && echo 'inkernel_rel,f16_f32_m16n16k16_browmajor,0.5' >> %t/one.csv
; RUN: cp %t/one.csv %t/all.csv && echo 'inkernel_rel,tcec_n2_f16_f32_m16n16k16_browmajor,0.5' >> %t/all.csv && echo 'inkernel_rel,tcec_n3_f16_f32_m16n16k16_browmajor,0.5' >> %t/all.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/col.csv && echo 'inkernel_rel,f16_f32_m16n16k16_bcolmajor,0.5' >> %t/col.csv
; RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/ext.csv && echo 'inkernel_rel,f16_f32_m16n8k16_browmajor,0.5' >> %t/ext.csv
; RUN: %{raise} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cache=%t/c_class 2>&1 | FileCheck \
; RUN:   --check-prefixes=ALL,CLASS %s
; RUN: %{raise} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%t/col.csv -poseidon-cache=%t/c_col 2>&1 | FileCheck --check-prefixes=ALL,CLASS \
; RUN:   %s
; RUN: %{raise} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%t/ext.csv -poseidon-cache=%t/c_ext 2>&1 | FileCheck --check-prefixes=ALL,CLASS \
; RUN:   %s
; RUN: %{raise} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%t/one.csv -poseidon-cache=%t/c_one 2>&1 | FileCheck --check-prefixes=TCEC,ONE %s
; RUN: %{raise} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%t/all.csv -poseidon-cache=%t/c_all 2>&1 | FileCheck --check-prefix=EXACT \
; RUN:   --implicit-check-not='in-kernel raise as' %s
; RUN: %{raise} -poseidon-profile-use=%t/scaled -poseidon-early-prune -poseidon-cost-model=%t/all.csv -poseidon-cache=%t/c_scaled 2>&1 | FileCheck --check-prefix=CLASS \
; RUN:   --implicit-check-not='in-kernel raise as' %s
; RUN: cp %t/one.csv %t/strict.csv && echo 'wmma_mma_m8n8k4,f64_f64,64' >> %t/strict.csv
; RUN: for c in f16_f16 bf16_f32 tf32_f32 f64_f64 tcec_n5_f16_f32 tcec_n2_bf16_f32 tcec_n2_tf32_f32 tcec_n3_tf32_f32 tcec_n5_tf32_f32; do echo "inkernel_rel,$c,0.5"; echo \
; RUN:   "inkernel_rel,${c}_m16n16k16_browmajor,0.5"; done >> %t/strict.csv
; RUN: not %{raise} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%t/strict.csv -poseidon-cache=%t/c_strict -poseidon-strict-mode 2>&1 | \
; RUN:   FileCheck --check-prefix=STRICT-RAISE %s
; RUN: sed 's/^MaxGridDims = 1 1 1$/MaxGridDims = 1 8 1/' %S/../../Inputs/matmul_site_profile/matmul_body.fpprofile > %t/rect/matmul_body.fpprofile
; RUN: cp %cost_models/cm_sm_120_synthetic_library.csv %t/oz.csv && echo 'library_ozaki_rect_rel,nm14_m128n16k16,0.1' >> %t/oz.csv
; RUN: cp %t/oz.csv %t/low.csv && echo 'library_ozaki_rect_rel,dgemm_m128n16k16,0.05' >> %t/low.csv
; RUN: cp %t/oz.csv %t/high.csv && echo 'library_ozaki_rect_rel,dgemm_m128n16k16,0.2' >> %t/high.csv
; RUN: cp %t/oz.csv %t/mk.csv && echo 'library_ozaki_rect_rel,dgemm_m128k16,0.05' >> %t/mk.csv
; RUN: grep -v '^library_ozaki_rel,dgemm,' %cost_models/cm_sm_120_synthetic_library.csv > %t/none.csv
; RUN: cp %t/none.csv %t/rowonly.csv && echo 'library_ozaki_rect_rel,dgemm_m128n16k16,0.2' >> %t/rowonly.csv
; RUN: cp %cost_models/cm_sm_120_synthetic_library.csv %t/ozmk.csv && echo 'library_ozaki_rect_rel,nm14_m128k16,0.1' >> %t/ozmk.csv
; RUN: cp %cost_models/cm_sm_120_synthetic_library.csv %t/square.csv && echo 'library_ozaki_rect_rel,dgemm_m16n16k16,0.2' >> %t/square.csv
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-early-prune -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-cache=%t/c_sq -poseidon-print 2>&1 | FileCheck \
; RUN:   --check-prefixes=WARN,OZEST,SQ,NATIVE %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-early-prune -poseidon-cost-model=%t/oz.csv -poseidon-cache=%t/c_oz -poseidon-print 2>&1 | FileCheck --check-prefixes=WARN,SQ,OZAKI %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-early-prune -poseidon-cost-model=%t/ozmk.csv -poseidon-cache=%t/c_ozmk -poseidon-print 2>&1 | FileCheck --check-prefixes=WARN,OZEST,SQ,NATIVE %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-early-prune -poseidon-cost-model=%t/low.csv -poseidon-cache=%t/c_low -poseidon-print 2>&1 | FileCheck --check-prefixes=LOW,NATIVE \
; RUN:   --implicit-check-not='native-dgemm library call' %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-early-prune -poseidon-cost-model=%t/high.csv -poseidon-cache=%t/c_high -poseidon-print 2>&1 | FileCheck --check-prefixes=HIGH,OZAKI \
; RUN:   --implicit-check-not='native-dgemm library call' %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-early-prune -poseidon-cost-model=%t/mk.csv -poseidon-cache=%t/c_mk -poseidon-print 2>&1 | FileCheck --check-prefixes=WARN,SQ,OZAKI %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-early-prune -poseidon-cost-model=%t/rowonly.csv -poseidon-cache=%t/c_rowonly -poseidon-print 2>&1 | FileCheck --check-prefix=HIGH \
; RUN:   --implicit-check-not='native-dgemm library call' %s
; RUN: %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-cache=%t/c_square \
; RUN:   -poseidon-print 2>&1 | FileCheck --check-prefix=SQ --implicit-check-not='native-dgemm library call' %s
; RUN: %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%t/square.csv -poseidon-cache=%t/c_squarerow -poseidon-print 2>&1 | FileCheck \
; RUN:   --check-prefix=HIGH --implicit-check-not='native-dgemm library call' %s
; RUN: %{opt} -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cost-model=%t/none.csv -poseidon-cache=%t/c_none 2>&1 | FileCheck --check-prefix=NONE \
; RUN:   --implicit-check-not='native-dgemm cublas' %s
; RUN: not %{opt} -poseidon-profile-use=%t/rect -poseidon-early-prune -poseidon-cost-model=%t/oz.csv -poseidon-cache=%t/c_strict -poseidon-strict-mode 2>&1 | FileCheck --check-prefix=STRICT-LIB %s

; The raises and library calls of Inputs/stale_cache/matmul_device.ll are priced
; from the cost rows of their compile-time extents, else from class or square
; rows with one warning naming the row; strict mode refuses the estimate.

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

; STRICT-RAISE: error: {{.*}}strict mode: in-kernel raise as tcec_n2_f16_f32, tcec_n3_f16_f32 has no measured 'inkernel_rel,<class>_m16n16k16_browmajor' row in {{.*}}strict.csv; poseidon-calibrate --gpu <id> --product 16x16x16:browmajor measures it

; OZEST: Poseidon: matrix product 0, 16x16x16: ozaki-ii library call at nm=8, 9, 10, 11, 12, 13, 14 is priced as the square product it embeds in (the square row times 64.00): this device's cost model carries no measured 'library_ozaki_rect_rel,nm<nm>_m128n16k16' row; poseidon-calibrate --gpu <id> --product 128x16x16 measures it
; WARN: Poseidon: matrix product 0, 16x16x16: native-dgemm library call at 128x16x16 is priced from the square 'library_ozaki_rel,dgemm' row: this device's cost model carries no measured 'library_ozaki_rect_rel,dgemm_m128n16k16' row; poseidon-calibrate --gpu <id> --product 128x16x16 measures it
; SQ: native-dgemm cublas f64/f64  compCost/MAC=2.560000e+02  rel=5.000000e-01
; LOW: native-dgemm cublas f64/f64  compCost/MAC=2.560000e+01  rel=5.000000e-02
; HIGH: native-dgemm cublas f64/f64  compCost/MAC=1.024000e+02  rel=2.000000e-01

; NATIVE: Applying solution for matmul[0] -> native-dgemm cublas f64/f64
; OZAKI: Applying solution for matmul[0] -> ozaki-ii nm=14 wmma m16n16k16 s8/s32

; NONE: Poseidon: matrix product 0, 16x16x16: native-dgemm library call is not proposed: this device's cost model carries no measured 'library_ozaki_rel,dgemm' row; poseidon-calibrate --gpu <id> measures it

; STRICT-LIB: error: {{.*}}strict mode: {{.*}} has no measured '{{[a-z0-9_]+,[a-z0-9_]+}}' row in {{.*}}oz.csv; poseidon-calibrate --gpu <id>{{.*}} measures it

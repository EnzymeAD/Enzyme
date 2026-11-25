; DEFINE: %{opt} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-tau=1e-14 -S \
; DEFINE:   -o /dev/null
; RUN: rm -rf %t && mkdir -p %t/rect
; RUN: sed 's/^MaxGridDims = 1 1 1$/MaxGridDims = 1 8 1/' %S/../Inputs/raise_tcec/preprocess_matmul_body.fpprofile > %t/rect/preprocess_matmul_body.fpprofile
; RUN: cp %cost_models/cm_sm_120_synthetic_dispatch.csv %t/oz.csv && echo 'ozaki_dispatch_rect_rel,nm14_m128k16,0.1' >> %t/oz.csv
; RUN: cp %t/oz.csv %t/low.csv && echo 'ozaki_dispatch_rect_rel,dgemm_m128n16k16,0.05' >> %t/low.csv
; RUN: cp %t/oz.csv %t/high.csv && echo 'ozaki_dispatch_rect_rel,dgemm_m128n16k16,0.2' >> %t/high.csv
; RUN: cp %t/oz.csv %t/mk.csv && echo 'ozaki_dispatch_rect_rel,dgemm_m128k16,0.05' >> %t/mk.csv
; RUN: grep -v '^ozaki_dispatch_rel,dgemm,' %cost_models/cm_sm_120_synthetic_dispatch.csv > %t/none.csv
; RUN: cp %t/none.csv %t/rowonly.csv && echo 'ozaki_dispatch_rect_rel,dgemm_m128n16k16,0.2' >> %t/rowonly.csv
; RUN: cp %cost_models/cm_sm_120_synthetic_dispatch.csv %t/square.csv && echo 'ozaki_dispatch_rect_rel,dgemm_m16n16k16,0.2' >> %t/square.csv
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/c_sq -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefixes=WARN,SQ,NATIVE --implicit-check-not='the native DGEMM runs at' %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-cost-model=%t/oz.csv -poseidon-cache=%t/c_oz -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefixes=WARN,SQ,OZAKI %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-cost-model=%t/low.csv -poseidon-cache=%t/c_low -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefixes=LOW,NATIVE --implicit-check-not='the native DGEMM runs at' %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-cost-model=%t/high.csv -poseidon-cache=%t/c_high -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefixes=HIGH,OZAKI --implicit-check-not='the native DGEMM runs at' %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-cost-model=%t/mk.csv -poseidon-cache=%t/c_mk -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefixes=WARN,SQ,OZAKI %s
; RUN: %{opt} -poseidon-profile-use=%t/rect -poseidon-cost-model=%t/rowonly.csv -poseidon-cache=%t/c_rowonly -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefix=HIGH --implicit-check-not='the native DGEMM runs at' %s
; RUN: %{opt} -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/c_square -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefix=SQ --implicit-check-not='the native DGEMM runs at' %s
; RUN: %{opt} -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-cost-model=%t/square.csv -poseidon-cache=%t/c_squarerow -poseidon-print 2>&1 \
; RUN:   | FileCheck --check-prefix=HIGH --implicit-check-not='the native DGEMM runs at' %s
; RUN: not %{opt} -poseidon-profile-use=%t/rect -poseidon-cost-model=%t/none.csv -poseidon-cache=%t/c_none 2>&1 \
; RUN:   | FileCheck --check-prefix=NONE %s

; The native DGEMM candidate of the product of Inputs/stale_cache/
; matmul_device.ll, whose dispatch extents are constants of the compile. Made
; rectangular by profiling it over eight CTAs along its rows, it is dispatched
; as 128 x 16 x 16 (padWaste 64): priced from the row measured at those
; extents, else from the square row with a warning naming that row; a row
; measured at another column count is not read, and a model with no DGEMM row
; refuses the compile. At tau=1e-14 only Ozaki-II nm=14 and the native DGEMM
; clear the target, so the row decides between them. With the profile of
; Inputs/raise_tcec the square 16 x 16 x 16 product reads its own row and is
; never warned about.
; native_dgemm_runtime_cols_nvptx.cu covers a product with a run-time column
; count.

; WARN: [poseidon] WARNING: matmul[0] 16x16x16: the native DGEMM runs at 128x16x16 (padWaste=6.400000e+01), but the cost model has no measured 'ozaki_dispatch_rect_rel,dgemm_m128n16k16' row, so it is priced from the SQUARE 'ozaki_dispatch_rel,dgemm' row, a rate a DGEMM of this shape need not reach. Measure the row (poseidon-calibrate --ozaki-rect 128x16x16) before trusting a pick at this site.
; SQ: native-dgemm cublas f64/f64  compCost/MAC=2.560000e+02  rel=5.000000e-01
; LOW: native-dgemm cublas f64/f64  compCost/MAC=2.560000e+01  rel=5.000000e-02
; HIGH: native-dgemm cublas f64/f64  compCost/MAC=1.024000e+02  rel=2.000000e-01

; NATIVE: Poseidon error-budget: matmul -> native-dgemm cublas f64/f64 (domain err
; OZAKI: Poseidon error-budget: matmul -> ozaki-ii nm=14 wmma m16n16k16 s8/s32 (domain err

; NONE: error: {{.*}}Poseidon: the cost model {{.*}}none.csv has no 'ozaki_dispatch_rel,dgemm' row for the native DGEMM candidate; measure it with poseidon-calibrate.

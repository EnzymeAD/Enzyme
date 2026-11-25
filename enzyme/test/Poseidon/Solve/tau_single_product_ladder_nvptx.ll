; DEFINE: %{opt} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize" -poseidon-enable-rewrite=false \
; DEFINE:   -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-print \
; DEFINE:   -S -o /dev/null
; DEFINE: %{cm} = -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-2 2>&1 | FileCheck --check-prefix=F16 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-3 2>&1 | FileCheck --check-prefix=F16 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-4 2>&1 | FileCheck --check-prefix=NM8 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-8 2>&1 | FileCheck --check-prefix=NM8 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-9 2>&1 | FileCheck --check-prefix=NM9 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-10 2>&1 | FileCheck --check-prefix=NM10 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-11 2>&1 | FileCheck --check-prefix=NM11 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-12 2>&1 | FileCheck --check-prefix=NM12 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-13 2>&1 | FileCheck --check-prefix=NM13 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-14 2>&1 | FileCheck --check-prefix=NM14 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-15 2>&1 | FileCheck --check-prefix=NM14 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-16 2>&1 | FileCheck --check-prefix=NATIVE %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/c -poseidon-tau=1e-18 2>&1 | FileCheck --check-prefix=NATIVE %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/k -poseidon-matmul-library-calls=false -poseidon-tau=1e-3 2>&1 | FileCheck --check-prefix=WMMA %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/k -poseidon-matmul-library-calls=false -poseidon-tau=1e-6 2>&1 | FileCheck --check-prefix=TCEC2 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/k -poseidon-matmul-library-calls=false -poseidon-tau=1e-7 2>&1 | FileCheck --check-prefix=TCEC3 %s
; RUN: %{opt} %{cm} -poseidon-cache=%t/k -poseidon-matmul-library-calls=false -poseidon-tau=1e-8 2>&1 | FileCheck --check-prefix=WRITTEN %s
; RUN: sed 's/^library_direct_rel,tf32_f32,.*/library_direct_rel,tf32_f32,0.00784375/' %cost_models/cm_sm_120_synthetic_dispatch.csv > %t/tie.csv
; RUN: sed 's/^library_direct_rel,tf32_f32,.*/library_direct_rel,tf32_f32,0.0078671875/' %cost_models/cm_sm_120_synthetic_dispatch.csv > %t/apart.csv
; RUN: %{opt} -poseidon-cost-model=%t/tie.csv -poseidon-cache=%t/tie -poseidon-tau=1e-3 2>&1 | FileCheck --check-prefix=TF32 %s
; RUN: %{opt} -poseidon-cost-model=%t/apart.csv -poseidon-cache=%t/apart -poseidon-tau=1e-3 2>&1 | FileCheck --check-prefix=F16 %s

; A site whose only unit is one matrix product, under a ladder of accuracy
; targets that crosses the modelled relative error of each candidate. Per MAC,
; the loop as written costs 512; the candidates cost (error): direct f16 4
; (3.14e-4), bf16 6 (2.34e-3), tf32 8 (3.137e-4), Ozaki-II nm=8 to nm=14 16 to
; 28 (8.0e-9, 5.8e-10, 4.2e-11, 2.8e-12, 1.8e-13, 1.2e-14, 8.6e-16), native
; DGEMM 256 (0); in the kernel, wmma f16 32 (3.14e-4), TCEC n=2 40 (1.6e-7),
; n=3 48 (5.9e-8). Each target takes the cheapest candidate within it, and the
; loop as written when none is.
;
; Prices are measured ratios that repeat to 0.6%: of two candidates within the
; target whose costs are closer than that, the one of smaller error is taken.
; With tf32 priced 0.4% above f16 it is taken for its slightly smaller error;
; priced 0.7% above, f16 is.

; F16: [poseidon] pin: matmul_body:M0=direct-dispatch cublas f16/f32{{$}}
; TF32: [poseidon] pin: matmul_body:M0=direct-dispatch cublas tf32/f32{{$}}
; NM8: [poseidon] pin: matmul_body:M0=ozaki-ii nm=8 wmma m16n16k16 s8/s32{{$}}
; NM9: [poseidon] pin: matmul_body:M0=ozaki-ii nm=9 wmma m16n16k16 s8/s32{{$}}
; NM10: [poseidon] pin: matmul_body:M0=ozaki-ii nm=10 wmma m16n16k16 s8/s32{{$}}
; NM11: [poseidon] pin: matmul_body:M0=ozaki-ii nm=11 wmma m16n16k16 s8/s32{{$}}
; NM12: [poseidon] pin: matmul_body:M0=ozaki-ii nm=12 wmma m16n16k16 s8/s32{{$}}
; NM13: [poseidon] pin: matmul_body:M0=ozaki-ii nm=13 wmma m16n16k16 s8/s32{{$}}
; NM14: [poseidon] pin: matmul_body:M0=ozaki-ii nm=14 wmma m16n16k16 s8/s32{{$}}
; NATIVE: [poseidon] pin: matmul_body:M0=native-dgemm cublas f64/f64{{$}}
; WMMA: [poseidon] pin: matmul_body:M0=wmma m16n16k16 f16/f32{{$}}
; TCEC2: [poseidon] pin: matmul_body:M0=tcec n=2 wmma m16n16k16 f16/f32{{$}}
; TCEC3: [poseidon] pin: matmul_body:M0=tcec n=3 wmma m16n16k16 f16/f32{{$}}
; WRITTEN-NOT: Applying solution for
; WRITTEN: [poseidon] no rewrite applied for preprocess_matmul_body; calling original body matmul_body (bit-transparent no-op){{$}}

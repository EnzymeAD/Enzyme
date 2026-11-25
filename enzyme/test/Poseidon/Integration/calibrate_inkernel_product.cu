// poseidon-calibrate's in-kernel arm: the rows of one product of constant
// extents with B contiguous along K, measured into an existing model whose
// class rows it already has, from profiles of the harness the arm records
// itself. Every class the arm proposes is measured (a run that leaves one out
// fails), every other row is kept, and a product the model already has is not
// measured again.
//
// REQUIRES: enzyme, cuda-runtime
// RUN: rm -rf %t && mkdir %t
// RUN: cp %gpu_cost_model %t/cm.csv
// RUN: %poseidon_calibrate --out %t/cm.csv --only inkernel --n 256 --reps 3 --product 256x32x256:bcolmajor \
// RUN:   | FileCheck --check-prefix=FIRST --implicit-check-not='did not' %s
// RUN: grep '^wmma_inkernel_rel,' %t/cm.csv | FileCheck --check-prefix=ROWS %s
// RUN: grep -v '_m256n32k256_bcolmajor,' %t/cm.csv | diff %gpu_cost_model -
// RUN: %poseidon_calibrate --out %t/cm.csv --only inkernel --n 256 --reps 3 --product 256x32x256:bcolmajor \
// RUN:   | FileCheck --check-prefix=AGAIN --implicit-check-not='  + ' %s

// FIRST: [calibrate] in-kernel raises at 256x256x256, B strided along K
// FIRST: = f16_f32 already calibrated
// FIRST: [calibrate] in-kernel raises at 256x32x256, B contiguous along K
// FIRST: + f16_f32_m256n32k256_bcolmajor {{[0-9.]+}} ms  rel {{[0-9.]+}}
// FIRST: + tcec_n2_tf32_f32_m256n32k256_bcolmajor {{[0-9.]+}} ms  rel {{[0-9.]+}}

// ROWS: wmma_inkernel_rel,f16_f32_m256n32k256_bcolmajor,{{[0-9.]+$}}
// ROWS: wmma_inkernel_rel,tcec_n2_tf32_f32_m256n32k256_bcolmajor,{{[0-9.]+$}}

// AGAIN: = f16_f32_m256n32k256_bcolmajor already calibrated
// AGAIN: [calibrate] in-kernel: nothing new to write

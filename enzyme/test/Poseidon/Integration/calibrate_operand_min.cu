// poseidon-calibrate's operand-min arm measures the floor of every in-kernel
// class it proposes and of every library call the model prices into an
// existing model, keeps every other row, and skips a row the model has.
// REQUIRES: enzyme, cuda-device
// UNSUPPORTED: tcec-cumpsgemm
// RUN: rm -rf %t && mkdir %t
// RUN: cp %gpu_cost_model %t/cm.csv
// RUN: %poseidon_calibrate --out %t/cm.csv --only operand-min --n 512 | FileCheck --check-prefix=FIRST --implicit-check-not='  ! ' %s
// RUN: %poseidon_calibrate --out %t/cm.csv --only operand-min --n 512 | FileCheck --check-prefix=AGAIN --implicit-check-not='  + ' %s
// RUN: grep -v _operand_min, %t/cm.csv | diff %gpu_cost_model -
// RUN: grep _operand_min, %t/cm.csv | FileCheck --check-prefix=ROWS %s

// FIRST:      poseidon-calibrate: operand floors of the in-kernel raises at 512x512x512, B strided along K
// FIRST-NEXT: poseidon-calibrate: 11 in-kernel class(es) proposed
// FIRST-NEXT:   + inkernel_operand_min,bf16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// FIRST:      poseidon-calibrate: measuring the operand floors of the library calls
// FIRST-NEXT:   + library_direct_operand_min,bf16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// FIRST-NEXT:   + library_direct_operand_min,f16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// FIRST-NEXT:   + library_direct_operand_min,tf32_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// FIRST-NEXT: poseidon-calibrate: cost model: {{.*}}/cm.csv

// AGAIN: = inkernel_operand_min,tf32_f32 already measured
// AGAIN: = library_direct_operand_min,tf32_f32 already measured

// ROWS:      inkernel_operand_min,bf16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,f16_f16,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,f16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,tcec_n2_bf16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,tcec_n2_f16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,tcec_n2_tf32_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,tcec_n3_f16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,tcec_n3_tf32_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,tcec_n5_f16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,tcec_n5_tf32_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: inkernel_operand_min,tf32_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: library_direct_operand_min,bf16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: library_direct_operand_min,f16_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NEXT: library_direct_operand_min,tf32_f32,{{[1-9][0-9.]*e-[0-9]+$}}
// ROWS-NOT:  _operand_min

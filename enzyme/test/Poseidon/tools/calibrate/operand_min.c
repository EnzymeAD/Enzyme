// poseidon-calibrate's operand-min arm with stand-in tools: on an NVIDIA GPU
// the in-kernel classes the pass proposes on the --n harness and the library
// calls the model prices, which the library calibrator is asked for by class;
// on an AMD GPU the library calls alone. Rows the model has are skipped; a class
// without a floor is named and measured alone when the command is run again. A
// model with such a row at 0 is refused.
// RUN: rm -rf %t && mkdir -p %t/nv/bin %t/amd/bin %t/amd/cm
// RUN: cp %poseidon_calibrate %t/nv/bin/ && cp %S/../../Inputs/fake-poseidon-clang.sh %t/nv/bin/poseidon-clang++
// RUN: printf '#!/bin/sh\nprintf "NVIDIA GeForce RTX 5090\\tsm_120\\n"\n' > %t/nv/bin/poseidon-calibrate-ops-cuda
// RUN: printf '#!/bin/sh\necho "$*" >> %t/nv/library.log\n[ "$1" = operand-min ] || exit 1\nshift 3\nfor c; do [ "$c" = "$FAKE_LIBRARY_LOST" ] || echo "library_direct_operand_min,$c,6.103515625e-05"; done\n' > %t/nv/bin/poseidon-calibrate-library-cuda
// RUN: chmod +x %t/nv/bin/poseidon-clang++ %t/nv/bin/poseidon-calibrate-ops-cuda %t/nv/bin/poseidon-calibrate-library-cuda
// RUN: cp %cost_models/cm_sm_120_synthetic_library.csv %t/nv/cm.csv
// RUN: %t/nv/bin/poseidon-calibrate --gpu 0 --out %t/nv/cm.csv --only operand-min --n 256 | FileCheck --check-prefix=FIRST %s
// RUN: grep -v _operand_min, %t/nv/cm.csv | diff %cost_models/cm_sm_120_synthetic_library.csv -
// RUN: grep _operand_min, %t/nv/cm.csv | FileCheck --check-prefix=ROWS %s
// RUN: %t/nv/bin/poseidon-calibrate --gpu 0 --out %t/nv/cm.csv --only operand-min --n 256 | FileCheck --check-prefix=AGAIN --implicit-check-not='  + ' %s
// RUN: cp %cost_models/cm_sm_120_synthetic_library.csv %t/nv/part.csv
// RUN: not env FAKE_CLANG_FAIL=M0_1 FAKE_LIBRARY_LOST=tf32_f32 %t/nv/bin/poseidon-calibrate --gpu 0 --out %t/nv/part.csv --only operand-min --n 256 > %t/nv/fail.out 2>&1
// RUN: FileCheck --check-prefix=FAIL %s < %t/nv/fail.out
// RUN: %t/nv/bin/poseidon-calibrate --gpu 0 --out %t/nv/part.csv --only operand-min --n 256 | FileCheck --check-prefix=REPEAT --implicit-check-not='  + ' %s
// RUN: FileCheck --check-prefix=ARGS %s < %t/nv/library.log
// RUN: cp %poseidon_calibrate %t/amd/bin/
// RUN: printf '#!/bin/sh\nprintf "AMD Instinct MI250X / MI250\\tgfx90a\\n"\n' > %t/amd/bin/poseidon-calibrate-ops-hip
// RUN: printf '#!/bin/sh\n[ "$*" = "operand-min 2048 30 bf16_f32 f16_f32" ] || exit 1\nprintf "library_direct_operand_min,f16_f32,6.103515625e-05\\nlibrary_direct_operand_min,bf16_f32,1.1754943508222875e-38\\n"\n' > %t/amd/bin/poseidon-calibrate-library-hip
// RUN: chmod +x %t/amd/bin/poseidon-calibrate-ops-hip %t/amd/bin/poseidon-calibrate-library-hip
// RUN: sed -e 's/^# native_arch=sm_120$/# native_arch=gfx90a/' -e '/^library_direct_rel,tf32_f32,/d' %cost_models/cm_sm_120_synthetic_library.csv > %t/amd/before.csv
// RUN: cp %t/amd/before.csv %t/amd/cm/cm.csv
// RUN: env PATH=%t/amd/bin %t/amd/bin/poseidon-calibrate --gpu 0 --only operand-min --out %t/amd/cm/cm.csv | FileCheck --check-prefix=AMD %s
// RUN: grep -v _operand_min, %t/amd/cm/cm.csv | diff %t/amd/before.csv -
// RUN: grep _operand_min, %t/amd/cm/cm.csv | FileCheck --check-prefix=AMD-ROWS %s
// RUN: env PATH=%t/amd/bin %t/amd/bin/poseidon-calibrate --gpu 0 --only operand-min --out %t/amd/cm/cm.csv | FileCheck --check-prefix=AMD-AGAIN %s
// RUN: not env PATH=%t/amd/bin %t/amd/bin/poseidon-calibrate --gpu 0 --check --out %t/amd/cm/cm.csv | FileCheck --check-prefix=AMD-CHECK %s
// RUN: sed 's/^library_direct_operand_min,f16_f32,.*/library_direct_operand_min,f16_f32,0/' %t/amd/cm/cm.csv > %t/amd/cm/zero.csv
// RUN: not env PATH=%t/amd/bin %t/amd/bin/poseidon-calibrate --gpu 0 --check --out %t/amd/cm/zero.csv 2>&1 | FileCheck --check-prefix=AMD-ZERO %s
// RUN: printf '#!/bin/sh\nprintf "library_direct_operand_min,f16_f32,6.103515625e-05\\n"\necho "poseidon-calibrate-library: library_direct_operand_min,bf16_f32: no operand floor, and no row" >&2\n' > %t/amd/bin/poseidon-calibrate-library-hip
// RUN: cp %t/amd/before.csv %t/amd/cm/part.csv
// RUN: not env PATH=%t/amd/bin %t/amd/bin/poseidon-calibrate --gpu 0 --only operand-min --out %t/amd/cm/part.csv > %t/amd/part.out 2>&1
// RUN: FileCheck --check-prefix=AMD-PART %s < %t/amd/part.out
// RUN: grep _operand_min, %t/amd/cm/part.csv | FileCheck --check-prefix=AMD-PART-ROWS %s

// FIRST:      poseidon-calibrate: operand-min workdir {{.*}}poseidon-operand-min-{{.*}}
// FIRST-NEXT: poseidon-calibrate: operand floors of the in-kernel raises at 256x256x256, B strided along K
// FIRST-NEXT: poseidon-calibrate: 3 in-kernel class(es) proposed
// FIRST-NEXT:   + inkernel_operand_min,bf16_f32,9.183549615799121e-41
// FIRST-NEXT:   + inkernel_operand_min,f16_f32,5.9604644775390625e-08
// FIRST-NEXT:   + inkernel_operand_min,tcec_n2_tf32_f32,1.1479437019748901e-41
// FIRST-NEXT: poseidon-calibrate: measuring the operand floors of the library calls
// FIRST-NEXT:   + library_direct_operand_min,bf16_f32,6.103515625e-05
// FIRST-NEXT:   + library_direct_operand_min,f16_f32,6.103515625e-05
// FIRST-NEXT:   + library_direct_operand_min,tf32_f32,6.103515625e-05
// FIRST-NEXT: poseidon-calibrate: cost model: {{.*}}/cm.csv

// ROWS:      inkernel_operand_min,bf16_f32,9.183549615799121e-41
// ROWS-NEXT: inkernel_operand_min,f16_f32,5.9604644775390625e-08
// ROWS-NEXT: inkernel_operand_min,tcec_n2_tf32_f32,1.1479437019748901e-41
// ROWS-NEXT: library_direct_operand_min,bf16_f32,6.103515625e-05
// ROWS-NEXT: library_direct_operand_min,f16_f32,6.103515625e-05
// ROWS-NEXT: library_direct_operand_min,tf32_f32,6.103515625e-05
// ROWS-NOT:  _operand_min

// AGAIN:      poseidon-calibrate: 3 in-kernel class(es) proposed
// AGAIN-NEXT:   = inkernel_operand_min,bf16_f32 already measured
// AGAIN-NEXT:   = inkernel_operand_min,f16_f32 already measured
// AGAIN-NEXT:   = inkernel_operand_min,tcec_n2_tf32_f32 already measured
// AGAIN-NEXT:   = library_direct_operand_min,bf16_f32 already measured
// AGAIN-NEXT:   = library_direct_operand_min,f16_f32 already measured
// AGAIN-NEXT:   = library_direct_operand_min,tf32_f32 already measured
// AGAIN-NEXT: poseidon-calibrate: cost model: {{.*}}/cm.csv

// FAIL:      poseidon-calibrate: 3 in-kernel class(es) proposed
// FAIL-NEXT: initialization error
// FAIL-NEXT:   ! inkernel_operand_min,bf16_f32: no floor established
// FAIL-NEXT:   + inkernel_operand_min,f16_f32,5.9604644775390625e-08
// FAIL-NEXT:   + inkernel_operand_min,tcec_n2_tf32_f32,1.1479437019748901e-41
// FAIL-NEXT: poseidon-calibrate: measuring the operand floors of the library calls
// FAIL-NEXT:   + library_direct_operand_min,bf16_f32,6.103515625e-05
// FAIL-NEXT:   + library_direct_operand_min,f16_f32,6.103515625e-05
// FAIL-NEXT:   ! library_direct_operand_min,tf32_f32: no floor established
// FAIL-NEXT: poseidon-calibrate: cost model: {{.*}}/part.csv
// FAIL-NEXT: poseidon-calibrate: 2 row(s) not measured, which the same command measures alone when run again: inkernel_operand_min,bf16_f32, library_direct_operand_min,tf32_f32{{$}}
// FAIL-NOT:  {{.}}

// REPEAT:      poseidon-calibrate: 3 in-kernel class(es) proposed
// REPEAT-NEXT:   + inkernel_operand_min,bf16_f32,9.183549615799121e-41
// REPEAT-NEXT:   = inkernel_operand_min,f16_f32 already measured
// REPEAT-NEXT:   = inkernel_operand_min,tcec_n2_tf32_f32 already measured
// REPEAT-NEXT:   = library_direct_operand_min,bf16_f32 already measured
// REPEAT-NEXT:   = library_direct_operand_min,f16_f32 already measured
// REPEAT-NEXT: poseidon-calibrate: measuring the operand floors of the library calls
// REPEAT-NEXT:   + library_direct_operand_min,tf32_f32,6.103515625e-05

// ARGS:      operand-min 256 30 bf16_f32 f16_f32 tf32_f32
// ARGS-NEXT: operand-min 256 30 bf16_f32 f16_f32 tf32_f32
// ARGS-NEXT: operand-min 256 30 tf32_f32
// ARGS-NOT:  {{.}}

// AMD:      poseidon-calibrate: device MI250X (gfx90a, id 0)
// AMD-NEXT: poseidon-calibrate: operand-min: tensor-core raises inside a kernel are NVPTX-only; their rows stay absent
// AMD-NEXT: poseidon-calibrate: measuring the operand floors of the library calls
// AMD-NEXT:   + library_direct_operand_min,bf16_f32,1.1754943508222875e-38
// AMD-NEXT:   + library_direct_operand_min,f16_f32,6.103515625e-05
// AMD-NEXT: poseidon-calibrate: cost model: {{.*}}/cm/cm.csv

// AMD-ROWS:      library_direct_operand_min,bf16_f32,1.1754943508222875e-38
// AMD-ROWS-NEXT: library_direct_operand_min,f16_f32,6.103515625e-05
// AMD-ROWS-NOT:  _operand_min

// AMD-AGAIN:      poseidon-calibrate: operand-min: tensor-core raises inside a kernel are NVPTX-only; their rows stay absent
// AMD-AGAIN-NEXT:   = library_direct_operand_min,bf16_f32 already measured
// AMD-AGAIN-NEXT:   = library_direct_operand_min,f16_f32 already measured
// AMD-AGAIN-NEXT: poseidon-calibrate: cost model: {{.*}}/cm/cm.csv

// AMD-CHECK:      have        library_direct_operand_min  (smallest operand of the reduced-precision library call)
// AMD-CHECK-NEXT: n/a         library_tcec_operand_min  (smallest operand of the error-corrected library call: none on gfx90a)
// AMD-CHECK-NEXT: n/a         inkernel_operand_min  (smallest operand of the in-kernel tensor-core raises: none on gfx90a)

// AMD-ZERO: cost model {{.*}}/cm/zero.csv line {{[0-9]+}} has the row 'library_direct_operand_min,f16_f32' at 0, which is no operand magnitude; poseidon-calibrate --gpu <id> --only operand-min measures it

// AMD-PART:      poseidon-calibrate: measuring the operand floors of the library calls
// AMD-PART-NEXT: poseidon-calibrate-library: library_direct_operand_min,bf16_f32: no operand floor, and no row
// AMD-PART-NEXT:   ! library_direct_operand_min,bf16_f32: no floor established
// AMD-PART-NEXT:   + library_direct_operand_min,f16_f32,6.103515625e-05
// AMD-PART-NEXT: poseidon-calibrate: cost model: {{.*}}/cm/part.csv
// AMD-PART-NEXT: poseidon-calibrate: 1 row(s) not measured, which the same command measures alone when run again: library_direct_operand_min,bf16_f32{{$}}

// AMD-PART-ROWS:     library_direct_operand_min,f16_f32,6.103515625e-05
// AMD-PART-ROWS-NOT: _operand_min

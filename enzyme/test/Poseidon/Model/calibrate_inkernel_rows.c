// poseidon-calibrate's in-kernel arm with stand-in tools (poseidon-clang++ logs
// its arguments): class rows on the --n harness, a --product with a layout at
// its extents, profiled at most 128; rows the model has are skipped; a class
// that fails is named and measured alone when the command is run again.
// RUN: rm -rf %t && mkdir -p %t/bin
// RUN: cp %poseidon_calibrate %t/bin/ && cp %S/../Inputs/fake-poseidon-clang.sh %t/bin/poseidon-clang++
// RUN: printf '#!/bin/sh\nprintf "NVIDIA GeForce RTX 5090\\tsm_120\\n"\n' > %t/bin/poseidon-calibrate-ops-cuda
// RUN: chmod +x %t/bin/poseidon-clang++ %t/bin/poseidon-calibrate-ops-cuda
// RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/cm.csv
// RUN: env FAKE_CLANG_LOG=%t/clang.log %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --n 256 --product 2048x112x2048:bcolmajor,64x32x16 | FileCheck --check-prefix=FIRST %s
// RUN: FileCheck --check-prefix=ARGS --implicit-check-not=-poseidon-print %s < %t/clang.log
// RUN: grep '^inkernel_rel,' %t/cm.csv | FileCheck --check-prefix=ROWS %s
// RUN: not grep 'm64n32k16' %t/cm.csv
// RUN: %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --n 256 --product 2048x112x2048:bcolmajor | FileCheck --check-prefix=AGAIN --implicit-check-not='  + ' %s
// RUN: not env FAKE_CLANG_FAIL=M0_2 %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --n 256 --product 64x16x32:browmajor > %t/fail.out 2>&1
// RUN: FileCheck --check-prefix=FAIL %s < %t/fail.out
// RUN: grep -c '_m64n16k32_browmajor,' %t/cm.csv | FileCheck --check-prefix=TWO %s
// RUN: %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --n 256 --product 64x16x32:browmajor | FileCheck --check-prefix=REPEAT --implicit-check-not='  + ' %s
// RUN: not %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --product 2048x112:bcolmajor 2>&1 | FileCheck --check-prefix=BAD %s
// RUN: not %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --product 64x64x64:colmajor 2>&1 | FileCheck --check-prefix=BAD %s

// FIRST: poseidon-calibrate: in-kernel raises at 256x256x256, B strided along K
// FIRST: poseidon-calibrate: 3 in-kernel class(es) proposed
// FIRST-NEXT: + bf16_f32 0.5000 ms  rel 0.250000
// FIRST-NEXT: = f16_f32 already calibrated
// FIRST-NEXT: + tcec_n2_tf32_f32 0.5000 ms  rel 0.250000
// FIRST-NEXT: poseidon-calibrate: in-kernel raises at 2048x112x2048, B contiguous along K
// FIRST: + bf16_f32_m2048n112k2048_bcolmajor 0.5000 ms  rel 0.250000
// FIRST-NEXT: + f16_f32_m2048n112k2048_bcolmajor 0.5000 ms  rel 0.250000
// FIRST-NEXT: + tcec_n2_tf32_f32_m2048n112k2048_bcolmajor 0.5000 ms  rel 0.250000
// FIRST-NOT: in-kernel raises at

// ARGS: -DCAL_M=128 -DCAL_N=128 -DCAL_K=128 -DCAL_B_COL_MAJOR=0 -poseidon-profile-generate=[[CLASS:[^ ]*]]/profile {{.*}}inkernel.cu
// ARGS-NEXT: -DCAL_M=256 -DCAL_N=256 -DCAL_K=256 -DCAL_B_COL_MAJOR=0 {{.*}}inkernel.cu -o [[CLASS]]/base.exe
// ARGS-NEXT: -DCAL_M=256 -DCAL_N=256 -DCAL_K=256 -DCAL_B_COL_MAJOR=0 -poseidon-profile-use=[[CLASS]]/profile {{.*}}-poseidon-pin=M0_0 {{.*}} -o /dev/null
// ARGS: -DCAL_M=128 -DCAL_N=112 -DCAL_K=128 -DCAL_B_COL_MAJOR=1 -poseidon-profile-generate=[[PROD:[^ ]*]]/profile {{.*}}inkernel.cu
// ARGS-NEXT: -DCAL_M=2048 -DCAL_N=112 -DCAL_K=2048 -DCAL_B_COL_MAJOR=1 {{.*}}inkernel.cu -o [[PROD]]/base.exe
// ARGS-NEXT: -DCAL_M=2048 -DCAL_N=112 -DCAL_K=2048 -DCAL_B_COL_MAJOR=1 -poseidon-profile-use=[[PROD]]/profile

// ROWS: inkernel_rel,f16_f32,0.0625
// ROWS-NEXT: inkernel_rel,tcec_n2_f16_f32,0.078125
// ROWS-NEXT: inkernel_rel,tcec_n3_f16_f32,0.09375
// ROWS-NEXT: inkernel_rel,bf16_f32,0.250000
// ROWS-NEXT: inkernel_rel,tcec_n2_tf32_f32,0.250000
// ROWS-NEXT: inkernel_rel,bf16_f32_m2048n112k2048_bcolmajor,0.250000
// ROWS-NEXT: inkernel_rel,f16_f32_m2048n112k2048_bcolmajor,0.250000
// ROWS-NEXT: inkernel_rel,tcec_n2_tf32_f32_m2048n112k2048_bcolmajor,0.250000
// ROWS-NOT: inkernel_rel

// AGAIN: = tcec_n2_tf32_f32 already calibrated
// AGAIN: = tcec_n2_tf32_f32_m2048n112k2048_bcolmajor already calibrated
// AGAIN: poseidon-calibrate: in-kernel: nothing new to write

// FAIL: poseidon-calibrate: in-kernel raises at 64x16x32, B strided along K
// FAIL: initialization error
// FAIL-NEXT: ! tcec_n2_tf32_f32_m64n16k32_browmajor did not run
// FAIL: poseidon-calibrate: cost model: {{.*}}cm.csv
// FAIL-NEXT: poseidon-calibrate: 1 row(s) not measured, which the same command measures alone when run again: inkernel_rel,tcec_n2_tf32_f32_m64n16k32_browmajor{{$}}
// FAIL-NOT: {{.}}

// TWO: 2

// REPEAT: = bf16_f32_m64n16k32_browmajor already calibrated
// REPEAT-NEXT: = f16_f32_m64n16k32_browmajor already calibrated
// REPEAT-NEXT: + tcec_n2_tf32_f32_m64n16k32_browmajor 0.5000 ms  rel 0.250000

// BAD: poseidon-calibrate: --product: '{{.*}}' is not a product MxNxK or MxNxK:<layout>, <layout> browmajor or bcolmajor

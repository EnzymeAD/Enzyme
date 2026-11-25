// poseidon-calibrate's in-kernel arm, with stand-ins for the GPU tools and for
// poseidon-clang++, which logs its arguments: the class rows are measured on
// the square harness of --n with B strided along K, and a --product that names
// a layout at its extents with B in that layout, each harness profiled at
// extents of at most 128 in the same layout. A row the model has is not
// measured again, and a product without a layout has no in-kernel rows.
//
// RUN: rm -rf %t && mkdir -p %t/bin
// RUN: cp %poseidon_calibrate %t/bin/ && cp %S/../Inputs/fake-poseidon-clang.sh %t/bin/poseidon-clang++
// RUN: printf '#!/bin/sh\nprintf "NVIDIA GeForce RTX 5090\\tsm_120\\n"\n' > %t/bin/poseidon-microbm
// RUN: chmod +x %t/bin/poseidon-clang++ %t/bin/poseidon-microbm
// RUN: cp %cost_models/cm_sm_120_synthetic.csv %t/cm.csv
// RUN: env FAKE_CLANG_LOG=%t/clang.log %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --n 256 --product 2048x112x2048:bcolmajor,64x32x16 \
// RUN:   | FileCheck --check-prefix=FIRST %s
// RUN: FileCheck --check-prefix=ARGS %s < %t/clang.log
// RUN: grep '^wmma_inkernel_rel,' %t/cm.csv | FileCheck --check-prefix=ROWS %s
// RUN: not grep 'm64n32k16' %t/cm.csv
// RUN: %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --n 256 --product 2048x112x2048:bcolmajor \
// RUN:   | FileCheck --check-prefix=AGAIN --implicit-check-not='  + ' %s
// RUN: not %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --product 2048x112:bcolmajor 2>&1 | FileCheck --check-prefix=BAD %s
// RUN: not %t/bin/poseidon-calibrate --gpu 0 --out %t/cm.csv --only inkernel --product 64x64x64:colmajor 2>&1 | FileCheck --check-prefix=BAD %s

// FIRST: [calibrate] in-kernel raises at 256x256x256, B strided along K
// FIRST: [calibrate] 3 in-kernel class(es) proposed
// FIRST-NEXT: + bf16_f32 0.5000 ms  rel 0.250000
// FIRST-NEXT: = f16_f32 already calibrated
// FIRST-NEXT: + tcec_n2_tf32_f32 0.5000 ms  rel 0.250000
// FIRST-NEXT: [calibrate] in-kernel raises at 2048x112x2048, B contiguous along K
// FIRST: + bf16_f32_m2048n112k2048_bcolmajor 0.5000 ms  rel 0.250000
// FIRST-NEXT: + f16_f32_m2048n112k2048_bcolmajor 0.5000 ms  rel 0.250000
// FIRST-NEXT: + tcec_n2_tf32_f32_m2048n112k2048_bcolmajor 0.5000 ms  rel 0.250000
// FIRST-NOT: in-kernel raises at

// ARGS: -DCAL_M=128 -DCAL_N=128 -DCAL_K=128 -DCAL_B_COL_MAJOR=0 -poseidon-profile-generate=[[CLASS:[^ ]*]]/profile {{.*}}inkernel_calibrate.cu
// ARGS-NEXT: -DCAL_M=256 -DCAL_N=256 -DCAL_K=256 -DCAL_B_COL_MAJOR=0 {{.*}}inkernel_calibrate.cu -o [[CLASS]]/base.exe
// ARGS-NEXT: -DCAL_M=256 -DCAL_N=256 -DCAL_K=256 -DCAL_B_COL_MAJOR=0 -poseidon-profile-use=[[CLASS]]/profile {{.*}}-poseidon-apply-rewrites=M0_0 {{.*}} -o /dev/null
// ARGS: -DCAL_M=128 -DCAL_N=112 -DCAL_K=128 -DCAL_B_COL_MAJOR=1 -poseidon-profile-generate=[[PROD:[^ ]*]]/profile {{.*}}inkernel_calibrate.cu
// ARGS-NEXT: -DCAL_M=2048 -DCAL_N=112 -DCAL_K=2048 -DCAL_B_COL_MAJOR=1 {{.*}}inkernel_calibrate.cu -o [[PROD]]/base.exe
// ARGS-NEXT: -DCAL_M=2048 -DCAL_N=112 -DCAL_K=2048 -DCAL_B_COL_MAJOR=1 -poseidon-profile-use=[[PROD]]/profile

// ROWS: wmma_inkernel_rel,f16_f32,0.0625
// ROWS-NEXT: wmma_inkernel_rel,tcec_n2_f16_f32,0.078125
// ROWS-NEXT: wmma_inkernel_rel,tcec_n3_f16_f32,0.09375
// ROWS-NEXT: wmma_inkernel_rel,bf16_f32,0.250000
// ROWS-NEXT: wmma_inkernel_rel,tcec_n2_tf32_f32,0.250000
// ROWS-NEXT: wmma_inkernel_rel,bf16_f32_m2048n112k2048_bcolmajor,0.250000
// ROWS-NEXT: wmma_inkernel_rel,f16_f32_m2048n112k2048_bcolmajor,0.250000
// ROWS-NEXT: wmma_inkernel_rel,tcec_n2_tf32_f32_m2048n112k2048_bcolmajor,0.250000
// ROWS-NOT: wmma_inkernel_rel

// AGAIN: = tcec_n2_tf32_f32 already calibrated
// AGAIN: = tcec_n2_tf32_f32_m2048n112k2048_bcolmajor already calibrated
// AGAIN: [calibrate] in-kernel: nothing new to write

// BAD: poseidon-calibrate: --product: '{{.*}}' is not a product MxNxK or MxNxK:<layout>, <layout> browmajor or bcolmajor

// poseidon-calibrate's ozaki-rect arm: one ozaki_dispatch_rect_rel row per
// modulus count and (M, K) shape, and only the native DGEMM row of an MxNxK
// shape, measured into an existing model. A shape the model already has is not
// measured again, --force measures it again, and the rows of the shapes a run
// does not name are kept, as is every other row.
//
// REQUIRES: cuda-runtime
// RUN: rm -rf %t && mkdir -p %t
// RUN: cp %gpu_cost_model %t/cm.csv
// RUN: %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 125x648,216x1029 --n 512 --reps 3 | FileCheck --check-prefix=FIRST %s
// RUN: %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 125x648 --n 512 --reps 3 | FileCheck --check-prefix=AGAIN %s
// RUN: %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 216x1029 --n 512 --reps 3 --force | FileCheck --check-prefix=FORCE %s
// RUN: %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 125x16x648,64x16x512 --n 512 --reps 3 | FileCheck --check-prefix=FULL %s
// RUN: grep -c '^ozaki_dispatch_rect_rel,nm[0-9]*_m125k648,' %t/cm.csv | FileCheck --check-prefix=SEVEN %s
// RUN: grep -c '^ozaki_dispatch_rect_rel,nm[0-9]*_m216k1029,' %t/cm.csv | FileCheck --check-prefix=SEVEN %s
// RUN: grep -c '^ozaki_dispatch_rect_rel,dgemm_' %t/cm.csv | FileCheck --check-prefix=TWO %s
// RUN: not grep '_m64k512,' %t/cm.csv
// RUN: grep -v '^ozaki_dispatch_rect_rel,' %t/cm.csv > %t/rest.csv
// RUN: diff %gpu_cost_model %t/rest.csv
// RUN: not %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 12x 2>&1 | FileCheck --check-prefix=BAD %s

// FIRST: [calibrate] measuring ozaki-rect host dispatch
// FIRST: ozaki_dispatch_rect_rel,nm8_m125k648,
// FIRST: ozaki_dispatch_rect_rel,nm14_m125k648,
// FIRST: ozaki_dispatch_rect_rel,nm8_m216k1029,
// FIRST: ozaki_dispatch_rect_rel,nm14_m216k1029,

// AGAIN: = 125x648 already measured
// AGAIN-NOT: measuring

// FORCE: [calibrate] measuring ozaki-rect host dispatch

// FULL-NOT: already measured
// FULL: [calibrate] measuring ozaki-rect host dispatch
// FULL: ozaki_dispatch_rect_rel,dgemm_m125n16k648,
// FULL: ozaki_dispatch_rect_rel,dgemm_m64n16k512,

// SEVEN: 7

// TWO: 2

// BAD: --ozaki-rect: '12x' is not a shape MxK or MxNxK

// poseidon-calibrate's ozaki-rect arm: rows per modulus count and shape (and
// the native row of a --product), a shape it has skipped unless --force, other
// rows kept.
// REQUIRES: cuda-device
// RUN: rm -rf %t && mkdir %t
// RUN: cp %gpu_cost_model %t/cm.csv
// RUN: %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 125x648,216x1029 --n 512 --reps 3 | FileCheck --check-prefix=FIRST %s
// RUN: %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 125x648 --n 512 --reps 3 | FileCheck --check-prefix=AGAIN %s
// RUN: %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 216x1029 --n 512 --reps 3 --force | FileCheck --check-prefix=FORCE %s
// RUN: %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --product 125x16x648,64x16x512 --n 512 --reps 3 | FileCheck --check-prefix=FULL %s
// RUN: grep -c '^library_ozaki_rect_rel,nm[0-9]*_m125k648,' %t/cm.csv | FileCheck --check-prefix=SEVEN %s
// RUN: grep -c '^library_ozaki_rect_rel,nm[0-9]*_m216k1029,' %t/cm.csv | FileCheck --check-prefix=SEVEN %s
// RUN: grep -c '^library_ozaki_rect_rel,nm[0-9]*_m125n16k648,' %t/cm.csv | FileCheck --check-prefix=SEVEN %s
// RUN: grep -c '^library_ozaki_rect_rel,dgemm_' %t/cm.csv | FileCheck --check-prefix=TWO %s
// RUN: not grep '_m64k512,' %t/cm.csv
// RUN: grep -v '^library_ozaki_rect_rel,' %t/cm.csv > %t/rest.csv
// RUN: diff %gpu_cost_model %t/rest.csv
// RUN: not %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 12x 2>&1 | FileCheck --check-prefix=BAD %s
// RUN: not %poseidon_calibrate --out %t/cm.csv --only ozaki-rect --ozaki-rect 125x16x648 2>&1 | FileCheck --check-prefix=BAD %s

// FIRST: poseidon-calibrate: measuring ozaki-rect library call
// FIRST: library_ozaki_rect_rel,nm8_m125k648,
// FIRST: library_ozaki_rect_rel,nm14_m125k648,
// FIRST: library_ozaki_rect_rel,nm8_m216k1029,
// FIRST: library_ozaki_rect_rel,nm14_m216k1029,

// AGAIN: = 125x648 already measured
// AGAIN-NOT: measuring

// FORCE: poseidon-calibrate: measuring ozaki-rect library call

// FULL-NOT: already measured
// FULL: poseidon-calibrate: measuring ozaki-rect library call
// FULL: library_ozaki_rect_rel,nm8_m125n16k648,
// FULL: library_ozaki_rect_rel,nm14_m125n16k648,
// FULL: library_ozaki_rect_rel,dgemm_m125n16k648,
// FULL: library_ozaki_rect_rel,dgemm_m64n16k512,

// SEVEN: 7

// TWO: 2

// BAD: --ozaki-rect: '{{12x|125x16x648}}' is not a shape MxK; the rows of a product of constant extents are measured with --product MxNxK

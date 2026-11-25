// RUN: rm -rf %t && mkdir %t
// RUN: %opt %S/../../Inputs/expansion_match/module.ll %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-expansion-match%shlibext -passes=expansion-match-raw -S -o %t/raw.ll
// RUN: %clang++ -O2 -c %t/raw.ll -o %t/raw.o
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off %S/../../Inputs/expansion_match/reference_harness.cpp %t/raw.o -o %t/run
// RUN: %t/run | FileCheck %s
// RUN: %opt %S/../../Inputs/expansion_match/module_nvptx.ll %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-expansion-match%shlibext -passes=expansion-match-raw -S | FileCheck --check-prefix=NVPTX --implicit-check-not=nvvm.sqrt.approx %s

// The FP32 expansions the materializer emits equal, run on the host, the
// external routines they reproduce, bit for bit with NaN payload and sign
// excepted, on 80,000 operands per routine in eight classes
// (Inputs/expansion_match/gen.h): three and four components against mX_real's
// Sloppy types (Inputs/expansion_match/mx_real_subset.h), squares and the
// square root's special cases included; two components against MultiFloats.jl's
// addition, subtraction, multiplication and square root
// (Inputs/expansion_match/multifloats_subset.h) and mX_real's division. Where
// both agree on a two-component special value the emitted code does too: the
// square root of a zero is the zero, and an exact quotient by a negative
// divisor has a +0 low component.

// CHECK: n=3 add: 80000 cases, 0 differ
// CHECK-NEXT: n=3 sub: 80000 cases, 0 differ
// CHECK-NEXT: n=3 mul: 80000 cases, 0 differ
// CHECK-NEXT: n=3 a*a: 80000 cases, 0 differ
// CHECK-NEXT: n=3 div: 80000 cases, 0 differ
// CHECK-NEXT: n=3 sqrt: 80000 cases, 0 differ
// CHECK-NEXT: n=3 neg: 80000 cases, 0 differ
// CHECK-NEXT: n=3 fma: 80000 cases, 0 differ
// CHECK-NEXT: n=3 fma(a,a,c): 80000 cases, 0 differ
// CHECK-NEXT: n=4 add: 80000 cases, 0 differ
// CHECK-NEXT: n=4 sub: 80000 cases, 0 differ
// CHECK-NEXT: n=4 mul: 80000 cases, 0 differ
// CHECK-NEXT: n=4 a*a: 80000 cases, 0 differ
// CHECK-NEXT: n=4 div: 80000 cases, 0 differ
// CHECK-NEXT: n=4 sqrt: 80000 cases, 0 differ
// CHECK-NEXT: n=4 neg: 80000 cases, 0 differ
// CHECK-NEXT: n=4 fma: 80000 cases, 0 differ
// CHECK-NEXT: n=4 fma(a,a,c): 80000 cases, 0 differ
// CHECK-NEXT: n=2 add: 80000 cases, 0 differ
// CHECK-NEXT: n=2 sub: 80000 cases, 0 differ
// CHECK-NEXT: n=2 mul: 80000 cases, 0 differ
// CHECK-NEXT: n=2 sqrt: 80000 cases, 0 differ
// CHECK-NEXT: n=2 div: 80000 cases, 0 differ
// CHECK-NEXT: n=2 sqrt(0x0p+0, 0x0p+0) = 0x0p+0 0x0p+0
// CHECK-NEXT: n=2 sqrt(-0x0p+0, 0x0p+0) = -0x0p+0 0x0p+0
// CHECK-NEXT: n=2 sqrt(0x0p+0, -0x0p+0) = 0x0p+0 -0x0p+0
// CHECK-NEXT: n=2 sqrt(-0x0p+0, -0x0p+0) = -0x0p+0 -0x0p+0
// CHECK-NEXT: n=2 (-10.5, 0) / (-14, 0) = 0x1.8p-1 0x0p+0

// The square roots start from the correctly rounded FP32 square root on
// NVPTX too, not from the approximate one, so the device computes what the
// host does.
// NVPTX-LABEL: define void @p2_sqrt(
// NVPTX: call float @llvm.sqrt.f32(
// NVPTX-LABEL: define void @p3_sqrt(
// NVPTX: call float @llvm.sqrt.f32(
// NVPTX-LABEL: define void @p4_sqrt(
// NVPTX: call float @llvm.sqrt.f32(

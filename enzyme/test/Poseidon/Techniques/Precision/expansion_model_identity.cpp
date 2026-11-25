// RUN: rm -rf %t && mkdir %t
// RUN: env EXPANSION_MATCH_OUT=%t %opt %S/../../Inputs/expansion_match/module.ll %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-expansion-match%shlibext -passes=expansion-match-model -S -o %t/model.ll
// RUN: mkdir -p %t/again && env EXPANSION_MATCH_OUT=%t/again %opt %S/../../Inputs/expansion_match/module.ll %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-expansion-match%shlibext -passes=expansion-match-model -S -o %t/again/model.ll
// RUN: diff %t/model.ll %t/again/model.ll
// RUN: %clang++ -O0 -c %t/model.ll -o %t/model.o
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -I%t %S/../../Inputs/expansion_match/model_harness.cpp %t/model.o -o %t/run
// RUN: %t/run %t | FileCheck %s

// The accuracy model's emulation of the FP32 expansions computes what the
// materializer emits, bit for bit with NaN payload and sign excepted, for two,
// three and four components: every routine on 80,000 raw operands in eight
// classes (Inputs/expansion_match/gen.h), the boundary conversions, and,
// through getFPValues with a precision candidate, FP64 functions of one
// operation and chains of three to six operations on 4,000 points each. Within
// one change the model keeps an operation's components for later operations,
// as applyExpansion does; across changes and around an FP64 operation both
// split again; an FP32 function is converted and restored in FP32. The model is
// the plugin's own, as built, and two runs emit the same model.ll. The emitted
// code is compiled unoptimized because up to LLVM 22 InstCombine turns the -0
// low component of some squares (fma(x, x, y)) into +0.

// CHECK: n=2 FP32 chains: 16000 cases, 0 differ
// CHECK-NEXT: n=2 a*a: 80000 cases, 0 differ
// CHECK-NEXT: n=2 add: 80000 cases, 0 differ
// CHECK-NEXT: n=2 chains, one change: 16000 cases, 0 differ
// CHECK-NEXT: n=2 chains, one operation left in FP64: 16000 cases, 0 differ
// CHECK-NEXT: n=2 chains, two changes: 16000 cases, 0 differ
// CHECK-NEXT: n=2 div: 80000 cases, 0 differ
// CHECK-NEXT: n=2 fma: 80000 cases, 0 differ
// CHECK-NEXT: n=2 fma(a,a,c): 80000 cases, 0 differ
// CHECK-NEXT: n=2 from_f64: 80000 cases, 0 differ
// CHECK-NEXT: n=2 mul: 80000 cases, 0 differ
// CHECK-NEXT: n=2 neg: 80000 cases, 0 differ
// CHECK-NEXT: n=2 single operations: 40000 cases, 0 differ
// CHECK-NEXT: n=2 sqrt: 80000 cases, 0 differ
// CHECK-NEXT: n=2 sub: 80000 cases, 0 differ
// CHECK-NEXT: n=2 to_f32: 80000 cases, 0 differ
// CHECK-NEXT: n=2 to_f64: 80000 cases, 0 differ
// CHECK-NEXT: n=3 FP32 chains: 16000 cases, 0 differ
// CHECK-NEXT: n=3 a*a: 80000 cases, 0 differ
// CHECK-NEXT: n=3 add: 80000 cases, 0 differ
// CHECK-NEXT: n=3 chains, one change: 16000 cases, 0 differ
// CHECK-NEXT: n=3 chains, one operation left in FP64: 16000 cases, 0 differ
// CHECK-NEXT: n=3 chains, two changes: 16000 cases, 0 differ
// CHECK-NEXT: n=3 div: 80000 cases, 0 differ
// CHECK-NEXT: n=3 fma: 80000 cases, 0 differ
// CHECK-NEXT: n=3 fma(a,a,c): 80000 cases, 0 differ
// CHECK-NEXT: n=3 from_f64: 80000 cases, 0 differ
// CHECK-NEXT: n=3 mul: 80000 cases, 0 differ
// CHECK-NEXT: n=3 neg: 80000 cases, 0 differ
// CHECK-NEXT: n=3 single operations: 40000 cases, 0 differ
// CHECK-NEXT: n=3 sqrt: 80000 cases, 0 differ
// CHECK-NEXT: n=3 sub: 80000 cases, 0 differ
// CHECK-NEXT: n=3 to_f32: 80000 cases, 0 differ
// CHECK-NEXT: n=3 to_f64: 80000 cases, 0 differ
// CHECK-NEXT: n=4 FP32 chains: 16000 cases, 0 differ
// CHECK-NEXT: n=4 a*a: 80000 cases, 0 differ
// CHECK-NEXT: n=4 add: 80000 cases, 0 differ
// CHECK-NEXT: n=4 chains, one change: 16000 cases, 0 differ
// CHECK-NEXT: n=4 chains, one operation left in FP64: 16000 cases, 0 differ
// CHECK-NEXT: n=4 chains, two changes: 16000 cases, 0 differ
// CHECK-NEXT: n=4 div: 80000 cases, 0 differ
// CHECK-NEXT: n=4 fma: 80000 cases, 0 differ
// CHECK-NEXT: n=4 fma(a,a,c): 80000 cases, 0 differ
// CHECK-NEXT: n=4 from_f64: 80000 cases, 0 differ
// CHECK-NEXT: n=4 mul: 80000 cases, 0 differ
// CHECK-NEXT: n=4 neg: 80000 cases, 0 differ
// CHECK-NEXT: n=4 single operations: 40000 cases, 0 differ
// CHECK-NEXT: n=4 sqrt: 80000 cases, 0 differ
// CHECK-NEXT: n=4 sub: 80000 cases, 0 differ
// CHECK-NEXT: n=4 to_f32: 80000 cases, 0 differ
// CHECK-NEXT: n=4 to_f64: 80000 cases, 0 differ

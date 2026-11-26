// DEFINE: %{match} = %opt %loadPoseidon -load-pass-plugin=%poseidon_test_programs/poseidon-test-expansion-match%shlibext
// RUN: rm -rf %t && mkdir -p %t/again
// RUN: %{match} -passes=expansion-match-raw -S -o %t/raw.ll < /dev/null
// RUN: %clang++ -O2 -c %t/raw.ll -o %t/raw.o
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off %S/../../Inputs/expansion_match/reference_harness.cpp %t/raw.o -o %t/ref
// RUN: %t/ref | FileCheck --check-prefix=REF %s
// RUN: echo 'target triple = "nvptx64-nvidia-cuda"' | %{match} -passes=expansion-match-raw -S | FileCheck --check-prefix=NVPTX --implicit-check-not=nvvm.sqrt.approx %s
// RUN: env EXPANSION_MATCH_OUT=%t %{match} -passes=expansion-match-model -S -o %t/model.ll < /dev/null
// RUN: env EXPANSION_MATCH_OUT=%t/again %{match} -passes=expansion-match-model -S -o %t/again/model.ll < /dev/null
// RUN: diff %t/model.ll %t/again/model.ll
// RUN: %clang++ -O0 -c %t/model.ll -o %t/model.o
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off -I%t %S/../../Inputs/expansion_match/model_harness.cpp %t/model.o -o %t/model
// RUN: %t/model %t | FileCheck --check-prefix=MODEL %s

// The emitted FP32 expansions, run on the host, equal the external routines
// they reproduce bit for bit (NaN payload and sign excepted) on 80,000 operands
// per routine (Inputs/expansion_match): n = 3, 4 and the n = 2 division against
// mX_real, n = 2 add, sub, mul, sqrt against MultiFloats.jl; and the n = 2
// special values both agree on (sqrt of a zero, an exact negative quotient).
// On NVPTX the square roots also start from the correctly rounded FP32 root.

// REF: n=3 add: 80000 cases, 0 differ
// REF-NEXT: n=3 sub: 80000 cases, 0 differ
// REF-NEXT: n=3 mul: 80000 cases, 0 differ
// REF-NEXT: n=3 a*a: 80000 cases, 0 differ
// REF-NEXT: n=3 div: 80000 cases, 0 differ
// REF-NEXT: n=3 sqrt: 80000 cases, 0 differ
// REF-NEXT: n=3 neg: 80000 cases, 0 differ
// REF-NEXT: n=3 fma: 80000 cases, 0 differ
// REF-NEXT: n=3 fma(a,a,c): 80000 cases, 0 differ
// REF-NEXT: n=4 add: 80000 cases, 0 differ
// REF-NEXT: n=4 sub: 80000 cases, 0 differ
// REF-NEXT: n=4 mul: 80000 cases, 0 differ
// REF-NEXT: n=4 a*a: 80000 cases, 0 differ
// REF-NEXT: n=4 div: 80000 cases, 0 differ
// REF-NEXT: n=4 sqrt: 80000 cases, 0 differ
// REF-NEXT: n=4 neg: 80000 cases, 0 differ
// REF-NEXT: n=4 fma: 80000 cases, 0 differ
// REF-NEXT: n=4 fma(a,a,c): 80000 cases, 0 differ
// REF-NEXT: n=2 add: 80000 cases, 0 differ
// REF-NEXT: n=2 sub: 80000 cases, 0 differ
// REF-NEXT: n=2 mul: 80000 cases, 0 differ
// REF-NEXT: n=2 sqrt: 80000 cases, 0 differ
// REF-NEXT: n=2 div: 80000 cases, 0 differ
// REF-NEXT: n=2 sqrt(0x0p+0, 0x0p+0) = 0x0p+0 0x0p+0
// REF-NEXT: n=2 sqrt(-0x0p+0, 0x0p+0) = -0x0p+0 0x0p+0
// REF-NEXT: n=2 sqrt(0x0p+0, -0x0p+0) = 0x0p+0 -0x0p+0
// REF-NEXT: n=2 sqrt(-0x0p+0, -0x0p+0) = -0x0p+0 -0x0p+0
// REF-NEXT: n=2 (-10.5, 0) / (-14, 0) = 0x1.8p-1 0x0p+0

// NVPTX-LABEL: define void @p2_sqrt(
// NVPTX: call float @llvm.sqrt.f32(
// NVPTX-LABEL: define void @p3_sqrt(
// NVPTX: call float @llvm.sqrt.f32(
// NVPTX-LABEL: define void @p4_sqrt(
// NVPTX: call float @llvm.sqrt.f32(

// The accuracy model's emulation of the FP32 expansions computes what the
// materializer emits, bit for bit (NaN payload and sign excepted), for n = 2, 3,
// 4: every routine on 80,000 raw operands, the boundary conversions, and
// getFPValues on functions of one operation and chains of three to six,
// components kept within one change and split again across changes. The model
// is the plugin's own; two runs emit the same model.ll. Compiled at -O0: up to
// LLVM 22 InstCombine turns the -0 low component of some squares into +0.

// MODEL: n=2 FP32 chains: 16000 cases, 0 differ
// MODEL-NEXT: n=2 a*a: 80000 cases, 0 differ
// MODEL-NEXT: n=2 add: 80000 cases, 0 differ
// MODEL-NEXT: n=2 chains, one change: 16000 cases, 0 differ
// MODEL-NEXT: n=2 chains, one operation left in FP64: 16000 cases, 0 differ
// MODEL-NEXT: n=2 chains, two changes: 16000 cases, 0 differ
// MODEL-NEXT: n=2 div: 80000 cases, 0 differ
// MODEL-NEXT: n=2 fma: 80000 cases, 0 differ
// MODEL-NEXT: n=2 fma(a,a,c): 80000 cases, 0 differ
// MODEL-NEXT: n=2 from_f64: 80000 cases, 0 differ
// MODEL-NEXT: n=2 mul: 80000 cases, 0 differ
// MODEL-NEXT: n=2 neg: 80000 cases, 0 differ
// MODEL-NEXT: n=2 single operations: 40000 cases, 0 differ
// MODEL-NEXT: n=2 sqrt: 80000 cases, 0 differ
// MODEL-NEXT: n=2 sub: 80000 cases, 0 differ
// MODEL-NEXT: n=2 to_f32: 80000 cases, 0 differ
// MODEL-NEXT: n=2 to_f64: 80000 cases, 0 differ
// MODEL-NEXT: n=3 FP32 chains: 16000 cases, 0 differ
// MODEL-NEXT: n=3 a*a: 80000 cases, 0 differ
// MODEL-NEXT: n=3 add: 80000 cases, 0 differ
// MODEL-NEXT: n=3 chains, one change: 16000 cases, 0 differ
// MODEL-NEXT: n=3 chains, one operation left in FP64: 16000 cases, 0 differ
// MODEL-NEXT: n=3 chains, two changes: 16000 cases, 0 differ
// MODEL-NEXT: n=3 div: 80000 cases, 0 differ
// MODEL-NEXT: n=3 fma: 80000 cases, 0 differ
// MODEL-NEXT: n=3 fma(a,a,c): 80000 cases, 0 differ
// MODEL-NEXT: n=3 from_f64: 80000 cases, 0 differ
// MODEL-NEXT: n=3 mul: 80000 cases, 0 differ
// MODEL-NEXT: n=3 neg: 80000 cases, 0 differ
// MODEL-NEXT: n=3 single operations: 40000 cases, 0 differ
// MODEL-NEXT: n=3 sqrt: 80000 cases, 0 differ
// MODEL-NEXT: n=3 sub: 80000 cases, 0 differ
// MODEL-NEXT: n=3 to_f32: 80000 cases, 0 differ
// MODEL-NEXT: n=3 to_f64: 80000 cases, 0 differ
// MODEL-NEXT: n=4 FP32 chains: 16000 cases, 0 differ
// MODEL-NEXT: n=4 a*a: 80000 cases, 0 differ
// MODEL-NEXT: n=4 add: 80000 cases, 0 differ
// MODEL-NEXT: n=4 chains, one change: 16000 cases, 0 differ
// MODEL-NEXT: n=4 chains, one operation left in FP64: 16000 cases, 0 differ
// MODEL-NEXT: n=4 chains, two changes: 16000 cases, 0 differ
// MODEL-NEXT: n=4 div: 80000 cases, 0 differ
// MODEL-NEXT: n=4 fma: 80000 cases, 0 differ
// MODEL-NEXT: n=4 fma(a,a,c): 80000 cases, 0 differ
// MODEL-NEXT: n=4 from_f64: 80000 cases, 0 differ
// MODEL-NEXT: n=4 mul: 80000 cases, 0 differ
// MODEL-NEXT: n=4 neg: 80000 cases, 0 differ
// MODEL-NEXT: n=4 single operations: 40000 cases, 0 differ
// MODEL-NEXT: n=4 sqrt: 80000 cases, 0 differ
// MODEL-NEXT: n=4 sub: 80000 cases, 0 differ
// MODEL-NEXT: n=4 to_f32: 80000 cases, 0 differ
// MODEL-NEXT: n=4 to_f64: 80000 cases, 0 differ

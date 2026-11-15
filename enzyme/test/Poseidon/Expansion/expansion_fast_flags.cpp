// RUN: rm -rf %t && mkdir %t
// RUN: %clang++ $(llvm-config --cxxflags) -fPIC -shared -I%S/../../../Poseidon %S/../Inputs/expansion_fast_flags/apply_expansion.cpp -o %t/apply_expansion%shlibext
// RUN: %opt %S/../Inputs/expansion_fast_flags/fast.ll %loadPoseidon -load-pass-plugin=%t/apply_expansion%shlibext -passes=apply-expansion -S -o %t/expanded.ll
// RUN: FileCheck --check-prefix=IR --implicit-check-not='{{ (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) }}' %s < %t/expanded.ll
// RUN: %clang++ -O2 -c %t/expanded.ll -o %t/expanded.o
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off %S/../Inputs/expansion_fast_flags/harness.cpp %t/expanded.o -o %t/run
// RUN: %t/run | FileCheck --check-prefix=OUT %s

// The expansion materializer applied to FP64 operations that carry every
// fast-math flag (Inputs/expansion_fast_flags/fast.ll) emits operations with
// none, for two, three and four components: an error-free transformation is
// exact only if each operation rounds as written (with 'nnan ninf contract'
// kept, -O2 folds each operand's split and the result keeps FP32 precision).
// Compiled at -O2 and run on the host, the two-component sum and product equal,
// bit for bit, MultiFloats.jl's two-limb addition and multiplication on the
// operands' splits (Inputs/expansion_fast_flags/harness.cpp).

// IR-LABEL: define double @ex2_add(
// IR: fptrunc double %a to float
// IR-LABEL: define double @ex2_mul(
// IR: call float @llvm.fma.f32(
// IR-LABEL: define double @ex2_div(
// IR: fdiv float
// IR-LABEL: define double @ex2_fma(
// IR: call float @llvm.fma.f32(
// IR-LABEL: define double @ex3_mul(
// IR: call float @llvm.fma.f32(
// IR-LABEL: define double @ex4_add(
// IR: fadd float

// OUT: ex2_add(0.1, 0.2) = 0x1.333333333333p-2
// OUT: sum 100000 of 100000, product 100000 of 100000 equal to the reference

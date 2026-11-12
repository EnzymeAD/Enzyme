// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang++ $(llvm-config --cxxflags) -fPIC -shared -I%S/../../../Poseidon %S/../Inputs/expansion_fast_flags/apply_expansion.cpp -o %t/apply_expansion%shlibext
// RUN: %opt %S/../Inputs/expansion_fast_flags/fast_attributes.ll %loadPoseidon -load-pass-plugin=%t/apply_expansion%shlibext -passes=apply-expansion -S -o %t/expanded.ll
// RUN: %clang++ -O2 -c %t/expanded.ll -o %t/expanded.o
// RUN: %clang++ -std=c++17 -O2 -ffp-contract=off %S/../Inputs/expansion_fast_flags/harness.cpp %t/expanded.o -o %t/run
// RUN: %t/run | FileCheck %s
// UNSUPPORTED: global-fp-contract

// expansion_fast_flags.cpp in functions that carry every fast-math function
// attribute (unsafe-fp-math, no-nans-fp-math, no-infs-fp-math,
// approx-func-fp-math, no-signed-zeros-fp-math). From LLVM 24 no attribute
// reaches an operation without fast-math flags, so the expansion code compiled
// at -O2 still equals MultiFloats.jl's two-limb addition and multiplication bit
// for bit, and a site in such a function gets its expansions.

// CHECK: ex2_add(0.1, 0.2) = 0x1.333333333333p-2
// CHECK: sum 100000 of 100000, product 100000 of 100000 equal to the reference

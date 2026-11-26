// REQUIRES: global-fp-contract, enzyme
// RUN: (%poseidon_clangxx -poseidon-profile-use -O2 -ffast-math -c %s -o %t.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=FAST %s
// RUN: (%poseidon_clangxx -poseidon-profile-use -ffp-contract=fast -c %s -o %t.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=CONTRACT %s
// RUN: (%poseidon_clangxx -poseidon-profile-use -ffp-model=fast -fno-fast-math -c %s -o %t.o 2>&1; echo "exit status $?") | FileCheck --check-prefix=UNDONE %s
// RUN: %poseidon_clangxx -poseidon-profile-generate -ffast-math -c %s -o %t.o

// A profile-use compile whose flags let the compiler fuse or reorder FP
// operations across statements is refused, naming the flag; one a later flag
// undoes, or a profile-generate compile, is not. Before LLVM 24 only.
// FAST: poseidon-clang++: error: Poseidon: -ffast-math lets the compiler fuse or reorder floating-point operations across statements (clang passes {{-funsafe-math-optimizations|-mreassociate|-ffp-contract=fast}}), which collapses the floating-point expansions Poseidon emits; drop it from the profile-use compile{{$}}
// FAST-NEXT: exit status 1
// CONTRACT: poseidon-clang++: error: Poseidon: -ffp-contract=fast lets the compiler fuse or reorder floating-point operations across statements (clang passes -ffp-contract=fast), which collapses the floating-point expansions Poseidon emits; drop it from the profile-use compile{{$}}
// CONTRACT-NEXT: exit status 1
// UNDONE: exit status 0

int main(void) { return 0; }

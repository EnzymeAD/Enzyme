// RUN: not %poseidon_clangxx -poseidon-profile-generate -poseidon-profile-use -c %s -o %t.o 2>&1 | FileCheck --check-prefix=MODES %s

// A mistake in what poseidon-clang is given is reported in the form the pass
// reports one in, exit status 1: both halves of the workflow at once.

// MODES: poseidon-clang++: error: Poseidon: -poseidon-profile-generate and -poseidon-profile-use are the two halves of the workflow; pass one

int main(void) { return 0; }

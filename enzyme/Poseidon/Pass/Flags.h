//=- Flags.h - every Poseidon command-line flag, read once -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Poseidon's flags are defined in Flags.cpp, the one place a flag's name and
// help live and the only reader of a flag; its default is the initializer of
// its field in Core/Options.h. Every flag is spelled -poseidon-<name>, reaches
// the pass as -mllvm -poseidon-<name>, and is supplied by poseidon-clang from
// the -poseidon-<name> the user writes.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_FLAGS_H
#define POSEIDON_FLAGS_H

#include "Core/Options.h"

namespace poseidon {

// The options of this compile, with the defaults of the flag spellings that
// carry one only when written without a value. Fails the compile when a
// -poseidon-* flag is listed by neither or both of solveFlagValues() and
// notSolveInputs().
Options readOptions();

} // namespace poseidon
#endif // POSEIDON_FLAGS_H

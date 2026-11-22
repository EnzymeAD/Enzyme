//===- Flags.h - every Poseidon command-line flag, read once --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Every -poseidon-<name> flag is defined and read in Flags.cpp; its default is
// the initializer of its field in Core/Options.h.
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

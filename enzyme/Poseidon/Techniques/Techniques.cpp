//===- Techniques.cpp - the techniques of the pass ------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The one list of techniques (Core/Technique.h).
//
//===----------------------------------------------------------------------===//

#include "Core/Technique.h"
#include "Techniques/Algebraic/Algebraic.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision/Precision.h"

using namespace llvm;

namespace poseidon {

ArrayRef<const Technique *> techniques() {
  static const Technique *const List[] = {
      &algebraicTechnique(), &precisionTechnique(), &matmulTechnique()};
  return List;
}

} // namespace poseidon

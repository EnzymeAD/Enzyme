//=- Pipeline.h - the Poseidon passes and where they run --------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PIPELINE_H
#define POSEIDON_PIPELINE_H

namespace llvm {
class PassBuilder;
} // namespace llvm

namespace poseidon {

void registerPasses(llvm::PassBuilder &PB);

} // namespace poseidon

#endif // POSEIDON_PIPELINE_H

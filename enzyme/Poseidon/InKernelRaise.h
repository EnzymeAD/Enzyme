//=- InKernelRaise.h - Ozaki Scheme I matmul candidates for Poseidon ------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ozaki Scheme I tensor-core raise for scalar-loop matmuls: a Veltkamp N-slice
// split, an N(N+1)/2-mma chain per K-block and a scaled readback combine.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_IN_KERNEL_RAISE_H
#define POSEIDON_IN_KERNEL_RAISE_H

#include "matmul/Matmul.h"

namespace poseidon {

// Emit the Ozaki Scheme I N-slice rewrite for a scalar-loop matmul with an
// F16/BF16/TF32 input slice and an F32 accumulator; aborts on any precondition
// violation (callers filter to valid candidates upstream).
void materializeOzakiIRaise(const AbstractMatmul &m,
                            const CandidateMatmul::Option &opt);

std::string ozakiIOptionLabel(const CandidateMatmul::Option &opt);

} // namespace poseidon
#endif // POSEIDON_IN_KERNEL_RAISE_H

//===- Instrument.h - profiling probes on a site clone --------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_INSTRUMENT_H
#define POSEIDON_INSTRUMENT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"

#include <cstddef>

namespace llvm {
class Function;
class Module;
} // namespace llvm

namespace poseidon {

// The profiler runtime function `name` of type `FT`, declared in `M` if absent,
// marked enzyme_inactive and nofree so that AD never differentiates it.
llvm::FunctionCallee getProfilerHook(llvm::Module &M, llvm::StringRef name,
                                     llvm::FunctionType *FT);

// Place the profiling probes on a canonicalized, slot-numbered site clone.
// Every optimizable instruction gets its value logged in the primal and its
// result routed through an identity probe whose Enzyme custom derivative logs
// the accumulated adjoint. That probe's augmented forward also carries the
// condition-number perturbation, armed at run time by `siteId`, so an armed run
// computes the whole site at the probed relative precision. Returns the number
// of instrumented instructions.
size_t instrumentForProfiling(llvm::Function &clone, unsigned siteId);

// Embed the site's compile-time static data (header lines the profile-use
// compile cannot recompute) in the instrumented module and register it with the
// profiler runtime, which copies it verbatim into the site's .fpprofile header.
// `text` is a sequence of newline-terminated "Key = value" lines.
void emitProfileStaticData(llvm::Function &clone, llvm::StringRef text);

// Register `text` before main with the host profiler runtime as the header of
// the site whose profile is written under `name`. False if host module M
// already does.
bool registerProfileStatic(llvm::Module &M, llvm::StringRef name,
                           llvm::StringRef text);

} // namespace poseidon

#endif // POSEIDON_INSTRUMENT_H

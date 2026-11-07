#ifndef POSEIDON_INSTRUMENT_H
#define POSEIDON_INSTRUMENT_H

#include "llvm/ADT/StringRef.h"

#include <cstddef>

namespace llvm {
class Function;
} // namespace llvm

namespace poseidon {

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

} // namespace poseidon

#endif // POSEIDON_INSTRUMENT_H

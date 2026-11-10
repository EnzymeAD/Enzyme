//===- Peers.h - the three targets, for Targets/ only ---------------------===//
#ifndef POSEIDON_TARGET_PEERS_H
#define POSEIDON_TARGET_PEERS_H

#include "Target.h"

#include <string>
#include <unordered_set>

namespace poseidon {

const Target &hostTarget();
const Target &nvptxTarget();
const Target &amdgpuTarget();

// What the probes of either GPU target call: one device profiler serves both.
const ProbeABI &deviceProbes();

// libm's functions Poseidon handles, with fmin and fmax, which getLLValue
// emits as llvm.minnum/llvm.maxnum and both GPU targets select.
const std::unordered_set<std::string> &mathBases();

// libm's `name` at `Ty`, empty if it is none Poseidon can retype.
std::string retypeLibmFunction(llvm::StringRef name, llvm::Type *Ty);

// Herbie's operators a rewrite writes as math-library calls on either GPU
// target: their intrinsics mostly have no instruction there and fail
// instruction selection ("Cannot select: fNN = fsin").
bool deviceCallsMathLibrary(llvm::StringRef op, llvm::Type *Ty);

} // namespace poseidon

#endif // POSEIDON_TARGET_PEERS_H

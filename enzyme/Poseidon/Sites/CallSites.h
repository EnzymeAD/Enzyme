//===- CallSites.h - calls of __poseidon_optimize -------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// __poseidon_optimize on a call, in host and device code: the call made a call
// of a copy of the function made for it alone, which is a site, and in device
// code a marker over that copy.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CALL_SITES_H
#define POSEIDON_CALL_SITES_H

#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class Argument;
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

struct Options;

bool lowerPerCallSites(const Options &O, llvm::Module &M);
bool lowerDeviceCallSites(const Options &O, llvm::Module &M);

// The object of `caller` that `p` points into, through the address arithmetic
// its shadow can repeat, when the pass knows the object's whole extent: a
// static alloca, or an aggregate `caller` takes by value (byval). Null if there
// is none. (An AMDGPU kernel takes an aggregate by reference into its constant
// argument segment, and the frontend copies it into an alloca once its address
// is taken.)
llvm::Value *localObject(llvm::Value *p, const llvm::Function *caller);

// A pointer argument that points into no object: null, or undefined because the
// call's copy never reads it and the optimizer stopped passing the object (the
// address of an empty functor).
bool pointsNowhere(llvm::Value *p);

// What a function reaches through one of its pointer parameters, in bytes from
// the pointer it is passed: [lo, hi) read or written, [writeLo, writeHi)
// written, and the strictest alignment an access claims, `align` at offset
// `alignedAt`.
struct PointerExtent {
  int64_t lo = 0, hi = 0, writeLo = 0, writeHi = 0;
  uint64_t align = 1;
  int64_t alignedAt = 0;
  bool accessed() const { return lo < hi; }
  bool written() const { return writeLo < writeHi; }
};
// The extent of `A`, or none, with in `why` what the function does with the
// pointer that defeats it, phrased to follow "a pointer the call".
std::optional<PointerExtent> accessedExtent(llvm::Argument &A,
                                            std::string &why);

} // namespace poseidon

#endif // POSEIDON_CALL_SITES_H

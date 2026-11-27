//===- CallSites.h - calls of __poseidon_optimize -------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// __poseidon_optimize on a call, in host and device code: the call made a call
// of a copy of the function made for it alone, which is a site, and in device
// code a marker over that copy. Also what a site's body reaches through its
// parameters: the extent of a pointer argument, and the parameter or by-value
// field a value is derived from.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITES_CALL_SITES_H
#define POSEIDON_SITES_CALL_SITES_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

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
struct SiteRecords;

/// Turns each __poseidon_optimize call into a call of a copy of its target
/// made for that call, marked as a host body and recorded in `T`.
llvm::Expected<bool> lowerPerCallSites(const Options &O, SiteRecords &T,
                                       llvm::Module &M);
/// In a device module, turns each call of a per-call copy into a marker call
/// whose pointers the profiler shadows.
llvm::Expected<bool> lowerDeviceCallSites(const Options &O, llvm::Module &M);

/// The object of `caller` that `p` points into, through the address arithmetic
/// its shadow can repeat, when the pass knows the object's whole extent: a
/// static alloca, or an aggregate `caller` takes by value (byval). Null if
/// there is none. (An AMDGPU kernel takes an aggregate by reference into its
/// constant argument segment, and the frontend copies it into an alloca once
/// its address is taken.)
llvm::Value *localObject(llvm::Value *p, const llvm::Function *caller);

/// A pointer argument that points into no object: null, or undefined because
/// the call's copy never reads it and the optimizer stopped passing the object
/// (the address of an empty functor).
bool pointsNowhere(llvm::Value *p);

/// What a function reaches through one of its pointer parameters, in bytes from
/// the pointer it is passed: [lo, hi) read or written, [writeLo, writeHi)
/// written, and the strictest alignment an access claims, `align` at offset
/// `alignedAt`.
struct PointerExtent {
  int64_t lo = 0, hi = 0, writeLo = 0, writeHi = 0;
  uint64_t align = 1;
  int64_t alignedAt = 0;
  /// Whether the function reaches any byte through the pointer.
  bool accessed() const { return lo < hi; }
  /// Whether the function writes any byte through the pointer.
  bool written() const { return writeLo < writeHi; }
};
/// The extent of `A`; otherwise an error saying what the function does with
/// the pointer that defeats it, phrased to follow "a pointer the call".
llvm::Expected<PointerExtent> accessedExtent(llvm::Argument &A);

/// A value the host knows at the launch: a parameter of the optimized body, or
/// a field of an aggregate a parameter passes by value (the captures of a
/// lambda a kernel takes by value).
struct LaunchRef {
  /// The parameter index; -1 for no value.
  int param = -1;
  /// Byte offset of the field in the aggregate; -1 for the parameter itself.
  int64_t offset = -1;
  /// Whether the value is a field of an aggregate parameter.
  bool isField() const { return offset >= 0; }
  /// Whether both name the same parameter and field.
  bool operator==(const LaunchRef &o) const {
    return param == o.param && offset == o.offset;
  }
  /// Orders by parameter, then by offset, the parameter itself first.
  bool operator<(const LaunchRef &o) const {
    return param != o.param ? param < o.param : offset < o.offset;
  }
};

/// `r` as a diagnostic names it: `prefix` and the parameter index, then
/// "[+<offset>]" for a field.
std::string launchRefString(const LaunchRef &r, llvm::StringRef prefix);

/// Marks a parameter of an outlined kernel body that carries the kernel's
/// by-value aggregate argument: the body reads the kernel's own copy, so the
/// parameter is no longer byval, and only this says its fields are launch
/// values.
constexpr llvm::StringLiteral kByValueParam = "poseidon-by-value";

/// The field of a by-value aggregate parameter `v` reads (a load at a constant
/// offset from a parameter the body only reads, or an extractvalue of an
/// aggregate parameter); an invalid ref otherwise.
LaunchRef launchFieldOf(const llvm::Value *v);

/// The launch value a pointer is derived from (its underlying object): a
/// parameter, or a pointer field of a by-value aggregate parameter.
LaunchRef tracePointerRoot(const llvm::Value *v);

/// The parameter a pointer is derived from (its underlying object), -1 if none.
int traceToArgIndex(const llvm::Value *v);

} // namespace poseidon

#endif // POSEIDON_SITES_CALL_SITES_H

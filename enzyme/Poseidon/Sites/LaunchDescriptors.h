//=- LaunchDescriptors.h - launch-stub descriptor plumbing ----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device and host code are separate compilations, so a device-side solve that
// changes how a kernel must be launched reaches the host through a file: the
// manifest of the translation unit in the cache directory, which holds one
// single-line descriptor per wrapper kernel. The device compile writes it
// whole once its sites are processed, and the host compile of the same source
// and unit reads it and no other. Every feature that rewrites a launch stub
// shares the note map, the manifest and the stub locator here and keeps only
// its own payload and rewrite body.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_LAUNCH_DESCRIPTORS_H
#define POSEIDON_LAUNCH_DESCRIPTORS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <string>

namespace poseidon {

struct Options;
class Session;

// A module compile's stash: the solve records a payload keyed by the
// optimized body clone, and the marker handler retrieves it after the solve
// returns. `Note` must have a `valid` member.
template <class Note> class NoteMap {
public:
  void note(const llvm::Function *body, const Note &n) { notes[body] = n; }
  bool get(const llvm::Function *body, Note &out) const {
    auto it = notes.find(body);
    if (it == notes.end() || !it->second.valid)
      return false;
    out = it->second;
    return true;
  }

private:
  std::map<const llvm::Function *, Note> notes;
};

struct Recorded {
  std::string line;
  std::string message;
};

struct ProfileHeader {
  std::string profileName;
  std::string header;
};

enum class LaunchKind {
  // A raised GEMM the host stub dispatches to a library (the payload is
  // written by HostDispatch.cpp).
  HostDispatch,
  // An annotated kernel the host stub launches through the profiling helper:
  //   <kernel> <site id> <original arg count> <pointer arg count>
  //     <pointer arg index>...
  //     <seed width in bytes of that pointer, 0 if it is not an output>...
  //     [off <byte offset of that pointer in its argument>...]
  // The offsets are written when a pointer is a field of an aggregate argument.
  Profiling,
};

// Whether the host half of this compile rewrites launch stubs from a manifest:
// under profile generation, and under profile use with host dispatch.
bool handsOverLaunches(const Options &O);

// Records "<wrapperName><payload>" as the `kind` descriptor of that kernel in
// place of one recorded before; `emitPayload` appends everything after the
// name. `message` is printed when the manifest is written.
void recordDescriptor(Session &S, LaunchKind kind, llvm::StringRef wrapperName,
                      llvm::function_ref<void(llvm::raw_ostream &)> emitPayload,
                      std::string message);

// Records, beside the profile generation descriptor of `wrapperName`, the name
// its site's profile is written under and the header the device compile
// computed for it, which the device profiler writes only once the kernel runs.
void recordProfileHeader(Session &S, llvm::StringRef wrapperName,
                         llvm::StringRef profileName, llvm::StringRef header);

// Replaces the manifest of device module M by the descriptors recorded since
// the last write and prints their messages, all in order of kernel name; with
// none recorded, removes it, since no manifest means nothing to rewrite. A
// manifest that cannot be written or removed stops the compile.
void writeLaunchManifest(Session &S, const llvm::Module &M);

// Hands `parse` the space-separated tokens of every `kind` descriptor in the
// manifest of host module M; token 0 is always the wrapper kernel name. A unit
// without a manifest has none.
void readDescriptors(
    const Options &O, const llvm::Module &M, LaunchKind kind,
    llvm::function_ref<void(llvm::ArrayRef<llvm::StringRef>)> parse);

// Hands `use` the profile name and header of every profiled kernel in the
// manifest of host module M. A unit without a manifest has none; one that
// cannot be read is reported by readDescriptors.
void readProfileHeaders(const Options &O, const llvm::Module &M,
                        llvm::function_ref<void(llvm::StringRef profileName,
                                                llvm::StringRef header)>
                            use);

// The part of a kernel's mangled name shared with its launch stub: kernel
// `_Z<L><id><params>` and stub `_Z<L+15>__device_stub__<id><params>`.
llvm::StringRef mangledSuffix(llvm::StringRef mangled);

// Whether F is the kernel named `kernelName` or that kernel's launch stub.
bool isLaunchStubFor(const llvm::Function &F, llvm::StringRef kernelName);

// A value the host knows at the launch: a parameter of the optimized body, or a
// field of an aggregate a parameter passes by value (the captures of a lambda a
// kernel takes by value: DeviceTensor data pointers and sizes).
struct LaunchRef {
  int param = -1;
  int64_t offset = -1; // byte offset of the field; -1 = the parameter itself
  bool isField() const { return offset >= 0; }
  bool operator==(const LaunchRef &o) const {
    return param == o.param && offset == o.offset;
  }
  bool operator<(const LaunchRef &o) const {
    return param != o.param ? param < o.param : offset < o.offset;
  }
};

// Marks a parameter of an outlined kernel body that carries the kernel's
// by-value aggregate argument: the body reads the kernel's own copy, so the
// parameter is no longer byval, and only this says its fields are launch
// values.
constexpr llvm::StringLiteral kByValueParam = "poseidon-by-value";

// The field of a by-value aggregate parameter `v` reads (a load at a constant
// offset from a parameter the body only reads, or an extractvalue of an
// aggregate parameter); an invalid ref otherwise.
LaunchRef launchFieldOf(const llvm::Value *v);

// The launch value a pointer is derived from through GEP/cast chains: a
// parameter, or a pointer field of a by-value aggregate parameter.
LaunchRef tracePointerRoot(const llvm::Value *v);

// Walk GEP/bitcast/addrspacecast chains back to a Function Argument (-1 if
// none).
int traceToArgIndex(const llvm::Value *v);

} // namespace poseidon
#endif // POSEIDON_LAUNCH_DESCRIPTORS_H

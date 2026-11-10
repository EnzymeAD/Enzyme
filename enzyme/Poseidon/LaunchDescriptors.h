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

#include <map>
#include <string>

namespace poseidon {

// Process-local (device cc1) stash: the solve records a payload keyed by the
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
bool handsOverLaunches();

// Records "<wrapperName><payload>" as the `kind` descriptor of that kernel in
// place of one recorded before; `emitPayload` appends everything after the
// name. `message` is printed when the manifest is written.
void recordDescriptor(LaunchKind kind, llvm::StringRef wrapperName,
                      llvm::function_ref<void(llvm::raw_ostream &)> emitPayload,
                      std::string message);

// Replaces the manifest of device module M by the descriptors recorded since
// the last write and prints their messages, all in order of kernel name; with
// none recorded, removes it, since no manifest means nothing to rewrite. A
// manifest that cannot be written or removed stops the compile.
void writeLaunchManifest(const llvm::Module &M);

// Hands `parse` the space-separated tokens of every `kind` descriptor in the
// manifest of host module M; token 0 is always the wrapper kernel name. A unit
// without a manifest has none.
void readDescriptors(
    const llvm::Module &M, LaunchKind kind,
    llvm::function_ref<void(llvm::ArrayRef<llvm::StringRef>)> parse);

// The part of a kernel's mangled name shared with its launch stub: kernel
// `_Z<L><id><params>` and stub `_Z<L+15>__device_stub__<id><params>`.
llvm::StringRef mangledSuffix(llvm::StringRef mangled);

// `name` without the postfix clang appends under -fgpu-rdc to the device-side
// name of a kernel with internal linkage ("__intern__<hash>", in HIP
// ".intern.<hash>"). The host stub's name has none, and the hash follows the
// compile's macros, so neither the stub nor a later compile without -fgpu-rdc
// can be matched on it.
llvm::StringRef withoutExternalizedPostfix(llvm::StringRef name);

// Whether F is the kernel named `kernelName` or that kernel's launch stub.
bool isLaunchStubFor(const llvm::Function &F, llvm::StringRef kernelName);

} // namespace poseidon
#endif // POSEIDON_LAUNCH_DESCRIPTORS_H

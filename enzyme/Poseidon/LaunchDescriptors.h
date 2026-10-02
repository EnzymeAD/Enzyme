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
// changes how a kernel must be launched reaches the host through a file: one
// single-line descriptor per wrapper kernel, named "<wrapper><scheme>" in the
// cache directory. Every feature that rewrites a launch stub (host GEMM
// dispatch, df64 parameter staging) shares the note map, the file format and
// the stub locator here and keeps only its own payload and rewrite body.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_LAUNCH_DESCRIPTORS_H
#define POSEIDON_LAUNCH_DESCRIPTORS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

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

// Write "<wrapperName><payload>\n" to <cacheDir>/<wrapperName><scheme>;
// `emitPayload` appends everything after the name. False if the file could not
// be opened (reported on stderr under `tag`).
bool writeDescriptor(llvm::StringRef cacheDir, llvm::StringRef wrapperName,
                     llvm::StringRef scheme, llvm::StringRef tag,
                     llvm::function_ref<void(llvm::raw_ostream &)> emitPayload);

// Drop a descriptor an earlier solve left behind, so a stale one can never be
// replayed against a kernel this solve did not rewrite.
void removeDescriptor(llvm::StringRef cacheDir, llvm::StringRef wrapperName,
                      llvm::StringRef scheme);

// Hand `parse` the space-separated tokens of every "<*><scheme>" descriptor in
// cacheDir; token 0 is always the wrapper kernel name.
void readDescriptors(
    llvm::StringRef cacheDir, llvm::StringRef scheme,
    llvm::function_ref<void(llvm::ArrayRef<llvm::StringRef>)> parse);

// The part of a kernel's mangled name shared with its launch stub: kernel
// `_Z<L><id><params>` and stub `_Z<L+15>__device_stub__<id><params>`.
llvm::StringRef mangledSuffix(llvm::StringRef mangled);

// Whether F is the kernel named `kernelName` or that kernel's launch stub.
bool isLaunchStubFor(const llvm::Function &F, llvm::StringRef kernelName);

} // namespace poseidon
#endif // POSEIDON_LAUNCH_DESCRIPTORS_H

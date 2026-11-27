//===- LaunchManifest.h - launch-stub descriptor plumbing -----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device and host code are separate compilations, so a device-side solve that
// changes how a kernel must be launched reaches the host through a file: the
// manifest of the translation unit in the cache directory. The device compile
// writes it whole once its sites are processed, and the host compile of the
// same source and unit reads it and no other.
//
// The manifest is a cache entry (Support/CacheKey.h) whose value is a JSON
// object: "library calls" and "profile generation", each the list of the
// single-line descriptors of that LaunchKind, one per wrapper kernel; and from
// a profile-generate compile, "profile headers", which maps a kernel to
// {"profile": <profile name>, "header": <header text>}, and "profile table",
// {"functions": <count>, "slots": <count>}.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITES_LAUNCH_MANIFEST_H
#define POSEIDON_SITES_LAUNCH_MANIFEST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace poseidon {

struct Options;

/// A launch descriptor: its manifest line and the message printed when the
/// manifest is written.
struct Recorded {
  /// The descriptor as the manifest holds it: the kernel name, then its fields.
  std::string line;
  /// What is printed, under -poseidon-print, when the manifest is written.
  std::string message;
};

/// The header of a device site's profile, which the host half registers
/// under the profile's name.
struct ProfileHeaderText {
  /// The name the profile is written and registered under.
  std::string profileName;
  /// The profile's header text, as the profiling runtime writes it.
  std::string header;
};

/// What a device module hands its host half, gathered while its sites are
/// solved and written once by writeLaunchManifest.
struct LaunchManifest {
  static inline char ID;
  /// By kernel name, so that the manifest and its messages come out in one
  /// order whatever order the sites were solved in.
  std::map<std::string, Recorded> LibraryCalls, Profiling;
  /// The profile headers of the profiled kernels, by kernel name.
  std::map<std::string, ProfileHeaderText> ProfileHeaders;
  /// What the device profiler's tables must hold for this module: the
  /// functions it profiles and the probes of the largest.
  unsigned ProfiledFunctions = 0, ProfiledSlots = 0;
};

/// The two kinds of launch a device module hands its host half.
enum class LaunchKind {
  /// A raised GEMM the host stub dispatches to a library:
  ///   <kernel> <C arg> <A arg> <B arg> <standalone 0/1> <modulus field>
  ///     <A col-major 0/1> <B col-major 0/1> <C col-major 0/1> <scheme>
  ///     rd, then per dimension M, N, K, lda, ldb, ldc: <factor count>,
  ///       <factor arg> <factor offset> per factor, <mul> <add>; then <beta>
  ///     [lc <M axis> <N axis>] [pf <C offset> <A offset> <B offset>]
  ///     [eq <count> (<arg> <offset> <arg> <offset>)...]
  /// The keyword blocks after the scheme come in any order; rd is required.
  LibraryCall,
  /// An annotated kernel the host stub launches through the profiling helper:
  ///   <kernel> <site id> <original arg count> <pointer arg count>
  ///     <pointer arg index>...
  ///     <seed width in bytes of that pointer, 0 if it is not an output>...
  ///     [off <byte offset of that pointer in its argument>...]
  /// The offsets are written when a pointer is a field of an aggregate
  /// argument.
  Profiling,
};

/// Whether the host half of this compile rewrites launch stubs from a manifest:
/// under profile generation, and under profile use with library call.
bool handsOverLaunches(const Options &O);

/// Records "<wrapperName><payload>" as the `kind` descriptor of that kernel in
/// place of one recorded before; `emitPayload` appends everything after the
/// name. `message` is printed, under -poseidon-print, when the manifest is
/// written.
void recordDescriptor(LaunchManifest &L, LaunchKind kind,
                      llvm::StringRef wrapperName,
                      llvm::function_ref<void(llvm::raw_ostream &)> emitPayload,
                      std::string message);

/// Records, beside the profile generation descriptor of `wrapperName`, the name
/// its site's profile is written under and the header the device compile
/// computed for it, which the device profiler writes only once the kernel runs.
void recordProfileHeader(LaunchManifest &L, llvm::StringRef wrapperName,
                         llvm::StringRef profileName, llvm::StringRef header);

/// Records, for the host half, the functions device module M profiles and the
/// probes the largest of them holds.
void recordProfileTable(LaunchManifest &L, unsigned functions, unsigned slots);

/// What the device half of host module M recorded with recordProfileTable;
/// zero counts when it profiles nothing or its manifest cannot be read, which
/// is a warning; an error when the entry is not two counts.
llvm::Expected<std::pair<unsigned, unsigned>>
readProfileTable(const Options &O, const llvm::Module &M);

/// Replaces the manifest of device module M by the descriptors recorded since
/// the last write and, under -poseidon-print, prints their messages, all in
/// order of kernel name; with none recorded, removes it, since no manifest
/// means nothing to rewrite. An error when the manifest cannot be written or
/// removed.
llvm::Error writeLaunchManifest(const Options &O, LaunchManifest &L,
                                const llvm::Module &M);

/// Hands `parse` the space-separated tokens of every `kind` descriptor in the
/// manifest of host module M; token 0 is always the wrapper kernel name. A unit
/// without a manifest has none, and one that cannot be read is a warning. A
/// descriptor `parse` refuses is an error naming the manifest, the descriptor
/// and what `parse` said.
llvm::Error readDescriptors(
    const Options &O, const llvm::Module &M, LaunchKind kind,
    llvm::function_ref<llvm::Error(llvm::ArrayRef<llvm::StringRef>)> parse);

/// The tokens of a descriptor after the kernel name, read in order. The first
/// token that is missing or does not parse stops the reading: every later read
/// gives a default value, and takeError says what was wrong.
class DescriptorCursor {
public:
  /// A cursor at the first token after the kernel name `toks[0]`; `toks` must
  /// outlive it.
  explicit DescriptorCursor(llvm::ArrayRef<llvm::StringRef> toks)
      : toks(toks) {}

  /// Whether every read so far found its token and parsed it.
  bool ok() const { return why.empty(); }
  /// Whether no token is left to read, or reading has stopped.
  bool atEnd() const { return !ok() || at == toks.size(); }
  /// Takes the next token when it is `keyword`.
  bool consume(llvm::StringRef keyword) {
    if (atEnd() || toks[at] != keyword)
      return false;
    ++at;
    return true;
  }
  /// The next token as a decimal integer of type `T`; 0 once reading has
  /// stopped, and stops it if the token is missing or not such an integer.
  /// `what` names the field for the error.
  template <typename T> T integer(llvm::StringRef what) {
    T value{};
    if (llvm::StringRef tok = next(what); ok() && tok.getAsInteger(10, value))
      fail("its " + what + " '" + tok + "' is not an integer of its range");
    return value;
  }
  /// The next token as a double; 0 once reading has stopped, and stops it if
  /// the token is missing or not a number. `what` names the field for the
  /// error.
  double real(llvm::StringRef what) {
    double value = 0.0;
    if (llvm::StringRef tok = next(what); ok() && tok.getAsDouble(value))
      fail("its " + what + " '" + tok + "' is not a number");
    return value;
  }
  /// Stops the reading with `reason`, unless it has already stopped.
  void fail(const llvm::Twine &reason) {
    if (ok())
      why = reason.str();
  }
  /// Success if every token was read and parsed; otherwise an error saying what
  /// was wrong first, phrased to follow "the descriptor".
  llvm::Error takeError() {
    if (ok() && at != toks.size())
      fail("it has the unexpected token '" + toks[at] + "'");
    return ok() ? llvm::Error::success()
                : llvm::createStringError(llvm::inconvertibleErrorCode(), why);
  }

private:
  /// The next token; empty, stopping the reading, when there is none.
  llvm::StringRef next(llvm::StringRef what) {
    if (!atEnd())
      return toks[at++];
    fail("it ends before its " + what);
    return llvm::StringRef();
  }

  /// The descriptor's tokens, the kernel name first.
  llvm::ArrayRef<llvm::StringRef> toks;
  /// Index of the next token to read.
  size_t at = 1;
  /// Why reading stopped; empty while it has not.
  std::string why;
};

/// Hands `use` the profile name and header of every profiled kernel in the
/// manifest of host module M, in kernel order. A unit without a manifest has
/// none, and one that cannot be read is a warning; an entry that is not a
/// profile name and a header is an error naming the manifest.
llvm::Error
readProfileHeaders(const Options &O, const llvm::Module &M,
                   llvm::function_ref<void(llvm::StringRef profileName,
                                           llvm::StringRef header)>
                       use);

} // namespace poseidon
#endif // POSEIDON_SITES_LAUNCH_MANIFEST_H

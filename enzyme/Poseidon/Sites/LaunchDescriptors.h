//===- LaunchDescriptors.h - launch-stub descriptor plumbing --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device and host code are separate compilations, so a device-side solve that
// changes how a kernel must be launched reaches the host through a file: the
// manifest of the translation unit in the cache directory, which holds one
// single-line descriptor per wrapper kernel. The device compile writes it
// whole once its sites are processed, and the host compile of the same source
// and unit reads it and no other.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_LAUNCH_DESCRIPTORS_H
#define POSEIDON_LAUNCH_DESCRIPTORS_H

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
class Session;

struct Recorded {
  std::string line;
  std::string message;
};

struct ProfileHeader {
  std::string profileName;
  std::string header;
};

enum class LaunchKind {
  /// A raised GEMM the host stub dispatches to a library.
  HostDispatch,
  /// An annotated kernel the host stub launches through the profiling helper:
  ///   <kernel> <site id> <original arg count> <pointer arg count>
  ///     <pointer arg index>...
  ///     <seed width in bytes of that pointer, 0 if it is not an output>...
  ///     [off <byte offset of that pointer in its argument>...]
  /// The offsets are written when a pointer is a field of an aggregate argument.
  Profiling,
};

/// Whether the host half of this compile rewrites launch stubs from a manifest:
/// under profile generation, and under profile use with host dispatch.
bool handsOverLaunches(const Options &O);

/// Records "<wrapperName><payload>" as the `kind` descriptor of that kernel in
/// place of one recorded before; `emitPayload` appends everything after the
/// name. `message` is printed when the manifest is written.
void recordDescriptor(Session &S, LaunchKind kind, llvm::StringRef wrapperName,
                      llvm::function_ref<void(llvm::raw_ostream &)> emitPayload,
                      std::string message);

/// Records, beside the profile generation descriptor of `wrapperName`, the name
/// its site's profile is written under and the header the device compile
/// computed for it, which the device profiler writes only once the kernel runs.
void recordProfileHeader(Session &S, llvm::StringRef wrapperName,
                         llvm::StringRef profileName, llvm::StringRef header);

/// Records, for the host half, the functions device module M profiles and the
/// probes the largest of them holds.
void recordProfileTable(Session &S, unsigned functions, unsigned slots);

/// What the device half of host module M recorded with recordProfileTable;
/// none when it profiles nothing.
std::pair<unsigned, unsigned> readProfileTable(const Options &O,
                                               const llvm::Module &M);

/// Replaces the manifest of device module M by the descriptors recorded since
/// the last write and prints their messages, all in order of kernel name; with
/// none recorded, removes it, since no manifest means nothing to rewrite. A
/// manifest that cannot be written or removed stops the compile.
void writeLaunchManifest(Session &S, const llvm::Module &M);

/// Hands `parse` the space-separated tokens of every `kind` descriptor in the
/// manifest of host module M; token 0 is always the wrapper kernel name. A unit
/// without a manifest has none. A descriptor `parse` refuses stops the compile,
/// naming the manifest, the descriptor and the error.
void readDescriptors(
    const Options &O, const llvm::Module &M, LaunchKind kind,
    llvm::function_ref<llvm::Error(llvm::ArrayRef<llvm::StringRef>)> parse);

/// The tokens of a descriptor after the kernel name, read in order. The first
/// token that is missing or does not parse stops the reading: every later read
/// gives a default value, and takeError says what was wrong.
class DescriptorCursor {
public:
  explicit DescriptorCursor(llvm::ArrayRef<llvm::StringRef> toks)
      : toks(toks) {}

  bool ok() const { return why.empty(); }
  bool atEnd() const { return !ok() || at == toks.size(); }
  /// Takes the next token when it is `keyword`.
  bool consume(llvm::StringRef keyword) {
    if (atEnd() || toks[at] != keyword)
      return false;
    ++at;
    return true;
  }
  template <typename T> T integer(llvm::StringRef what) {
    T value{};
    if (llvm::StringRef tok = next(what); ok() && tok.getAsInteger(10, value))
      fail("its " + what + " '" + tok + "' is not an integer of its range");
    return value;
  }
  double real(llvm::StringRef what) {
    double value = 0.0;
    if (llvm::StringRef tok = next(what); ok() && tok.getAsDouble(value))
      fail("its " + what + " '" + tok + "' is not a number");
    return value;
  }
  void fail(const llvm::Twine &reason) {
    if (ok())
      why = reason.str();
  }
  llvm::Error takeError() {
    if (ok() && at != toks.size())
      fail("it has the unexpected token '" + toks[at] + "'");
    return ok() ? llvm::Error::success()
                : llvm::createStringError(llvm::inconvertibleErrorCode(), why);
  }

private:
  llvm::StringRef next(llvm::StringRef what) {
    if (!atEnd())
      return toks[at++];
    fail("it ends before its " + what);
    return llvm::StringRef();
  }

  llvm::ArrayRef<llvm::StringRef> toks;
  size_t at = 1;
  std::string why;
};

/// Hands `use` the profile name and header of every profiled kernel in the
/// manifest of host module M. A unit without a manifest has none; one that
/// cannot be read is reported by readDescriptors.
void readProfileHeaders(const Options &O, const llvm::Module &M,
                        llvm::function_ref<void(llvm::StringRef profileName,
                                                llvm::StringRef header)>
                            use);

/// A value the host knows at the launch: a parameter of the optimized body, or a
/// field of an aggregate a parameter passes by value (the captures of a lambda a
/// kernel takes by value).
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
#endif // POSEIDON_LAUNCH_DESCRIPTORS_H

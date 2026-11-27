//===- LaunchManifest.cpp - launch-stub descriptor plumbing ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LaunchManifest.h"
#include "SiteIdentity.h"
#include "Support/CacheKey.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Targets/OffloadRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

#include <optional>
#include <system_error>

using namespace llvm;

namespace poseidon {

static std::map<std::string, Recorded> &recorded(LaunchManifest &L,
                                                 LaunchKind kind) {
  return kind == LaunchKind::LibraryCall ? L.LibraryCalls : L.Profiling;
}

static constexpr LaunchKind kKinds[] = {LaunchKind::LibraryCall,
                                        LaunchKind::Profiling};

static constexpr StringLiteral kProfileHeaders = "profile headers";
static constexpr StringLiteral kProfileTable = "profile table";

static StringRef kindName(LaunchKind kind) {
  return kind == LaunchKind::LibraryCall ? "library calls"
                                         : "profile generation";
}

/// The name both halves of one compile know their translation unit by: the one
/// -poseidon-compilation-id gives, the id clang gives both halves of a HIP
/// compile, or else the values of every Poseidon flag, which a compile hands
/// both halves alike.
static std::string unitName(const Options &O, const Module &M) {
  if (!O.Cache.CompilationId.empty())
    return O.Cache.CompilationId;
  if (std::optional<StringRef> cuid = compilationUnitId(M))
    return cuid->str();
  CacheKey options;
  for (const auto &[name, value] : allFlagValues(O))
    options.add(name, value);
  return options.digest();
}

namespace {

struct Manifest {
  SmallString<256> path;
  CacheKey key;
};

} // namespace

/// The manifest of M, in the -poseidon-launch-dir directory or else the
/// -poseidon-cache one, keyed by M's source file and unit name; none when
/// neither names a directory.
static std::optional<Manifest> manifestOf(const Options &O, const Module &M) {
  StringRef dir = O.Cache.LaunchDir.empty() ? O.Cache.Dir : O.Cache.LaunchDir;
  if (dir.empty())
    return std::nullopt;
  Manifest m;
  StringRef source = M.getSourceFileName();
  m.key.add("source file", source);
  m.key.add("unit", unitName(O, M));
  m.path = dir;
  sys::path::append(
      m.path, "launches_" + profileNameStem(sys::path::filename(source).str()) +
                  "_" + m.key.digest() + ".json");
  return m;
}

namespace {

/// A manifest read back: its path and its JSON object.
struct ReadManifest {
  SmallString<256> path;
  json::Object value;
};

} // namespace

/// A warning about the manifest at `path`: `why` it cannot be used, then
/// `unused`, what is not read from it.
static void warnUnusable(const Module &M, StringRef path, const Twine &why,
                         StringRef unused) {
  M.getContext().diagnose(DiagnosticInfoGeneric(
      Twine("Poseidon: the launch manifest ") + path + why + "; " + unused,
      DS_Warning));
}

/// The manifest of M read back; none when M has none, or when the one found is
/// unreadable or was written for another compile, which is a warning that ends
/// in `unused`, what is not read from it.
static std::optional<ReadManifest>
readManifest(const Options &O, const Module &M, StringRef unused) {
  std::optional<Manifest> m = manifestOf(O, M);
  if (!m)
    return std::nullopt;
  CacheEntryRead read = readCacheEntry(m->path, m->key);
  json::Object *value = read.value ? read.value->getAsObject() : nullptr;
  if (read.value && !value)
    read.unreadable = "it holds no JSON object";
  if (!read.unreadable.empty() || !read.differs.empty()) {
    warnUnusable(M, m->path,
                 read.unreadable.empty()
                     ? " was written for a different " + read.differs
                     : " is unreadable (" + read.unreadable + ")",
                 unused);
    return std::nullopt;
  }
  if (!value)
    return std::nullopt;
  return ReadManifest{m->path, std::move(*value)};
}

/// The error for a malformed `what` in the manifest at `path`.
static Error malformed(StringRef path, const Twine &what) {
  return userError("the launch manifest " + path + " holds " + what +
                   "; remove the -poseidon-cache directory and compile the "
                   "unit again");
}

bool handsOverLaunches(const Options &O) {
  return O.Profiling.generatesProfile() ||
         (O.Matmul.LibraryCalls && O.Profiling.usesProfile());
}

void recordDescriptor(LaunchManifest &L, LaunchKind kind, StringRef wrapperName,
                      function_ref<void(raw_ostream &)> emitPayload,
                      std::string message) {
  Recorded r;
  raw_string_ostream os(r.line);
  os << wrapperName;
  emitPayload(os);
  r.message = std::move(message);
  recorded(L, kind)[wrapperName.str()] = std::move(r);
}

void recordProfileHeader(LaunchManifest &L, StringRef wrapperName,
                         StringRef profileName, StringRef header) {
  L.ProfileHeaders[wrapperName.str()] = {profileName.str(), header.str()};
}

void recordProfileTable(LaunchManifest &L, unsigned functions, unsigned slots) {
  L.ProfiledFunctions = functions;
  L.ProfiledSlots = slots;
}

Expected<std::pair<unsigned, unsigned>> readProfileTable(const Options &O,
                                                         const Module &M) {
  std::optional<ReadManifest> m =
      readManifest(O, M, "its profile table is not read");
  const json::Value *entry = m ? m->value.get(kProfileTable) : nullptr;
  if (!entry)
    return std::make_pair(0u, 0u);
  const json::Object *table = entry->getAsObject();
  std::optional<int64_t> functions =
      table ? table->getInteger("functions") : std::nullopt;
  std::optional<int64_t> slots =
      table ? table->getInteger("slots") : std::nullopt;
  if (!functions || !slots || *functions < 0 || *slots < 0 ||
      *functions > UINT32_MAX || *slots > UINT32_MAX)
    return malformed(m->path, "a profile table entry that is not two counts");
  return std::make_pair((unsigned)*functions, (unsigned)*slots);
}

Error writeLaunchManifest(const Options &O, LaunchManifest &L,
                          const Module &M) {
  const bool any =
      L.ProfiledFunctions || llvm::any_of(kKinds, [&](LaunchKind kind) {
        return !recorded(L, kind).empty();
      });
  std::optional<Manifest> found = manifestOf(O, M);
  if (!found) {
    if (any)
      return userError("this device compile hands the launches of its sites "
                       "to its host half, and -poseidon-cache= names no "
                       "directory to keep them in; pass "
                       "-poseidon-launch-dir=<dir> to both halves");
    return Error::success();
  }
  const Manifest &m = *found;
  // The device code is already compiled for the rewritten stubs (a dispatched
  // product fissioned out of its kernel, a profiled kernel taking shadow
  // buffers), so a host half reading no manifest or an earlier one would build
  // a wrong program.
  auto cannot = [&](StringRef action, const Twine &why) {
    return userError("cannot " + action + " the launch manifest " + m.path +
                     ": " + why +
                     "; make the -poseidon-launch-dir directory, or else the "
                     "-poseidon-cache one, writable");
  };
  if (!any) {
    if (sys::fs::exists(m.path))
      if (std::error_code EC = sys::fs::remove(m.path))
        return cannot("remove", EC.message());
    return Error::success();
  }
  json::Object value;
  for (LaunchKind kind : kKinds) {
    json::Array lines;
    for (const auto &[kernel, r] : recorded(L, kind))
      lines.push_back(r.line);
    value[kindName(kind)] = std::move(lines);
  }
  // Only a profile-generate compile has any, and every other manifest stays
  // as it was.
  if (!L.ProfileHeaders.empty()) {
    json::Object headers;
    for (const auto &[kernel, h] : L.ProfileHeaders)
      headers[kernel] =
          json::Object{{"profile", h.profileName}, {"header", h.header}};
    value[kProfileHeaders] = std::move(headers);
  }
  if (L.ProfiledFunctions)
    value[kProfileTable] = json::Object{{"functions", L.ProfiledFunctions},
                                        {"slots", L.ProfiledSlots}};
  if (Error E = writeCacheEntry(m.path, m.key, std::move(value)))
    return cannot("write", toString(std::move(E)));
  for (LaunchKind kind : kKinds) {
    if (O.Diagnostics.Print)
      for (const auto &[kernel, r] : recorded(L, kind))
        errs() << r.message;
    recorded(L, kind).clear();
  }
  L.ProfileHeaders.clear();
  L.ProfiledFunctions = L.ProfiledSlots = 0;
  return Error::success();
}

Error readDescriptors(const Options &O, const Module &M, LaunchKind kind,
                      function_ref<Error(ArrayRef<StringRef>)> parse) {
  constexpr StringLiteral unused = "no launch stub is rewritten from it";
  std::optional<ReadManifest> m = readManifest(O, M, unused);
  if (!m)
    return Error::success();
  const json::Array *lines = m->value.getArray(kindName(kind));
  if (!lines) {
    warnUnusable(M, m->path,
                 " is unreadable (it holds no " + kindName(kind) + " list)",
                 unused);
    return Error::success();
  }
  for (const json::Value &entry : *lines) {
    std::optional<StringRef> line = entry.getAsString();
    if (!line)
      return malformed(m->path,
                       "a " + kindName(kind) + " entry that is not a string");
    SmallVector<StringRef, 8> toks;
    line->trim().split(toks, ' ');
    if (Error E = parse(toks))
      return malformed(m->path, "the " + kindName(kind) + " descriptor '" +
                                    *line + "', which does not parse (" +
                                    toString(std::move(E)) + ")");
  }
  return Error::success();
}

Error readProfileHeaders(const Options &O, const Module &M,
                         function_ref<void(StringRef, StringRef)> use) {
  std::optional<ReadManifest> m =
      readManifest(O, M, "no profile header is registered from it");
  const json::Value *entry = m ? m->value.get(kProfileHeaders) : nullptr;
  if (!entry)
    return Error::success();
  const json::Object *headers = entry->getAsObject();
  if (!headers)
    return malformed(m->path, "profile headers that are not an object");
  // In kernel order: a JSON object's own order is its hash table's.
  std::map<std::string, const json::Object *> byKernel;
  for (const auto &[kernel, h] : *headers)
    byKernel[kernel.str()] = h.getAsObject();
  for (const auto &[kernel, h] : byKernel) {
    std::optional<StringRef> name = h ? h->getString("profile") : std::nullopt;
    std::optional<StringRef> header = h ? h->getString("header") : std::nullopt;
    if (!name || !header)
      return malformed(m->path, "a profile headers entry of " + kernel +
                                    " that is not a profile name and header");
    use(*name, *header);
  }
  return Error::success();
}

} // namespace poseidon

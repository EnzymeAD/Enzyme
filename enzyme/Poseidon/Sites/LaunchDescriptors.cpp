//=- LaunchDescriptors.cpp - launch-stub descriptor plumbing --------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LaunchDescriptors.h"
#include "Core/CacheKey.h"
#include "FPProfileName.h"
#include "Pass/Session.h"
#include "Targets/OffloadRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace poseidon {

namespace {

std::map<std::string, Recorded> &recorded(Session &S, LaunchKind kind) {
  return S.RecordedLaunches[kind == LaunchKind::HostDispatch ? 0 : 1];
}

constexpr LaunchKind kKinds[] = {LaunchKind::HostDispatch,
                                 LaunchKind::Profiling};

constexpr StringLiteral kProfileHeaders = "profile headers";

StringRef kindName(LaunchKind kind) {
  return kind == LaunchKind::HostDispatch ? "host dispatch"
                                          : "profile generation";
}

// The name both halves of one compile know their translation unit by: the one
// -poseidon-unit gives, the id clang gives both halves of a HIP compile, or
// else the values of every Poseidon flag, which a compile hands both halves
// alike.
std::string unitName(const Options &O, const Module &M) {
  if (!O.Herbie.Unit.empty())
    return O.Herbie.Unit;
  if (std::optional<StringRef> cuid = compilationUnitId(M))
    return cuid->str();
  CacheKey options;
  for (const auto &[name, value] : allFlagValues(O))
    options.add(name, value);
  return options.digest();
}

struct Manifest {
  SmallString<256> path;
  CacheKey key;
};

// The source file tells apart the units one compile writes into a cache, the
// unit name the compiles of one source into it.
Manifest manifestOf(const Options &O, const Module &M) {
  Manifest m;
  StringRef source = M.getSourceFileName();
  m.key.add("source file", source);
  m.key.add("unit", unitName(O, M));
  m.path = O.Herbie.Cache;
  sys::path::append(
      m.path, "launches_" + profileNameStem(sys::path::filename(source).str()) +
                  "_" + m.key.digest() + ".json");
  return m;
}

} // namespace

bool handsOverLaunches(const Options &O) {
  return O.Profiling.ProfileGenerate ||
         (O.Matmul.OzakiHostDispatch && O.Profiling.usesProfile());
}

void recordDescriptor(Session &S, LaunchKind kind, StringRef wrapperName,
                      function_ref<void(raw_ostream &)> emitPayload,
                      std::string message) {
  Recorded r;
  raw_string_ostream os(r.line);
  os << wrapperName;
  emitPayload(os);
  r.message = std::move(message);
  recorded(S, kind)[wrapperName.str()] = std::move(r);
}

void recordProfileHeader(Session &S, StringRef wrapperName,
                         StringRef profileName, StringRef header) {
  S.ProfileHeaders[wrapperName.str()] = {profileName.str(), header.str()};
}

void writeLaunchManifest(Session &S, const Module &M) {
  const Options &O = S.Opts;
  Manifest m = manifestOf(O, M);
  // The device code is already compiled for the rewritten stubs (a dispatched
  // product fissioned out of its kernel, a profiled kernel taking shadow
  // buffers), so a host half reading no manifest or an earlier one would build
  // a wrong program.
  auto fail = [&](StringRef action, const Twine &why) {
    report_fatal_error("Poseidon: cannot " + action + " the launch manifest " +
                           m.path + ": " + why,
                       /*gen_crash_diag=*/false);
  };
  if (llvm::all_of(
          kKinds, [&](LaunchKind kind) { return recorded(S, kind).empty(); })) {
    if (sys::fs::exists(m.path))
      if (std::error_code EC = sys::fs::remove(m.path))
        fail("remove", EC.message());
    return;
  }
  json::Object value;
  for (LaunchKind kind : kKinds) {
    json::Array lines;
    for (const auto &[kernel, r] : recorded(S, kind))
      lines.push_back(r.line);
    value[kindName(kind)] = std::move(lines);
  }
  // Only a profile-generate compile has any, and every other manifest stays
  // as it was.
  if (!S.ProfileHeaders.empty()) {
    json::Object headers;
    for (const auto &[kernel, h] : S.ProfileHeaders)
      headers[kernel] =
          json::Object{{"profile", h.profileName}, {"header", h.header}};
    value[kProfileHeaders] = std::move(headers);
  }
  if (std::error_code EC = sys::fs::create_directories(O.Herbie.Cache))
    fail("write", EC.message());
  if (Error E = writeCacheEntry(m.path, m.key, std::move(value)))
    fail("write", toString(std::move(E)));
  for (LaunchKind kind : kKinds) {
    for (const auto &[kernel, r] : recorded(S, kind))
      errs() << r.message;
    recorded(S, kind).clear();
  }
  S.ProfileHeaders.clear();
}

void readDescriptors(const Options &O, const Module &M, LaunchKind kind,
                     function_ref<void(ArrayRef<StringRef>)> parse) {
  Manifest m = manifestOf(O, M);
  CacheEntryRead read = readCacheEntry(m.path, m.key);
  const json::Object *value = read.value ? read.value->getAsObject() : nullptr;
  const json::Array *lines = value ? value->getArray(kindName(kind)) : nullptr;
  if (read.value && !lines)
    read.unreadable = ("it holds no " + kindName(kind) + " list").str();
  if (!read.unreadable.empty())
    errs() << "[poseidon] launch manifest " << m.path << " is unreadable ("
           << read.unreadable << "); no launch stub is rewritten from it\n";
  else if (!read.differs.empty())
    errs() << "[poseidon] launch manifest " << m.path
           << " was written for a different " << read.differs
           << "; no launch stub is rewritten from it\n";
  if (!lines)
    return;
  for (const json::Value &entry : *lines) {
    std::optional<StringRef> line = entry.getAsString();
    if (!line)
      continue;
    SmallVector<StringRef, 8> toks;
    line->trim().split(toks, ' ');
    parse(toks);
  }
}

void readProfileHeaders(const Options &O, const Module &M,
                        function_ref<void(StringRef, StringRef)> use) {
  Manifest m = manifestOf(O, M);
  CacheEntryRead read = readCacheEntry(m.path, m.key);
  const json::Object *value = read.value ? read.value->getAsObject() : nullptr;
  const json::Object *headers =
      value ? value->getObject(kProfileHeaders) : nullptr;
  if (!headers)
    return;
  // In kernel order: a JSON object's own order is its hash table's.
  std::map<std::string, const json::Object *> byKernel;
  for (const auto &[kernel, entry] : *headers)
    byKernel[kernel.str()] = entry.getAsObject();
  for (const auto &[kernel, h] : byKernel) {
    std::optional<StringRef> name = h ? h->getString("profile") : std::nullopt;
    std::optional<StringRef> header = h ? h->getString("header") : std::nullopt;
    if (name && header)
      use(*name, *header);
  }
}

StringRef mangledSuffix(StringRef mangled) {
  if (mangled.starts_with("_Z")) {
    mangled = mangled.drop_front(2);
    // An internal-linkage name (a static kernel template, as MFEM's are).
    mangled.consume_front("L");
    while (!mangled.empty() && mangled[0] >= '0' && mangled[0] <= '9')
      mangled = mangled.drop_front(1);
  }
  return mangled;
}

bool isLaunchStubFor(const Function &F, StringRef kernelName) {
  StringRef n = F.getName();
  if (n == kernelName)
    return true;
  const StringRef stubPrefix = launchStubPrefix();
  if (!n.contains(stubPrefix))
    return false;
  StringRef kernel = withoutExternalizedPostfix(kernelName);
  if (n.ends_with(mangledSuffix(kernel)))
    return true;
  // A kernel in a namespace or a class: its identifier, and with it the
  // stub's prefix, sits inside the nested name rather than at its front.
  std::string stub = demangle(n);
  size_t at = stub.find(stubPrefix);
  if (at == std::string::npos || stub == n)
    return false;
  stub.erase(at, stubPrefix.size());
  return stub == demangle(kernel);
}

} // namespace poseidon

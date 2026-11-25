//===- LaunchDescriptors.cpp - launch-stub descriptor plumbing ------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LaunchDescriptors.h"
#include "Core/CacheKey.h"
#include "Enzyme/Utils.h"
#include "IR/Utils.h"
#include "Pass/Session.h"
#include "SiteIdentity.h"
#include "Targets/OffloadRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

#include <optional>
#include <system_error>

using namespace llvm;

namespace poseidon {

static std::map<std::string, Recorded> &recorded(Session &S, LaunchKind kind) {
  return S.RecordedLaunches[static_cast<unsigned>(kind)];
}

static constexpr LaunchKind kKinds[] = {LaunchKind::HostDispatch,
                                        LaunchKind::Profiling};

static constexpr StringLiteral kProfileHeaders = "profile headers";
static constexpr StringLiteral kProfileTable = "profile table";

static StringRef kindName(LaunchKind kind) {
  return kind == LaunchKind::HostDispatch ? "host dispatch"
                                          : "profile generation";
}

/// The name both halves of one compile know their translation unit by: the one
/// -poseidon-unit gives, the id clang gives both halves of a HIP compile, or
/// else the values of every Poseidon flag, which a compile hands both halves
/// alike.
static std::string unitName(const Options &O, const Module &M) {
  if (!O.Herbie.Unit.empty())
    return O.Herbie.Unit;
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

/// The source file tells apart the units one compile writes into a cache, the
/// unit name the compiles of one source into it.
/// The manifest of M, in the -poseidon-launch-dir directory or else the
/// -poseidon-cache one; none when neither names a directory.
static std::optional<Manifest> manifestOf(const Options &O, const Module &M) {
  StringRef dir =
      O.Herbie.LaunchDir.empty() ? O.Herbie.Cache : O.Herbie.LaunchDir;
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

bool handsOverLaunches(const Options &O) {
  return O.Profiling.generatesProfile() ||
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

void recordProfileTable(Session &S, unsigned functions, unsigned slots) {
  S.ProfiledFunctions = functions;
  S.ProfiledSlots = slots;
}

std::pair<unsigned, unsigned> readProfileTable(const Options &O,
                                               const Module &M) {
  std::optional<Manifest> m = manifestOf(O, M);
  if (!m)
    return {0, 0};
  CacheEntryRead read = readCacheEntry(m->path, m->key);
  const json::Object *value = read.value ? read.value->getAsObject() : nullptr;
  const json::Object *table = value ? value->getObject(kProfileTable) : nullptr;
  if (!table)
    return {0, 0};
  std::optional<int64_t> functions = table->getInteger("functions");
  std::optional<int64_t> slots = table->getInteger("slots");
  if (!functions || !slots || *functions < 0 || *slots < 0 ||
      *functions > UINT32_MAX || *slots > UINT32_MAX)
    reportUserError(M.getContext(),
                    "the launch manifest " + m->path +
                        " holds a profile table entry that is not two counts; "
                        "remove the -poseidon-cache directory and compile the "
                        "unit again");
  return {(unsigned)*functions, (unsigned)*slots};
}

void writeLaunchManifest(Session &S, const Module &M) {
  const Options &O = S.Opts;
  const bool any =
      S.ProfiledFunctions || llvm::any_of(kKinds, [&](LaunchKind kind) {
        return !recorded(S, kind).empty();
      });
  std::optional<Manifest> found = manifestOf(O, M);
  if (!found) {
    if (any)
      reportUserError(M.getContext(),
                      "this device compile hands the launches of its sites to "
                      "its host half, and -poseidon-cache= names no directory "
                      "to keep them in; pass -poseidon-launch-dir=<dir> to "
                      "both halves");
    return;
  }
  const Manifest &m = *found;
  // The device code is already compiled for the rewritten stubs (a dispatched
  // product fissioned out of its kernel, a profiled kernel taking shadow
  // buffers), so a host half reading no manifest or an earlier one would build
  // a wrong program.
  auto fail = [&](StringRef action, const Twine &why) {
    reportUserError(M.getContext(),
                    "cannot " + action + " the launch manifest " + m.path +
                        ": " + why +
                        ". The manifest is kept in the -poseidon-launch-dir "
                        "directory, or else the -poseidon-cache one, which "
                        "has to be writable.");
  };
  if (!any) {
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
  if (S.ProfiledFunctions)
    value[kProfileTable] = json::Object{{"functions", S.ProfiledFunctions},
                                        {"slots", S.ProfiledSlots}};
  if (std::error_code EC =
          sys::fs::create_directories(sys::path::parent_path(m.path)))
    fail("write", EC.message());
  if (Error E = writeCacheEntry(m.path, m.key, std::move(value)))
    fail("write", toString(std::move(E)));
  for (LaunchKind kind : kKinds) {
    for (const auto &[kernel, r] : recorded(S, kind))
      errs() << r.message;
    recorded(S, kind).clear();
  }
  S.ProfileHeaders.clear();
  S.ProfiledFunctions = S.ProfiledSlots = 0;
}

void readDescriptors(const Options &O, const Module &M, LaunchKind kind,
                     function_ref<Error(ArrayRef<StringRef>)> parse) {
  std::optional<Manifest> found = manifestOf(O, M);
  if (!found)
    return;
  const Manifest &m = *found;
  CacheEntryRead read = readCacheEntry(m.path, m.key);
  const json::Object *value = read.value ? read.value->getAsObject() : nullptr;
  const json::Array *lines = value ? value->getArray(kindName(kind)) : nullptr;
  if (read.value && !lines)
    read.unreadable = ("it holds no " + kindName(kind) + " list").str();
  if (!read.unreadable.empty() || !read.differs.empty())
    M.getContext().diagnose(DiagnosticInfoGeneric(
        Twine("Poseidon: the launch manifest ") + m.path +
            (read.unreadable.empty()
                 ? " was written for a different " + read.differs
                 : " is unreadable (" + read.unreadable + ")") +
            "; no launch stub is rewritten from it",
        DS_Warning));
  if (!lines)
    return;
  for (const json::Value &entry : *lines) {
    std::optional<StringRef> line = entry.getAsString();
    if (!line)
      reportUserError(M.getContext(),
                      "the launch manifest " + m.path + " holds a " +
                          kindName(kind) +
                          " entry that is not a string; remove the "
                          "-poseidon-cache directory and compile the unit "
                          "again");
    SmallVector<StringRef, 8> toks;
    line->trim().split(toks, ' ');
    if (Error E = parse(toks))
      reportUserError(M.getContext(),
                      "the launch manifest " + m.path + " holds the " +
                          kindName(kind) + " descriptor '" + *line +
                          "', which does not parse: " + toString(std::move(E)) +
                          ". The device compile of this unit writes it into "
                          "the -poseidon-cache directory; remove that "
                          "directory and compile the unit again.");
  }
}

void readProfileHeaders(const Options &O, const Module &M,
                        function_ref<void(StringRef, StringRef)> use) {
  std::optional<Manifest> m = manifestOf(O, M);
  if (!m)
    return;
  CacheEntryRead read = readCacheEntry(m->path, m->key);
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

int traceToArgIndex(const Value *v) {
  const auto *A = dyn_cast<Argument>(getUnderlyingObject(v, /*MaxLookup=*/0));
  return A ? (int)A->getArgNo() : -1;
}

static bool isByValueParam(const Argument &A) {
  return A.hasByValAttr() || A.getParent()->getAttributes().hasParamAttr(
                                 A.getArgNo(), kByValueParam);
}

/// Whether nothing the body does writes the memory `A` points to. The readonly
/// attribute is not enough: under -fgpu-rdc, which profile generation adds, a
/// kernel is weak_odr and nothing infers it.
static bool onlyRead(const Argument &A) {
  if (isReadOnly(A.getParent(), A.getArgNo()))
    return true;
  SmallVector<const Value *, 8> work{&A};
  SmallPtrSet<const Value *, 8> seen{&A};
  while (!work.empty()) {
    const Value *v = work.pop_back_val();
    for (const Use &U : v->uses()) {
      const User *user = U.getUser();
      if (isa<GetElementPtrInst>(user) || isa<BitCastInst>(user) ||
          isa<AddrSpaceCastInst>(user)) {
        if (seen.insert(user).second)
          work.push_back(user);
        continue;
      }
      if (auto *L = dyn_cast<LoadInst>(user)) {
        if (!L->isVolatile())
          continue;
        return false;
      }
      if (auto *CB = dyn_cast<CallBase>(user))
        if (CB->isArgOperand(&U) && isReadOnly(CB, CB->getArgOperandNo(&U)) &&
            isNoCapture(CB, CB->getArgOperandNo(&U)))
          continue;
      return false;
    }
  }
  return true;
}

std::string launchRefString(const LaunchRef &r, StringRef prefix) {
  std::string s = (prefix + Twine(r.param)).str();
  if (r.isField())
    s += "[+" + std::to_string(r.offset) + "]";
  return s;
}

LaunchRef launchFieldOf(const Value *v) {
  if (auto *L = dyn_cast<LoadInst>(v)) {
    if (!L->isSimple())
      return {};
    const DataLayout &DL = L->getModule()->getDataLayout();
    APInt off(DL.getIndexTypeSizeInBits(L->getPointerOperandType()), 0);
    const Value *base =
        L->getPointerOperand()->stripAndAccumulateConstantOffsets(
            DL, off, /*AllowNonInbounds=*/true);
    // A field the body may overwrite is not the value the launch passed.
    auto *A = dyn_cast<Argument>(base);
    if (!A || !isByValueParam(*A) || !onlyRead(*A) || off.isNegative())
      return {};
    return {(int)A->getArgNo(), off.getSExtValue()};
  }
  if (auto *EV = dyn_cast<ExtractValueInst>(v)) {
    auto *A = dyn_cast<Argument>(EV->getAggregateOperand());
    if (!A)
      return {};
    uint64_t off = getAggregateElementOffset(EV->getModule()->getDataLayout(),
                                             A->getType(), EV->getIndices());
    return {(int)A->getArgNo(), (int64_t)off};
  }
  return {};
}

LaunchRef tracePointerRoot(const Value *v) {
  const Value *root = getUnderlyingObject(v, /*MaxLookup=*/0);
  if (const auto *A = dyn_cast<Argument>(root))
    return isByValueParam(*A) ? LaunchRef{} : LaunchRef{(int)A->getArgNo(), -1};
  return root->getType()->isPointerTy() ? launchFieldOf(root) : LaunchRef{};
}

} // namespace poseidon

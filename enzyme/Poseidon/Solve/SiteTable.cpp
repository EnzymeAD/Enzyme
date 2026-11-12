//=- SiteTable.cpp - a site's DP table in the cache -------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SiteTable.h"
#include "IR/Utils.h"
#include "Model/CostModel.h"
#include "Optimize.h"
#include "Pass/Session.h"
#include "Sites/SiteIdentity.h"
#include "Solvers.h"
#include "Techniques/Herbie.h"
#include "Techniques/Matmul/MatmulSimulation.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#if __has_include("llvm/Support/VCSRevision.h")
#include "llvm/Support/VCSRevision.h"
#endif

#include "BuildIdentity.h"

#include <map>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

// Pricing is also where candidates are dropped, and a table hit skips pricing,
// so the hit sees every candidate the table's positions were taken without; a
// cached step names its candidate as well as its position.
static std::string candidateKey(StringRef text) {
  return utohexstr(xxh3_64bits(text));
}

static std::optional<size_t> findCandidate(const SiteUnit &unit, size_t hint,
                                           StringRef key) {
  if (hint < unit.numCandidates() &&
      candidateKey(unit.candidateName(hint)) == key)
    return hint;
  std::optional<size_t> found;
  for (size_t i = 0; i < unit.numCandidates(); ++i) {
    if (candidateKey(unit.candidateName(i)) != key)
      continue;
    if (found)
      return std::nullopt;
    found = i;
  }
  return found;
}

static std::string tablePath(const Options &O) {
  return O.Herbie.Cache + "/table.json";
}

namespace {
struct StoredStep {
  std::string itemType, candidateKey;
  size_t itemIndex = 0, candidateIndex = 0;
};
bool fromJSON(const json::Value &V, StoredStep &S, json::Path P) {
  json::ObjectMapper O(V, P);
  return O && O.map("itemType", S.itemType) &&
         O.map("itemIndex", S.itemIndex) &&
         O.map("candidateIndex", S.candidateIndex) &&
         O.mapOptional("candidateKey", S.candidateKey);
}
} // namespace

// Everything a site's DP table is computed from, the code that prices and
// solves it included. The candidates are part of it although the inputs before
// them decide them: a Herbie search can end differently under the same inputs
// (a timeout), and under an accuracy target the matrix products are left out
// of the table.
CacheKey solveKey(Session &S, const FunctionFPState &st,
                  ArrayRef<SiteUnit *> units, StringRef profilePath) {
  const Options &O = S.Opts;
  const Function &F = *st.F;
  CacheKey key;
  // Two LLVM snapshots report one version string; the revision, which an LLVM
  // built without it does not define, tells them apart.
  std::string compiler =
      std::string(kPassIdentity) + " LLVM " + LLVM_VERSION_STRING;
#ifdef LLVM_REVISION
  compiler += " " LLVM_REVISION;
#endif
  key.add("compiler", compiler);
  std::string body;
  raw_string_ostream bodyOS(body);
  bodyOS << StructuralHash(F, /*DetailedHash=*/true);
  // StructuralHash does not see fast-math flags, and they change what a
  // rewrite costs.
  for (const Instruction &I : instructions(F))
    if (isa<FPMathOperator>(I)) {
      bodyOS << '|';
      I.getFastMathFlags().print(bodyOS);
    }
  key.addContent("function body", body);
  key.add("device", getCostModelNativeArch(S.CostModel));
  // Two builds of one source can link different TCEC backends.
  key.add("TCEC backend", tcecBackendName(kLinkedTcecBackend));
  key.add("cost model", utohexstr(getCostModelFingerprint(S.CostModel)));
  key.add("cost axis", "cost-model units times 2^(32 - log2 profiled cost)");

  SmallVector<std::string, 4> profilePaths = {profilePath.str()};
  for (const std::string &dir : O.Profiling.OutputProfiles) {
    SmallString<128> path(dir);
    sys::path::append(path, siteProfileStem(S, F) + ".fpprofile");
    profilePaths.push_back(path.str().str());
  }
  std::string profiles;
  for (const std::string &path : profilePaths) {
    auto buf = MemoryBuffer::getFile(path);
    if (!buf)
      reportUserError(F, F.getSubprogram(),
                      Twine("cannot read the profile ") + path +
                          " again to key the DP table of " + F.getName() +
                          " (" + buf.getError().message() +
                          "); a profile directory must not change while a "
                          "compile reads it.");
    profiles += contentDigest((*buf)->getBuffer()) + "\n";
  }
  key.addContent("profile", profiles);

  for (const auto &[name, value] : solveFlagValues(O))
    key.add(("-" + name).str(), value);
  // Only a site that sent Herbie cores depends on which Herbie answered them.
  if (!st.COs.empty())
    addHerbieKeyParts(S, F, key);

  std::string candidates;
  raw_string_ostream os(candidates);
  for (const SiteUnit *unit : units) {
    os << unit->keyHeader() << "\n";
    for (size_t i = 0; i < unit->numCandidates(); ++i)
      os << "  " << unit->candidateName(i) << "\n";
  }
  key.addContent("candidate set", candidates);
  return key;
}

// Fills `table` from the cache when it holds a table for `fnName` under
// table.key, and otherwise says why not unless there is none at all.
bool readSiteTable(const Options &O, StringRef fnName, SiteTable &table,
                   ArrayRef<SiteUnit *> units) {
  const std::string path = tablePath(O);
  if (!sys::fs::exists(path))
    return false;
  auto miss = [&](const std::string &why) {
    llvm::errs() << "[poseidon] DP cache entry for " << fnName << " " << why
                 << "; re-solving.\n";
    return false;
  };
  auto unreadable = [&](const std::string &why) {
    return miss("is unreadable (" + path + ": " + why + ")");
  };

  auto buf = MemoryBuffer::getFile(path);
  if (!buf)
    return unreadable(buf.getError().message());
  Expected<json::Value> root = json::parse((*buf)->getBuffer());
  if (!root)
    return unreadable(toString(root.takeError()));
  json::Object *funcs = root->getAsObject()
                            ? root->getAsObject()->getObject("functions")
                            : nullptr;
  if (!funcs)
    return miss("was written in another cache format");
  json::Object *entry = funcs->getObject(fnName);
  if (!entry)
    return false;
  if (std::optional<std::string> part =
          table.key.firstDifference(entry->getObject("key")))
    return miss(*part == "format" ? "was written in another cache format"
                                  : "was solved for a different " + *part);

  std::map<std::string, double> accMap;
  std::map<std::string, std::vector<StoredStep>> solMap;
  json::Path::Root parsed;
  const json::Value *accVal = entry->get("costToAccuracyMap");
  const json::Value *solVal = entry->get("costToSolutionMap");
  if (!accVal || !solVal || !fromJSON(*accVal, accMap, parsed) ||
      !fromJSON(*solVal, solMap, parsed))
    return unreadable("its table does not parse");
  auto cost = [](StringRef text) -> std::optional<DPCost> {
    int64_t c;
    if (text.getAsInteger(10, c))
      return std::nullopt;
    return c;
  };
  // Steps name their candidates by key: pricing drops candidates and a hit
  // prices nothing, so a solved table's positions are not the replay's.
  auto bind = [&](const StoredStep &s) -> std::optional<SolutionStep> {
    for (SiteUnit *unit : units) {
      if (unit->info().tag != s.itemType || unit->indexInKind() != s.itemIndex)
        continue;
      if (!unit->info().pricingDrops) {
        if (s.candidateIndex < unit->numCandidates())
          return SolutionStep(unit, s.candidateIndex);
        return std::nullopt;
      }
      if (auto i = findCandidate(*unit, s.candidateIndex, s.candidateKey))
        return SolutionStep(unit, *i);
      return std::nullopt;
    }
    return std::nullopt;
  };
  for (const auto &[text, acc] : accMap) {
    std::optional<DPCost> c = cost(text);
    if (!c)
      return unreadable("its table does not parse");
    table.costToAccuracyMap[*c] = acc;
  }
  for (const auto &[text, steps] : solMap) {
    std::optional<DPCost> c = cost(text);
    if (!c)
      return unreadable("its table does not parse");
    SmallVector<SolutionStep> &bound = table.costToSolutionMap[*c];
    for (const StoredStep &s : steps) {
      std::optional<SolutionStep> step = bind(s);
      if (!step)
        return unreadable("a step names no candidate of this compile");
      bound.push_back(*step);
    }
  }
  table.baseline = entry->getNumber("baselineAccCost");
  table.hit = true;
  return true;
}

void writeSiteTable(const Options &O, StringRef fnName, const CacheKey &key,
                    const CostMap &costToAccuracyMap,
                    const SolutionMap &costToSolutionMap, double baseline,
                    double profiledCost) {
  // llvm::json renders nan/inf as a bare `nan` token, which is not JSON. Refuse
  // to cache this function rather than drop entries: the loader treats a table
  // as a complete frontier, so a pruned one would change a later solve's
  // answer.
  unsigned nonFinite = 0;
  for (const auto &pair : costToAccuracyMap)
    if (!std::isfinite(pair.second))
      ++nonFinite;
  if (nonFinite) {
    llvm::errs()
        << "[poseidon] NOT caching the DP table for " << fnName << ": "
        << nonFinite << " of " << costToAccuracyMap.size()
        << " frontier points have a non-finite accuracy cost, so the "
           "table is not serializable and, more to the point, not "
           "authoritative. This function will be re-solved on the next "
           "compile; every other function's table is untouched.\n";
    return;
  }

  json::Object entryObj;
  entryObj["key"] = key.toJSON();

  json::Object costAccMap;
  for (const auto &pair : costToAccuracyMap) {
    costAccMap[std::to_string(pair.first)] = pair.second;
  }
  entryObj["costToAccuracyMap"] = std::move(costAccMap);

  json::Object costSolMap;
  for (const auto &pair : costToSolutionMap) {
    json::Array stepsArray;
    for (const auto &step : pair.second) {
      const SiteUnit &unit = unitOf(step);
      json::Object stepObj;
      stepObj["candidateIndex"] = static_cast<int64_t>(step.candidateIndex);
      stepObj["itemType"] = unit.info().tag;
      stepObj["itemIndex"] = static_cast<int64_t>(unit.indexInKind());
      if (unit.info().pricingDrops)
        stepObj["candidateKey"] =
            candidateKey(unit.candidateName(step.candidateIndex));
      stepsArray.push_back(std::move(stepObj));
    }
    costSolMap[std::to_string(pair.first)] = std::move(stepsArray);
  }
  entryObj["costToSolutionMap"] = std::move(costSolMap);
  if (std::isfinite(baseline))
    entryObj["baselineAccCost"] = baseline;
  entryObj["profiledCost"] = profiledCost;

  // Read-modify-write: the file holds one table per function, so a later site
  // in the same compilation unit must extend it rather than clobber the
  // earlier sites' tables. A file that does not parse is replaced.
  const std::string path = tablePath(O);
  json::Object functions;
  if (auto fileOrErr = MemoryBuffer::getFile(path)) {
    if (auto prior = json::parse(fileOrErr.get()->getBuffer())) {
      if (json::Object *priorObj = prior->getAsObject())
        if (json::Object *priorFuncs = priorObj->getObject("functions"))
          functions = std::move(*priorFuncs);
    } else {
      consumeError(prior.takeError());
    }
  }
  functions[fnName] = std::move(entryObj);

  json::Object jsonObj;
  jsonObj["functions"] = std::move(functions);

  if (Error E = writeFileAtomically(
          path,
          formatv("{0:2}", json::Value(std::move(jsonObj))).str() + "\n")) {
    llvm::errs() << "Error writing cache file: " << toString(std::move(E))
                 << "\n";
    return;
  }
  llvm::errs() << "DP table for " << fnName << " cached to file.\n";
}

} // namespace poseidon

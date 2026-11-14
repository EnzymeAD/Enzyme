//=- CostTable.cpp - the measured cost table of one compile ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CostTable.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <system_error>
#include <tuple>

using namespace llvm;

namespace poseidon {

// The CSV this compilation prices from, resolved once: the flag, the
// environment, then this user's cache. No model ships with Poseidon: a model is
// one device's measurement. Everything below reads it through costModelPath(),
// never the flag.
static bool resolveFromRequest(CostModelState &CM) {
  if (CM.IsResolved)
    return true;
  if (!CM.Requested.empty()) {
    CM.Resolved = CM.Requested;
    CM.IsResolved = true;
  } else if (const char *Env = std::getenv("POSEIDON_COST_MODEL")) {
    if (*Env) {
      CM.Resolved = Env;
      CM.IsResolved = true;
    }
  }
  return CM.IsResolved;
}

const std::string &costModelPath(CostModelState &CM) {
  resolveFromRequest(CM);
  return CM.Resolved;
}

static std::string userCacheDir() {
  if (const char *X = std::getenv("XDG_CACHE_HOME"))
    if (*X)
      return std::string(X) + "/poseidon";
  if (const char *H = std::getenv("HOME"))
    if (*H)
      return std::string(H) + "/.cache/poseidon";
  return std::string();
}

// One measured model per device: the file name carries the architecture it was
// measured on, so the module's target-cpu names it.
static Error resolveForTarget(CostModelState &CM, StringRef TargetCpu,
                              StringRef FnName) {
  if (resolveFromRequest(CM))
    return Error::success();
  if (TargetCpu.empty())
    return createStringError(inconvertibleErrorCode(),
                             "optimizing " + Twine(FnName) +
                                 " needs a measured cost model, and the module "
                                 "names no target-cpu to look one up by; pass "
                                 "-poseidon-cost-model=<csv>");
  std::string Dir = userCacheDir();
  std::string Prefix = ("cm_" + TargetCpu + "_").str();
  SmallVector<std::string, 4> Found;
  std::error_code EC;
  for (sys::fs::directory_iterator It(Dir, EC), E; It != E && !EC;
       It.increment(EC)) {
    StringRef Name = sys::path::filename(It->path());
    if (Name.starts_with(Prefix) && Name.ends_with(".csv"))
      Found.push_back(It->path());
  }
  if (Found.empty())
    return createStringError(
        inconvertibleErrorCode(),
        "no cost model for target-cpu '" + Twine(TargetCpu) + "' in " +
            (Dir.empty()
                 ? "the user cache, which neither XDG_CACHE_HOME nor HOME "
                   "locates"
                 : Dir) +
            "; measure this device with poseidon-calibrate, or pass "
            "-poseidon-cost-model=<csv>");
  if (Found.size() > 1) {
    std::string List;
    for (const std::string &P : Found)
      List += (List.empty() ? "" : ", ") + P;
    return createStringError(inconvertibleErrorCode(),
                             "several cost models for target-cpu '" +
                                 Twine(TargetCpu) + "': " + List +
                                 "; name the one to price from with "
                                 "-poseidon-cost-model=<csv>");
  }
  CM.Resolved = Found.front();
  CM.IsResolved = true;
  return Error::success();
}

// A price comes from the model or not at all: a part of the model that is
// missing or unreadable stops the compile, naming the model.
static Error modelError(CostModelState &CM, const Twine &What) {
  return createStringError(inconvertibleErrorCode(),
                           "cost model " + Twine(costModelPath(CM)) + " " +
                               What);
}

static Error loadCostModel(CostModelState &CM,
                           function_ref<bool(StringRef, StringRef)> IsRow) {
  if (CM.Loaded)
    return Error::success();
  std::ifstream CostFile(costModelPath(CM));
  if (!CostFile.is_open())
    return modelError(CM, "cannot be opened");
  std::string Line;
  unsigned LineNo = 0;
  while (std::getline(CostFile, Line)) {
    ++LineNo;
    if (Line.empty())
      continue;
    if (Line[0] == '#') {
      auto content = StringRef(Line).drop_front(1).trim();
      StringRef key, vals;
      std::tie(key, vals) = content.split('=');
      key = key.trim();
      vals = vals.trim();
      if (key == "native_arch") {
        CM.NativeArch = vals.str();
        continue;
      }
      if (key == "scalar_types") {
        SmallVector<StringRef, 8> tokens;
        vals.split(tokens, ',', -1, false);
        for (auto t : tokens)
          CM.ScalarTypes.insert(t.str());
      }
      continue;
    }
    SmallVector<StringRef, 3> Fields;
    StringRef(Line).split(Fields, ',');
    double Cost;
    if (Fields.size() != 3 || Fields[2].trim().getAsDouble(Cost))
      return modelError(CM, "line " + Twine(LineNo) +
                                " is not '<opcode>,<type>,<cost>': " + Line);
    if (!IsRow(Fields[0], Fields[1]))
      return modelError(CM, "line " + Twine(LineNo) + " has the row '" +
                                Fields[0] + "," + Fields[1] +
                                "', of no row family Poseidon reads (the "
                                "calibrator's README lists them); remove the "
                                "line, or compile with the Poseidon whose "
                                "poseidon-calibrate wrote it");
    if (!CM.Model.try_emplace({Fields[0].str(), Fields[1].str()}, Cost).second)
      return modelError(CM, "has the row '" + Fields[0] + "," + Fields[1] +
                                "' twice");
  }
  CM.Loaded = true;
  return Error::success();
}

const std::map<std::pair<std::string, std::string>, double> &
getCostModel(CostModelState &CM) {
  if (!CM.Loaded)
    report_fatal_error("Poseidon: the cost model is read before a site "
                       "required it");
  return CM.Model;
}

// Hashes the parsed map rather than the file bytes so comments and row order
// stay out of the DP-cache key; a changed price invalidates the cached table.
uint64_t getCostModelFingerprint(CostModelState &CM) {
  uint64_t &FP = CM.Fingerprint;
  if (CM.FingerprintComputed)
    return FP;
  CM.FingerprintComputed = true;
  if (costModelPath(CM).empty())
    return FP;
  const auto &Model = getCostModel(CM);
  uint64_t h = 1469598103934665603ULL;
  auto mix = [&](uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  auto mixStr = [&](const std::string &s) {
    for (unsigned char c : s)
      mix((uint64_t)c);
    mix(0x5eULL);
  };
  for (const auto &kv : Model) { // std::map => deterministic order
    mixStr(kv.first.first);
    mixStr(kv.first.second);
    uint64_t bits;
    double d = kv.second;
    std::memcpy(&bits, &d, sizeof(bits));
    mix(bits);
  }
  mixStr(CM.NativeArch);
  FP = h;
  return FP;
}

Error requireCostModel(CostModelState &CM, StringRef targetCpu,
                       StringRef function,
                       function_ref<bool(StringRef, StringRef)> IsRow) {
  if (Error E = resolveForTarget(CM, targetCpu, function))
    return E;
  if (Error E = loadCostModel(CM, IsRow))
    return E;
  if (CM.NativeArch.empty())
    return modelError(CM, "carries no '# native_arch=' header, so it cannot "
                          "be matched against the compile target");
  if (targetCpu != CM.NativeArch)
    return modelError(CM,
                      "was measured on native_arch '" + CM.NativeArch +
                          "' but " + function + " compiles for target-cpu '" +
                          (targetCpu.empty() ? "<unset>" : targetCpu) + "'");
  return Error::success();
}

const std::string &getCostModelNativeArch(CostModelState &CM) {
  if (!costModelPath(CM).empty())
    getCostModel(CM);
  return CM.NativeArch;
}

std::optional<double> lookupCost(CostModelState &CM, StringRef OpcodeName,
                                 StringRef TypeName) {
  const auto &CostModel = getCostModel(CM);
  auto It = CostModel.find({OpcodeName.str(), TypeName.str()});
  if (It == CostModel.end())
    return std::nullopt;
  return It->second;
}

double queryCostModelOr(CostModelState &CM, const std::string &OpcodeName,
                        const std::string &PrecisionName, double fallback) {
  return lookupCost(CM, OpcodeName, PrecisionName).value_or(fallback);
}

} // namespace poseidon

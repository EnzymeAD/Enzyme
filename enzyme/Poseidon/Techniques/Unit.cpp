//=- Unit.cpp - an optimization unit of a site ----------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Unit.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <utility>

using namespace llvm;

namespace poseidon {

static constexpr UnitKindInfo kUnitKinds[] = {
    {UnitKind::R, "CO", "R", "a rewrite", true, "candidate_outputs",
     "Candidate outputs"},
    {UnitKind::PT, "CS", "PT", "a PT", true, "candidate_subgraphs",
     "Candidate subgraphs"},
    {UnitKind::M, "CM", "M", "a rewrite", false, "", ""},
};

ArrayRef<UnitKindInfo> unitKinds() { return kUnitKinds; }

const UnitKindInfo &infoOf(UnitKind kind) {
  const UnitKindInfo &info = kUnitKinds[unsigned(kind)];
  assert(info.kind == kind && "unit kinds listed out of order");
  return info;
}

void SiteUnit::bind(Session &S, FunctionFPState &site, FPGraph &graph,
                    size_t indexInKind) {
  assert(!boundSite && "a unit is bound to its site once");
  boundSession = &S;
  boundSite = &site;
  boundGraph = &graph;
  boundIndex = indexInKind;
}

bool SiteUnit::footprintOverlaps(const OptimizationUnit &other) const {
  const SetVector<Instruction *> *a = footprint();
  const SetVector<Instruction *> *b = cast<SiteUnit>(other).footprint();
  if (!a || !b || a == b)
    return false;
  if (a->size() > b->size())
    std::swap(a, b);
  return any_of(*a, [&](Instruction *I) { return b->contains(I); });
}

bool SiteUnit::rewritesSameOutputAs(const OptimizationUnit &other) const {
  ArrayRef<Value *> theirs = cast<SiteUnit>(other).rewrittenOutputs();
  return any_of(rewrittenOutputs(),
                [&](Value *V) { return is_contained(theirs, V); });
}

SiteUnit &unitOf(const SolutionStep &step) {
  return cast<SiteUnit>(*step.unit);
}

std::string SiteUnit::candidateId(size_t i) const {
  return (info().idPrefix + Twine(indexInKind()) + "_" + Twine(ordinalOf(i)))
      .str();
}

std::optional<size_t> SiteUnit::positionOfOrdinal(size_t ordinal) const {
  if (!info().pricingDrops) {
    if (ordinal < numCandidates())
      return ordinal;
    return std::nullopt;
  }
  for (size_t i = 0; i < numCandidates(); ++i)
    if (ordinalOf(i) == ordinal)
      return i;
  return std::nullopt;
}

json::Value jsonFloat(double v) {
  if (std::isfinite(v))
    return json::Value(v);
  return json::Value(nullptr);
}

std::optional<Tradeoff> classifyTradeoff(double compDelta, double accDelta) {
  if (compDelta <= 0 && accDelta <= 0)
    return Tradeoff{"free_win", std::abs(accDelta) + std::abs(compDelta)};
  if (compDelta <= 0 && accDelta > 0)
    return Tradeoff{"speed_for_accuracy", (accDelta > 1e-30)
                                              ? std::abs(compDelta) / accDelta
                                              : std::abs(compDelta)};
  if (compDelta > 0 && accDelta <= 0)
    return Tradeoff{"accuracy_for_speed", (compDelta > 0)
                                              ? std::abs(accDelta) / compDelta
                                              : std::abs(accDelta)};
  return std::nullopt;
}

json::Object getSourceLocationJSON(Value *V) {
  json::Object loc;
  if (auto *I = dyn_cast<Instruction>(V)) {
    if (const auto &DL = I->getDebugLoc()) {
      loc["file"] = DL->getFilename().str();
      loc["line"] = static_cast<int64_t>(DL.getLine());
      loc["col"] = static_cast<int64_t>(DL.getCol());
    }
  }
  return loc;
}

std::string getSourceLocationStr(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    if (const auto &DL = I->getDebugLoc()) {
      return (DL->getFilename() + ":" + Twine(DL.getLine()) + ":" +
              Twine(DL.getCol()))
          .str();
    }
  }
  return "";
}

std::string getValueStr(Value *V) {
  std::string str;
  raw_string_ostream OS(str);
  V->print(OS);
  return str;
}

json::Array getSourceLocationsJSON(ArrayRef<Instruction *> insts) {
  json::Array locs;
  SmallSet<std::string, 4> seen;
  for (auto *I : insts) {
    auto locStr = getSourceLocationStr(I);
    if (!locStr.empty() && seen.insert(locStr).second) {
      locs.push_back(getSourceLocationJSON(I));
    }
  }
  return locs;
}

} // namespace poseidon

//===- SiteUnit.cpp - an optimization unit of a site ----------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SiteUnit.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <utility>

using namespace llvm;

namespace poseidon {

static constexpr UnitKindInfo kUnitKinds[] = {
    {UnitKind::Algebraic, "rewrite", "R", "a rewrite", true},
    {UnitKind::Precision, "precision change", "P", "a precision change", true},
    {UnitKind::Matmul, "matrix product", "M", "a candidate", false},
};

ArrayRef<UnitKindInfo> unitKinds() { return kUnitKinds; }

Expected<BaselineAccuracy> baselineAccuracy(SubgraphSamples &samples,
                                            ArrayRef<const FPNode *> nodes,
                                            ArrayRef<double> grads,
                                            bool print) {
  BaselineAccuracy B;
  Expected<ArrayRef<SubgraphSamples::Point>> drawn = samples.points();
  if (!drawn)
    return drawn.takeError();
  B.points = *drawn;
  for (auto [node, grad] : zip(nodes, grads)) {
    const SubgraphSamples::Reference &ref = samples.reference(node);
    std::optional<double> error = ref.meanError();
    if (!error)
      return createStringError(
          inconvertibleErrorCode(),
          "every sample of the original expression scored NaN");
    B.outputs.push_back({node, &ref, *error, grad});
    B.cost += *error * std::fabs(grad);
  }
  if (!std::isfinite(B.cost)) {
    // A non-finite baseline makes every candidate's delta NaN, and the DP
    // would quietly keep the subgraph as written.
    if (print)
      for (const BaselineAccuracy::Output &O : B.outputs)
        errs() << "[poseidon] output baseline accuracy cost " << O.error
               << ", gradient " << O.grad << '\n';
    return createStringError(
        inconvertibleErrorCode(),
        "the baseline accuracy cost is not finite; the site's profile is "
        "degenerate (a zero or null-space output shadow gives a gradient "
        "that cannot weight anything)");
  }
  return B;
}

const UnitKindInfo &infoOf(UnitKind kind) {
  const UnitKindInfo &info = kUnitKinds[unsigned(kind)];
  assert(info.kind == kind && "unit kinds listed out of order");
  return info;
}

void SiteUnit::bind(Session &S, SiteState &site, FPGraph &graph,
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
  return (info().letter + Twine(indexInKind()) + "_" + Twine(ordinalOf(i)))
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

SourceLoc locOf(const DebugLoc &DL) {
  if (!DL)
    return {};
  return {DL->getFilename().str(), DL.getLine(), DL.getCol()};
}

SourceLoc locOf(const Value *V) {
  if (auto *I = dyn_cast_or_null<Instruction>(V))
    return locOf(I->getDebugLoc());
  return {};
}

json::Object locJSON(const SourceLoc &L) {
  json::Object o;
  o["file"] = L.file;
  o["line"] = (int64_t)L.line;
  o["col"] = (int64_t)L.col;
  return o;
}

void putLoc(json::Object &o, StringRef key, const SourceLoc &L) {
  if (L.valid())
    o[key] = locJSON(L);
}

std::string getSourceLocationStr(Value *V) {
  if (auto *I = dyn_cast<Instruction>(V)) {
    if (I->getDebugLoc()) {
      SourceLoc L = locOf(I->getDebugLoc());
      return (L.file + ":" + Twine(L.line) + ":" + Twine(L.col)).str();
    }
  }
  return "";
}

std::string getValueStr(Value *V) { return formatv("{0}", *V).str(); }

} // namespace poseidon

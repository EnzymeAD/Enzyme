//===- Unit.cpp - optimization units; anchors of the Core interfaces ------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Unit.h"
#include "Core/Session.h"
#include "Core/Technique.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"

#include <utility>

using namespace llvm;

namespace poseidon {

AttachedStates::Holder::~Holder() = default;

OptimizationUnit::~OptimizationUnit() = default;

Technique::~Technique() = default;

void SiteUnit::bind(Session &S, SiteState &site, size_t indexInKind) {
  assert(!boundSite && "a unit is bound to its site once");
  boundSession = &S;
  boundSite = &site;
  boundIndex = indexInKind;
}

FPGraph &SiteUnit::graph() const { return site(); }

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

} // namespace poseidon

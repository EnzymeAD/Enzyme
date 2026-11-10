//=- Unit.h - an optimization unit of a site ------------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the pass, beyond the solver, needs of an optimization unit: the site it
// belongs to, its footprint as instructions, its candidates by name, the fixed
// formats it appears in, and its materialization. Every kind of unit derives
// from SiteUnit; the solver sees the OptimizationUnit base.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_UNIT_H
#define POSEIDON_TECHNIQUES_UNIT_H

#include "Core/Solver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/JSON.h"

#include <cassert>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace llvm {
class Instruction;
class Value;
class raw_ostream;
} // namespace llvm

#if LLVM_VERSION_MAJOR >= 21
#define GET_INSTRUCTION_COST(cost) (cost.getValue())
#else
#define GET_INSTRUCTION_COST(cost) (cost.getValue().value())
#endif

namespace poseidon {

class Session;
struct FunctionFPState;

// The kinds of unit, by the letter -poseidon-apply-rewrites names them with.
enum class UnitKind { R, PT, M };

// What the pass's fixed formats call one kind of unit.
struct UnitKindInfo {
  UnitKind kind;
  // The DP table's itemType; the noun of the solver's progress lines and of
  // the pin errors.
  llvm::StringLiteral tag;
  // The <unit> of the ids <unit><i>_<j> and of the pins <unit><i>=<name>.
  llvm::StringLiteral idPrefix;
  // What the duplicate-pin error says the unit already has.
  llvm::StringLiteral appliedNoun;
  // Pricing may drop candidates: a candidate's ordinal is not its position,
  // and a DP table step names its candidate by key as well as by position.
  bool pricingDrops;
  // The summary of -poseidon-report-path this kind fills, and its title in
  // the text report; empty for none.
  llvm::StringLiteral reportSection;
  llvm::StringLiteral reportTitle;
};

// The kinds, in the order the fixed formats list them.
llvm::ArrayRef<UnitKindInfo> unitKinds();
const UnitKindInfo &infoOf(UnitKind kind);

class SiteUnit;
// The unit a step takes a candidate of.
SiteUnit &unitOf(const SolutionStep &step);

class SiteUnit : public OptimizationUnit {
public:
  // Every unit of the pass is a SiteUnit.
  static bool classof(const OptimizationUnit *) { return true; }

  // Binds the unit to the session and the site it was proposed for, as the
  // site's `indexInKind`-th unit of its kind. Once.
  void bind(Session &S, FunctionFPState &site, size_t indexInKind);
  Session &session() const {
    assert(boundSession && "unit read before its site was sealed");
    return *boundSession;
  }
  FunctionFPState &site() const {
    assert(boundSite && "unit read before its site was sealed");
    return *boundSite;
  }
  // The i of R<i>, PT<i>, M<i>, and the DP table's itemIndex.
  size_t indexInKind() const {
    assert(boundSite && "unit read before its site was sealed");
    return boundIndex;
  }

  UnitKind unitKind() const { return UnitKind(kind()); }
  const UnitKindInfo &info() const { return infoOf(unitKind()); }
  llvm::StringRef kindName() const final { return info().tag; }

  // The instructions the unit's candidates edit or consume; null for none.
  virtual const llvm::SetVector<llvm::Instruction *> *footprint() const = 0;
  // The outputs each candidate replaces.
  virtual llvm::ArrayRef<llvm::Value *> rewrittenOutputs() const { return {}; }

  bool footprintOverlaps(const OptimizationUnit &other) const final;
  bool rewritesSameOutputAs(const OptimizationUnit &other) const final;

  // Candidate i's name: what a pin names it by and the DP table keys it by.
  virtual std::string candidateName(size_t i) const = 0;
  // The j of candidate i's id <unit><i>_<j>.
  virtual size_t ordinalOf(size_t i) const { return i; }
  // Candidate i's id, <unit><i>_<j>.
  std::string candidateId(size_t i) const;
  // The position of the candidate whose ordinal is `ordinal`, unless pricing
  // dropped it.
  std::optional<size_t> positionOfOrdinal(size_t ordinal) const;
  // The candidates pricing dropped, by name, each with why.
  virtual llvm::ArrayRef<std::pair<std::string, std::string>>
  droppedCandidates() const {
    return {};
  }
  // Candidate i as the line selecting it by -poseidon-apply-rewrites names it.
  virtual std::string selectionText(size_t i) const = 0;
  // Candidate i as the DP table print and the materialization log name it.
  virtual std::string choiceText(size_t i) const = 0;
  // Candidate i as a step of a -poseidon-report-path report, in its JSON and
  // in its text, each line led by `pad`.
  virtual llvm::json::Object stepJSON(size_t i) = 0;
  virtual void writeReportStep(llvm::raw_ostream &OS, size_t i,
                               llvm::StringRef pad) = 0;
  // Rewrites the site's IR with candidate i.
  virtual void materialize(size_t i) = 0;

  // Prices the candidates and drops those the accuracy model refuses; a unit
  // priced when proposed does nothing.
  virtual void price() {}
  // Multiplies every error the unit carries by `factor`.
  virtual void scaleErrors(double factor) = 0;
  // The -poseidon-print block of the priced candidates.
  virtual void printPriced(llvm::raw_ostream &OS) = 0;
  // The unit's line of the DP table key's candidate set; each candidate
  // follows on its own line as two spaces and its name.
  virtual std::string keyHeader() const = 0;

  // For a kind with a report section: the unit's entry of the section, its
  // block of the text report, and its entries of the curated rewrites.
  virtual void addReportSummary(llvm::json::Array &summary) {}
  virtual void writeReportSummary(llvm::raw_ostream &OS) {}
  virtual void addCuratedRewrites(llvm::json::Array &rewrites) {}

protected:
  explicit SiteUnit(UnitKind kind) : OptimizationUnit(unsigned(kind)) {}

private:
  Session *boundSession = nullptr;
  FunctionFPState *boundSite = nullptr;
  size_t boundIndex = 0;
};

// For the reports: a number, or null when it is not finite.
llvm::json::Value jsonFloat(double v);
// An instruction's source location: file, line and column; empty without one.
llvm::json::Object getSourceLocationJSON(llvm::Value *V);
// An instruction's source location as file:line:col; empty without one.
std::string getSourceLocationStr(llvm::Value *V);
// A value as the IR prints it.
std::string getValueStr(llvm::Value *V);
// The distinct source locations of `insts`, in their order.
llvm::json::Array
getSourceLocationsJSON(llvm::ArrayRef<llvm::Instruction *> insts);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_UNIT_H

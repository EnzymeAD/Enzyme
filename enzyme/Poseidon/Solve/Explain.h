//===- Explain.h - what a solve changed, considered and rejected ----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimization remarks (pass name "poseidon") and <func>.explain.json, built
// from the solver's own data after it has chosen and before anything is
// materialized. Output only: nothing here feeds back into a decision.
//
// <func>.explain.json, schema "poseidon-explain/1", keys in snake_case; a key
// whose value is unknown or not finite is left out:
//   schema     "poseidon-explain/1"
//   function   name (demangled), symbol, ir (the IR function solved), site (the
//              call: file, line, col), loc (the definition); for a solved site
//              also canonical_hash (of the form the profile was recorded
//              against) and profile (the profile file read)
//   target     tau and tau_from ("site" or "flag") under an accuracy target,
//              else budget (the -poseidon-cost-budget share); for a solved site
//              also cost_model, selector ("dp", "joint", "pinned"),
//              from_cache (the DP table was read from the cache),
//              sites (the IR functions of a joint solve)
//   scale      (solved sites) accuracy_scale S and baseline_accuracy_cost A0:
//              an accuracy cost a is the modelled relative error (A0 + a) / S;
//              profiled_cost, the cost-model units the profiled run spent, a
//              cost share is a fraction of; kappa, kappa_mark and kappa_factor
//              when the profiling run measured a condition number
//   result     status: "rewritten", "as_written" (solved, nothing applied),
//              "no_solution" (no frontier point meets the target or the
//              budget), "nothing_to_optimize", "no_fp_operation", "unprofiled"
//              (no profile of the site), "never_executed" (the profiling run
//              never ran it), "below_cost_share" (under
//              -poseidon-min-cost-share); reason: why, for every status but
//              "rewritten"; profile: the file looked at, for "unprofiled" and
//              "never_executed"; for a solved site also: applied (the ids
//              applied), steps (each step the solve chose, in order: id,
//              choice, the text of its "Applying solution for" line, and
//              applied, false for a step an earlier one made moot), pin (the
//              -poseidon-pin value that applies them again),
//              selected (the frontier point selected: cost, the share it adds,
//              accuracy_cost, modelled_rel_error), best_achievable_rel_error
//              (for "no_solution" under a target), cost_before, cost_after,
//              modelled_rel_error_as_written, modelled_rel_error
//   subgraphs  (solved sites) each optimization unit: id (S<i> an FP
//              subgraph, M<i> a matrix product), kind ("subgraph", "matmul"),
//              loc, shape, cost, executions, ops (each with opcode, type, loc,
//              executions, sensitivity, gradient, min, max, and for a call of
//              a math function float_call, the C function of its float form),
//              inputs, outputs, original, herbie, left_as_written, and
//              alternatives: each candidate with id, kind ("algebraic",
//              "precision", "expansion", "matmul"), summary (for a product,
//              the realization the solve names it by), inkernel_class (for an
//              in-kernel raise of a product, the class of the inkernel_rel
//              cost-model row that prices it), precision, outputs,
//              fpcore, c, approx, approx_note, changes (each with from, to
//              and the indices of its ops), cost_delta, modelled_error,
//              modelled_domain_error (of a product: its relative output
//              error, its term of modelled_error under a target),
//              herbie_error_bits, status ("applied", "pareto",
//              "rejected") and reason
//   pareto     (solved sites) the frontier in cost order: budget (the
//              -poseidon-cost-budget that selects the point; none for a point
//              no budget selects), cost, modelled_rel_error, applied, selected
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_EXPLAIN_H
#define POSEIDON_SOLVE_EXPLAIN_H

#include "Core/FPCore.h"
#include "Support/Utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace poseidon {

struct SiteState;
struct Options;
class Session;
class SiteUnit;
struct SolutionStep;
struct SolveRecord;

/// How the solve treated a candidate, in the order the remarks list rejected
/// ones: closest to being taken first.
enum class Standing {
  Applied,
  /// Above the target alone, and cheaper than as written.
  CheaperAboveTarget,
  /// On a frontier point the selection passed over.
  Pareto,
  Near,
  Far,
  /// Never priced.
  Dropped,
};

/// A candidate among the alternatives of an entry of the explanation; what is
/// empty or none is not written.
struct Alternative {
  /// Its id, kind, summary, and why the solve treated it as it did.
  std::string id, kind, summary, reason;
  /// The keys of its kind, written into its entry as they are.
  llvm::json::Object keys;
  /// What the remarks call a candidate of its kind ("the float expansion").
  std::string kindWords;
  /// Its rewritten expression, which the remarks show when it has no single C
  /// form; none for a kind whose remarks show its summary instead.
  std::optional<std::string> expr;
  /// Its C form, one expression per output.
  llvm::SmallVector<std::string, 1> c;
  /// The positions of the outputs it rewrites among its subgraph's.
  llvm::SmallVector<unsigned, 1> outputs;
  /// Its cost change, and the cost as written of what it rewrites, on the
  /// solve's axis.
  std::optional<int64_t> costDelta, baseCost;
  /// Its accuracy cost change and its modelled error.
  std::optional<double> accDelta, err;
  /// The accuracy cost as written of what it rewrites.
  double baseAcc = std::numeric_limits<double>::quiet_NaN();
  /// Where its remarks point.
  const llvm::Instruction *at = nullptr;
  /// Its unit and position there; none for a candidate never offered.
  const SiteUnit *item = nullptr;
  size_t index = 0;
  Standing standing = Standing::Far;
};

/// An entry of the explanation: an FP subgraph with the units over it, or a
/// unit over something else.
struct UnitExplanation {
  std::string id, kind;
  /// The subgraph; null for a unit over something else.
  const Subgraph *sg = nullptr;
  /// The unit over something else; null for a subgraph.
  const SiteUnit *owner = nullptr;
  /// How the remarks name an entry of a unit over something else.
  std::string name;
  /// The keys of its kind, written into the entry as they are.
  llvm::json::Object keys;
  /// Where it is, and the instruction its remarks point at.
  SourceLoc loc;
  const llvm::Instruction *root = nullptr;
  /// Its ops, inputs and outputs.
  llvm::json::Array ops, inputs, outputs;
  /// Each output as C, and the profiled ranges of the inputs as remarks name
  /// them.
  llvm::SmallVector<std::string, 2> outC, inputRanges;
  std::vector<Alternative> alts;
  /// Its cost as written on the solve's axis.
  std::optional<int64_t> baseCost;
  uint64_t executions = 0;
  double sensitivity = std::numeric_limits<double>::quiet_NaN();
  /// Why it is left as written.
  std::string left;
  /// Why its units offer no candidate, when they offer none.
  llvm::SmallVector<std::string, 2> whyNone;
  /// Whether a candidate of it is applied.
  bool applied = false;

  /// Notes that the candidates of one of its units could not be priced.
  void unpriced(llvm::StringRef reason) {
    whyNone.push_back(("its candidates could not be priced: " + reason).str());
  }
};

/// The explanation of a site's solve, as a unit adds its entry or its part of
/// its subgraph's (SiteUnit::explain).
class Explanation {
public:
  /// `cost`, in cost-model units over the profiled run, on the solve's axis;
  /// an error at the site when the axis cannot hold it.
  virtual llvm::Expected<int64_t> onAxis(double cost) const = 0;
  /// Sets A's accuracy cost change to `acc`, and its modelled error, when
  /// `acc` is finite.
  virtual void setAccuracy(Alternative &A, double acc) const = 0;
  /// How C writes the values of `sg`: its inputs as leaves, by source name.
  virtual CRendering rendering(const Subgraph &sg) = 0;
  /// A graph to parse expressions into, apart from the solve's own.
  virtual FPGraph &scratch() = 0;
  /// The steps the solve applies, in order.
  virtual llvm::ArrayRef<SolutionStep> appliedSteps() const = 0;

protected:
  ~Explanation() = default;
};

/// `items` as a list in words: "a", "a and b", "a, b and c".
std::string joinWords(llvm::ArrayRef<std::string> items);

/// `s`, cut after `n` characters and marked as cut; 160 is what a remark shows
/// of an expression.
std::string clip(const std::string &s, size_t n = 160);

/// `v` as the remarks write a number, to `digits` significant digits; "n/a"
/// when it is not finite.
std::string remarkNumber(double v, int digits = 2);

/// Whether the solve of `F` is explained: -poseidon-explain names a directory,
/// or a remark of the pass is enabled for `F`.
bool explainWanted(const Options &O, const llvm::Function &F);

/// The explanation of a site the pass leaves as written without solving it:
/// `status` is what kept it from the solve and `reason` says why. F is the
/// site's clone, or with `body` the site's body when no clone was made. `tau`
/// > 0 is the site's accuracy target, from the site when `siteTarget`.
void explainLeftAlone(Session &S, llvm::Function &F, double tau,
                      bool siteTarget, llvm::StringRef status,
                      llvm::StringRef reason, llvm::StringRef profile = "",
                      bool body = false);

/// A remark and the explanation of a site with nothing to optimize.
void explainNothingToOptimize(Session &S, llvm::Function &F, double tau,
                              bool siteTarget);

/// A remark and the explanation of a site no profile has records of: `status`
/// is "unprofiled" or "never_executed", `profile` the file looked at.
void explainUnprofiled(Session &S, llvm::Function &F, double tau,
                       bool siteTarget, llvm::StringRef status,
                       llvm::StringRef cause, llvm::StringRef profile);

/// Where a remark about the site F points: its call, else F.
llvm::DiagnosticLocation siteLocation(const Session &S,
                                      const llvm::Function &F);

/// Remarks and the explanation of the solve of the site of `st`; an operation
/// of a subgraph the cost model has no row for is an error.
llvm::Error explainSolve(Session &S, llvm::Function &F, SiteState &st,
                         llvm::ArrayRef<SolutionStep> steps,
                         const SolveRecord &rec);

} // namespace poseidon
#endif // POSEIDON_SOLVE_EXPLAIN_H

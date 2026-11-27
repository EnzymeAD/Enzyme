//===- Solve.h - a site's candidates, solve and materialization -----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The header of Optimize.cpp, SolveSites.cpp and Subgraphs.cpp. A site's FP
// subgraphs (Subgraphs.cpp), candidates, solve and materialization
// (Optimize.cpp); the DP over the units of one site or of several sites, the
// selection of a frontier point by budget or accuracy target, and the cache of
// a site's solved table (SolveSites.cpp): a later compile with the same key
// binds the table to its own candidates instead of pricing and solving again.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_SOLVE_H
#define POSEIDON_SOLVE_SOLVE_H

#include "Core/ErrorMoment.h"
#include "Core/FPNode.h"
#include "Core/Session.h"
#include "Core/Technique.h"
#include "Core/Unit.h"
#include "Model/CostModel.h"
#include "Profile/ProfileReader.h"
#include "Solve/DP.h"
#include "Support/CacheKey.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace poseidon {

/// What chose a solve's applied steps.
enum class SolveSelector {
  /// The DP over one site's units.
  DP,
  /// The DP over the units of the sites of -poseidon-solve-jointly.
  Joint,
  /// -poseidon-pin, without a DP.
  Pinned,
};

/// The selector as the explanation spells it: "dp", "joint" or "pinned".
llvm::StringRef selectorName(SolveSelector selector);

/// Why a solve's selection took or passed over a frontier point, best first.
enum class PointVerdict {
  /// The point the target or the budget selected.
  Selected,
  /// It meets the target or the budget; the point selected is preferred.
  Outranked,
  /// It is above the accuracy target or the cost budget.
  Excluded,
  /// Its accuracy cost is not finite, so no target or budget selects it.
  Unselectable,
};

/// How a site's solve went, for its explanation: the frontier it searched and
/// the point it chose.
struct SolveRecord {
  /// What chose the applied point.
  SolveSelector selector = SolveSelector::DP;
  /// Whether the frontier came from the DP table cache.
  bool fromCache = false;
  /// The frontier: per cost, the least accuracy cost of a strategy of that
  /// cost.
  CostMap table;
  /// Per cost on the frontier, its strategy.
  SolutionMap solutions;
  /// Per cost on the frontier, why the selection took or passed over it.
  std::map<DPCost, PointVerdict> verdicts;
  /// The accuracy cost of the site as written; NaN when it was not computed.
  double baseline = std::numeric_limits<double>::quiet_NaN();
  /// The sites' summed accuracy scale (SiteState::accScale).
  double accScale = 0.0;
  /// The moment `table`, `baseline` and `accScale` are in, the sites'.
  ErrorMoment moment = ErrorMoment::First;
  /// What the profiled run spent in the solved sites: the DP's cost axis is in
  /// shares of it.
  double profiledCost = 0.0;
  /// The relative error target the solve was held to; 0 when it solved for a
  /// cost budget.
  double tau = 0.0;
  /// Whether `tau` was written at the site rather than given by -poseidon-tau.
  bool siteTarget = false;
  /// The sites of a joint solve.
  std::vector<std::string> sites;
  /// The cost of the point the target or the budget selected; none when it
  /// selected none.
  std::optional<DPCost> selected;
  /// The site's time once its memory traffic is priced, in cost-model units
  /// over the profiled run: its arithmetic at the selected point, its
  /// arithmetic and memory combined for the code as written and at the
  /// selected point, and whether that point is taken. Absent when the traffic
  /// is not priced.
  struct Time {
    double arithSelected;
    double combinedAsWritten, combinedSelected;
    bool taken;
  };
  std::optional<Time> time;
};

/// A site's DP table in the cache: the key of what it is solved from, the file
/// the entry under that key is kept in and, when the cache holds it, the table
/// bound to this compile's candidates.
struct DPTable {
  /// The cache file the entry is kept in.
  std::string path;
  /// What the table is solved from.
  CacheKey key;
  /// Whether the cache held an entry under `key`.
  bool hit = false;
  /// The cached frontier, when `hit`.
  CostMap costToAccuracyMap;
  /// The strategies of the cached frontier, bound to this compile's candidates,
  /// when `hit`.
  SolutionMap costToSolutionMap;
  /// The accuracy cost of the site as written, as cached; none when the entry
  /// carries none.
  std::optional<double> baseline;
};

/// The cache entry of the DP table of the site `st` over `units`, read from
/// the cache when it holds one; an entry that is unreadable or was solved by
/// another compiler is reported and missed.
llvm::Expected<std::unique_ptr<DPTable>>
lookupDPTable(Session &S, const SiteState &st, llvm::ArrayRef<SiteUnit *> units,
              llvm::StringRef profilePath);

/// The order the DP takes `units[i]`, units of the i-th site, in, which is the
/// order of the steps of every solution: exactOrder by default, or
/// prunedOrder under -poseidon-early-prune.
llvm::SmallVector<OptimizationUnit *>
solveOrder(const Options &O, llvm::ArrayRef<llvm::ArrayRef<SiteUnit *>> units);

/// Solves `units[i]`, units of the site `sites[i]`, in one DP on the axis
/// `scale` of the sites' summed profiled cost, and selects a frontier point:
/// under `tau` > 0 the one the target picks, else the most accurate one within
/// -poseidon-cost-budget. Under `tau` a point's modelled relative error is
/// relativeError(A0 + its accuracy cost, `accScale`) in the sites' moment,
/// which they share, `accScale` the sites' summed accScale and A0 the accuracy
/// cost of the sites as written. For one site in the first moment that is the
/// sum over its units of an FP subgraph's accuracy cost over accScale (a
/// per-operation relative rounding level) and a matrix product's relative
/// domain error; in the second moment, the root of the sum of their squares. A
/// single site's table is read from or written to the cache when the site has
/// one. `joint` marks a solve of the sites of -poseidon-solve-jointly, however
/// many. Once the memory traffic of a single site is priced, its time is its
/// arithmetic and its traffic combined, and a point selected for its cost
/// replaces the code as written only when its combined time is lower; else
/// nothing is applied.
llvm::Expected<llvm::SmallVector<SolutionStep>>
solveSites(Session &S, llvm::ArrayRef<SiteState *> sites,
           llvm::ArrayRef<llvm::ArrayRef<SiteUnit *>> units,
           const CostScale &scale, double accScale, double tau, bool joint,
           SolveRecord *record);

/// What a site is solved from, for its solve and the explanation of it: the
/// profile, its condition number and its memory traffic, and the site's DP
/// table in the cache.
struct SolvedFrom {
  static inline char ID;
  /// The profile the site is solved from.
  std::string profilePath;
  /// The canonical form hash the profile was checked against.
  std::string canonicalHash;
  /// The condition number the profiling run measured; none when it measured
  /// none.
  std::optional<double> kappa;
  /// How the profiling run measured `kappa`.
  KappaMark kappaMark = KappaMark::None;
  /// The cost model's stream rows, when it has them.
  std::optional<StreamPrices> streamPrices;
  /// What the profiled run's memory traffic cost, in cost-model units, when
  /// the stream rows price it and the profile records it; otherwise why not.
  std::optional<double> trafficCost;
  std::string noTrafficReason;
  /// Set when the site is solved against the DP table cache.
  std::unique_ptr<DPTable> table;
};

/// Binds every unit of the site `st` to it and lists it in its `units`,
/// technique by technique. Once per site, after its last proposal: no
/// technique proposes a unit for the site after it.
void sealUnits(Session &S, SiteState &st);

/// The file in the profile directory `dir` that holds the profile of the site
/// clone `clone`.
std::string siteProfilePath(const Session &S, const llvm::Function &clone,
                            llvm::StringRef dir);

/// `tau` > 0 is the accuracy target written at the site; 0 means the site
/// wrote none and -poseidon-tau decides.
llvm::Expected<bool> optimizeSite(Session &S, llvm::Function &F, double tau);

/// The attribute that marks a site of the joint solve, holding the site's
/// accuracy target.
constexpr llvm::StringLiteral kJointTau = "poseidon-joint-tau";

/// Solves one DP over all sites of `M` marked with kJointTau.
llvm::Expected<bool> solveJointly(Session &S, llvm::Module &M);

/// The FP subgraphs of the site `st`, flood-filled over the optimizable
/// instructions of its function, and a node of its graph for every value they
/// read or compute. The instructions `profileMap` records no execution of are
/// in no subgraph.
void buildSubgraphs(const Options &O, const ProfileMap &profileMap,
                    SiteState &st);

/// Sets the profiled sensitivity, gradient (the accuracy weight under
/// `errorSum`), squared gradient, execution count and value ranges of the
/// subgraphs' nodes. A profile that gives an operand no range was not
/// generated from this program, an error at the operation.
llvm::Error attachProfile(const Options &O, const ProfileMap &profileMap,
                          ErrorSum errorSum, FPGraph &G,
                          llvm::ArrayRef<Subgraph> subgraphs);

/// The subgraphs the site's units are proposed for: split, then reshaped by
/// each technique.
void shapeSubgraphs(const Options &O, llvm::Function &F,
                    llvm::SmallVectorImpl<Subgraph> &subgraphs);

/// The steps of `steps` that are applied, in order: a step whose outputs an
/// earlier step already rewrote is skipped. The binary64 and binary32 searches
/// of one output are separate units, and the first rewrite erases the output
/// the second would replace.
llvm::SmallVector<SolutionStep>
stepsToApply(llvm::ArrayRef<SolutionStep> steps);

} // namespace poseidon
#endif // POSEIDON_SOLVE_SOLVE_H

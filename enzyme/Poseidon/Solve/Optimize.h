//===- Optimize.h - a site's candidates, solve and materialization --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A site's candidates, solve and materialization; the DP over the units of one
// site or of several sites, the selection of a frontier point by budget or
// accuracy target, and the cache of a site's solved table (SolveSites.cpp): a
// later compile with the same key binds the table to its own candidates instead
// of pricing and solving again.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_OPTIMIZE_H
#define POSEIDON_SOLVE_OPTIMIZE_H

#include "Model/FPNode.h"
#include "Profile/ProfileReader.h"
#include "Solve/DP.h"
#include "Support/CacheKey.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision/Precision.h"
#include "Techniques/Rewrite/Rewrite.h"
#include "Techniques/SiteUnit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace poseidon {

struct SiteState;
struct Options;
class Session;

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
  /// The point applied.
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
  /// The cost of the point applied; none when nothing was applied.
  std::optional<DPCost> selected;
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

/// Solves `units[i]`, units of the site `sites[i]`, in one DP and selects a
/// frontier point: under `tau` > 0 the one the target picks, else the most
/// accurate one within -poseidon-cost-budget. Under `tau` a point's modelled
/// relative error is (A0 + its accuracy cost) / the sites' summed accScale,
/// A0 the accuracy cost of the sites as written: for one site, the sum over
/// its units of an FP subgraph's accuracy cost over accScale (a per-operation
/// relative rounding level) and a matrix product's relative domain error. A
/// single site's table is read from or written to the cache when the site has
/// one. `joint` marks a solve of the sites of -poseidon-solve-jointly, however
/// many.
llvm::Expected<llvm::SmallVector<SolutionStep>>
solveSites(Session &S, llvm::ArrayRef<SiteState *> sites,
           llvm::ArrayRef<llvm::ArrayRef<SiteUnit *>> units, double tau,
           bool joint, SolveRecord *record);

/// One site's solve: its FP graph, its profile's scale, the units proposed for
/// it and the analyses its matrix products hold.
struct SiteState : FPGraph {
  /// The accuracy target written at the site; 0 when the site has none and
  /// -poseidon-tau applies.
  double tau = 0.0;
  /// Sum of the profile's per-slot sumSens, times the kappa factor the accuracy
  /// costs carry; the denominator that turns an accuracy cost into a
  /// per-operation relative rounding level.
  double accScale = 0.0;
  /// What the profiled run spent in the site, in cost-model units: the
  /// execution count times the price of every operation the profile records.
  double profiledCost = 0.0;
  /// The profile the site is solved from.
  std::string profilePath;
  /// The canonical form hash the profile was checked against.
  std::string canonicalHash;
  /// The condition number the profiling run measured; none when it measured
  /// none.
  std::optional<double> kappa;
  /// How the profiling run measured `kappa`.
  KappaMark kappaMark = KappaMark::None;
  /// The factor the accuracy costs were multiplied by to put them on the
  /// application's scale: kappa over the summed sensitivity, 1 without a
  /// measured kappa.
  double scaleFactor = 1.0;
  /// The site's FP subgraphs, the connected components of its operations.
  llvm::SmallVector<Subgraph, 1> subgraphs;
  /// Backing storage for the matmuls; MatmulUnit::matmul points into this,
  /// so it must outlive materialization.
  llvm::SmallVector<RecognizedMatmul, 4> abstractMatmuls;
  /// The rewrite units proposed for the site.
  llvm::SmallVector<RewriteUnit, 4> rewriteUnits;
  /// The precision units proposed for the site.
  llvm::SmallVector<PrecisionUnit, 4> precisionUnits;
  /// The matrix-product units proposed for the site.
  llvm::SmallVector<MatmulUnit, 4> matmulUnits;
  /// The site's units in site order (rewriteUnits, precisionUnits,
  /// matmulUnits), each bound to this site.
  llvm::SmallVector<SiteUnit *, 8> units;
  /// Whether sealUnits has run.
  bool unitsSealed = false;
  /// Set when the site is solved against the DP table cache.
  std::unique_ptr<DPTable> table;

  /// Analyses for the loop raises. The matmuls found hold raiseSE and SCEVs it
  /// owns, so these outlive materialization. raiseSE is declared last so that
  /// it is destroyed first: it references the others.
  std::optional<llvm::DominatorTree> raiseDT;
  /// The loop, assumption, library and scalar-evolution analyses of the site
  /// clone, for the loop raises; built together with raiseDT.
  std::optional<llvm::LoopInfo> raiseLI;
  std::optional<llvm::AssumptionCache> raiseAC;
  std::optional<llvm::TargetLibraryInfoImpl> raiseTLII;
  std::optional<llvm::TargetLibraryInfo> raiseTLI;
  std::optional<llvm::ScalarEvolution> raiseSE;

  /// Binds every unit to this site and lists it in `units`. Once per site,
  /// after its last proposal: no unit is added to rewriteUnits, precisionUnits
  /// or matmulUnits after it.
  void sealUnits(Session &S);
};

/// A site's profile: the file it was read from, its records and the canonical
/// form hash of the site they were checked against.
struct SiteProfileData {
  /// The profile file.
  std::string path;
  /// The profile's records by slot.
  ProfileMap map;
  /// The profile's header.
  SiteProfileHeader header;
  /// The canonical form hash of the site, as computed for this compile.
  std::string canonicalHash;
};

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

/// The FP subgraphs of F, flood-filled over its optimizable instructions, and
/// a node for every value they read or compute. The instructions `profileMap`
/// records no execution of are in no subgraph.
void buildSubgraphs(
    const Options &O, llvm::Function &F, const ProfileMap &profileMap,
    const llvm::SmallVectorImpl<RecognizedMatmul> &abstractMatmuls, FPGraph &G,
    llvm::SmallVectorImpl<Subgraph> &subgraphs);

/// Sets the profiled sensitivity, gradient, execution count and value ranges
/// of the subgraphs' nodes. A profile that gives an operand no range was not
/// generated from this program, an error at the operation.
llvm::Error attachProfile(const Options &O, const ProfileMap &profileMap,
                          FPGraph &G, llvm::ArrayRef<Subgraph> subgraphs);

/// The site's optimization units: the subgraphs after splitting and the
/// shared-staging merge.
void shapeSubgraphs(const Options &O, llvm::Function &F,
                    llvm::SmallVectorImpl<Subgraph> &subgraphs);

/// The steps of `steps` that are applied, in order: a step whose outputs an
/// earlier step already rewrote is skipped. The binary64 and binary32 searches
/// of one output are separate units, and the first rewrite erases the output
/// the second would replace.
llvm::SmallVector<SolutionStep>
stepsToApply(llvm::ArrayRef<SolutionStep> steps);

} // namespace poseidon
#endif // POSEIDON_SOLVE_OPTIMIZE_H

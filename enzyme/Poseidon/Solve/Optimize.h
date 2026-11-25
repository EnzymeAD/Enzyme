//===- Optimize.h - a site's candidates, solve and materialization --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_OPTIMIZE_H
#define POSEIDON_SOLVE_OPTIMIZE_H

#include "Model/FPNode.h"
#include "Profile/ProfileReader.h"
#include "SolveSites.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision/Precision.h"
#include "Techniques/Rewrite/Rewrite.h"

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

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace poseidon {

struct Options;
class Session;

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

/// Names, once for the site clone F, the shared buffers that more than one of
/// its optimization units reads or writes: a precision change narrows such a
/// buffer only when all of them pick one tier, which
/// -poseidon-subgraph-merge-staging makes them do.
void hintSharedStaging(const Session &S, llvm::Function &F,
                       llvm::ArrayRef<Subgraph> subgraphs);

/// The steps of `steps` that are applied, in order: a step whose outputs an
/// earlier step already rewrote is skipped. The binary64 and binary32 searches
/// of one output are separate units, and the first rewrite erases the output
/// the second would replace.
llvm::SmallVector<SolutionStep>
stepsToApply(llvm::ArrayRef<SolutionStep> steps);

} // namespace poseidon
#endif // POSEIDON_SOLVE_OPTIMIZE_H

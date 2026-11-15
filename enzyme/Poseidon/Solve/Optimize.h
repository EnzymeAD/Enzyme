//=- Optimize.h - a site's candidates, solve and materialization ------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_OPTIMIZE_H
#define POSEIDON_OPTIMIZE_H

#include "Model/FPNode.h"
#include "SiteTable.h"
#include "Techniques/Herbie.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"

#include <memory>
#include <optional>

namespace poseidon {

class Session;

struct FunctionFPState : FPGraph {
  double errTol = 0.0;
  unsigned sampleLogBits = 0;
  // Sum of the profile's per-slot sumSens, times the kappa factor the accuracy
  // costs carry; the denominator that turns an accuracy cost into a
  // per-operation relative rounding level. See accuracyDPSolver.
  double accScale = 0.0;
  // What the profiled run spent in the site, in cost-model units: the
  // execution count times the price of every operation the profile records.
  double profiledCost = 0.0;
  llvm::SmallVector<Subgraph, 1> subgraphs;
  // Backing storage for the matmuls; MatmulUnit::matmul points into this,
  // so it must outlive materialization.
  llvm::SmallVector<AbstractMatmul, 4> abstractMatmuls;
  llvm::SmallVector<RewriteUnit, 4> COs;
  llvm::SmallVector<PrecisionUnit, 4> CSs;
  llvm::SmallVector<MatmulUnit, 4> CMs;
  // The site's units in site order (COs, CSs, CMs), each bound to this site.
  llvm::SmallVector<SiteUnit *, 8> units;
  bool unitsSealed = false;
  // Set when the site is solved against the DP table cache.
  std::unique_ptr<SiteTable> table;

  // Analyses for the loop raises. The matmuls found hold raiseSE and SCEVs it
  // owns, so these outlive materialization. raiseSE is declared last so that
  // it is destroyed first: it references the others.
  std::optional<llvm::DominatorTree> raiseDT;
  std::optional<llvm::LoopInfo> raiseLI;
  std::optional<llvm::AssumptionCache> raiseAC;
  std::optional<llvm::TargetLibraryInfoImpl> raiseTLII;
  std::optional<llvm::TargetLibraryInfo> raiseTLI;
  std::optional<llvm::ScalarEvolution> raiseSE;

  // Binds every unit to this site and lists it in `units`. Once per site,
  // after its last proposal: no unit is added to COs, CSs or CMs after it.
  void sealUnits(Session &S);
};

// `errorTol` > 0 is the accuracy target written at the site; 0 means the site
// wrote none and -poseidon-tau decides.
bool fpOptimize(Session &S, llvm::Function &F, double errorTol = 0.0);

// The attribute that marks a site of the joint solve, holding the site's
// accuracy target.
constexpr llvm::StringLiteral kJointErrTol = "poseidon-joint-errtol";

// Solves one DP over all sites of `M` marked with kJointErrTol.
bool solveJointly(Session &S, llvm::Module &M,
                  llvm::FunctionAnalysisManager &FAM);

} // namespace poseidon
#endif // POSEIDON_OPTIMIZE_H

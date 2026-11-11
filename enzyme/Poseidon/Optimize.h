#ifndef POSEIDON_OPTIMIZE_H
#define POSEIDON_OPTIMIZE_H

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <memory>
#include <optional>

#include "Model/FPNode.h"
#include "SiteTable.h"
#include "Techniques/Herbie.h"
#include "Techniques/Matmul/Matmul.h"
#include "Techniques/Precision.h"

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
  // so it must outlive materialization (it does NOT live in a collect-local).
  llvm::SmallVector<AbstractMatmul, 4> abstractMatmuls;
  llvm::SmallVector<RewriteUnit, 4> COs;
  llvm::SmallVector<PrecisionUnit, 4> CSs;
  llvm::SmallVector<MatmulUnit, 4> CMs;
  // The site's units in site order (COs, CSs, CMs), each bound to this site.
  llvm::SmallVector<SiteUnit *, 8> units;
  bool unitsSealed = false;
  // Set when the site is solved against the DP table cache.
  std::unique_ptr<SiteTable> table;

  // WMMA-raise analyses. findScalarLoopMatmuls stashes a `ScalarEvolution *`
  // (and SCEVs it owns) into each ScalarLoopHandle for the materializer's
  // SCEVExpander, so they MUST outlive materialization and live here rather
  // than in a collect-local. Declared SE-last so it is destroyed first (it
  // references the others).
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

// The units the DP solves: every unit of the site, or under an accuracy
// target every unit but the matrix products, which errorBudgetSelector picks.
llvm::SmallVector<SiteUnit *> solvedUnits(const FunctionFPState &st,
                                          bool target);

bool collectFPCandidates(Session &S, llvm::Function &F, double errorTol,
                         unsigned sampleLogBits, FunctionFPState &st);

bool materializeFPSolution(Session &S, llvm::Function &F, FunctionFPState &st,
                           llvm::ArrayRef<SolutionStep> steps);

// `errorTol` > 0 is the accuracy target written AT the site; 0 means the site
// wrote none and -poseidon-tau decides.
bool fpOptimize(Session &S, llvm::Function &F, double errorTol = 0.0);

// Joint-DP entry: optimizeSiteBody marks each site with "poseidon-joint-errtol"
// instead of optimizing it inline; this solves one DP over all marked sites.
bool solveJointly(Session &S, llvm::Module &M,
                  llvm::FunctionAnalysisManager &FAM);

} // namespace poseidon
#endif // POSEIDON_OPTIMIZE_H

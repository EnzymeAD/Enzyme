//===- Session.h - the compile of one module ------------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One Session per module the pass compiles: the options of the compile and
// everything the passes record about the module while they run on it. Every
// Poseidon pass that runs on a module reaches the module's session through
// SessionAnalysis, a module analysis that is never invalidated.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_SESSION_H
#define POSEIDON_SOLVE_SESSION_H

#include "Model/CostTable.h"
#include "Sites/LaunchManifest.h"
#include "Sites/SiteIdentity.h"
#include "Solve/Pins.h"
#include "Support/Options.h"
#include "Techniques/Algebraic/HerbieRun.h"
#include "Techniques/Matmul/Matmul.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace poseidon {

/// What one module's compile keeps across Poseidon's passes: its options, the
/// sites found, and the state the solve and the hand-over to the host half
/// build up.
class Session {
public:
  /// A session compiling with `O`; `PB` must outlive it.
  Session(Options O, llvm::PassBuilder &PB, bool IsAsWritten = false)
      : Opts(std::move(O)), PB(PB), IsAsWritten(IsAsWritten),
        CostModel(Opts.Model.CostModel) {}

  /// The options of the compile, as given on its command line.
  const Options Opts;
  /// The pass builder of the compile, which knows its target: what a rewrite
  /// leaves behind is cleaned up with this builder's function simplification
  /// pipeline, as the rest of the module was.
  llvm::PassBuilder &PB;
  /// Whether this is the session of the module as written, compiled as a build
  /// without a profile: it is not snapshotted and hands no launch over.
  const bool IsAsWritten;
  /// A user error ended Poseidon's work on the module: every later Poseidon
  /// pass of the compile leaves it as it is.
  bool Stopped = false;

  /// The canonical clones of the module's site bodies.
  llvm::SmallVector<llvm::WeakVH, 4> SiteClones;
  /// The functions a solve rewrote a site in, with every function that calls
  /// them: what the module as written does not have.
  llvm::SmallVector<llvm::WeakVH, 4> RewrittenFunctions;
  /// Markers on the same body with the same target share one solve, as they
  /// share one profile record.
  std::map<std::pair<const llvm::Function *, double>, llvm::WeakVH> SolvedSites;
  /// The module as it was before Poseidon changed it, compiled as written at
  /// the end of the compile; null when no copy was taken.
  std::unique_ptr<llvm::Module> AsWritten;

  /// The module's site clones and site identities.
  SiteRecords Sites;

  /// Where each site is called from.
  std::map<const llvm::Function *, llvm::DebugLoc> SiteCalls;

  /// The items of -poseidon-pin and the sites they name.
  PinState Pins;

  /// The cost model the compile prices from, resolved on first use.
  CostModelState CostModel;

  /// The Herbie runs of the compile and what they answered.
  HerbieRunState Herbie;

  /// What this device module hands its host half.
  LaunchManifest Launches;

  /// The matrix products found in the module and their pricing.
  MatmulState Matmul;
};

/// The end of a pass a user error stopped: the error is reported, the module is
/// left valid for the passes after this one, and S.Stopped makes the session's
/// later passes leave it as it is. The pass may have changed the module before
/// the error, so no analysis is kept.
llvm::PreservedAnalyses stopOnError(Session &S, llvm::Module &M, llvm::Error E);

/// The session of each module an analysis manager is asked about. Registered
/// with the options of the compile, it makes a session per module; registered
/// with a session, it hands out that one, for the managers a pass runs a
/// nested pipeline of the session's module against.
class SessionAnalysis : public llvm::AnalysisInfoMixin<SessionAnalysis> {
public:
  /// The session of one module: owned by the result when the analysis made it,
  /// borrowed when it was given one.
  struct Result {
    /// The session this result made; null for a given one.
    std::unique_ptr<Session> Owned;
    /// The module's session, owned by `Owned` or by whoever gave it.
    Session *S = nullptr;
    /// The session lasts as long as the module's compile, whatever a pass
    /// changes.
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// Makes a session with `O` for each module it is asked about; `PB` must
  /// outlive the sessions.
  SessionAnalysis(Options O, llvm::PassBuilder &PB)
      : Opts(std::move(O)), PB(&PB) {}
  /// Hands out `S`, which must outlive the analysis, for every module.
  explicit SessionAnalysis(Session &S) : Given(&S) {}
  /// The session of `M`. A made session whose options are invalid is stopped,
  /// with the error reported, before it is returned.
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);

private:
  friend llvm::AnalysisInfoMixin<SessionAnalysis>;
  /// The analysis' identity in an analysis manager.
  static llvm::AnalysisKey Key;
  /// The options a made session compiles with; none when the analysis hands out
  /// a given session.
  std::optional<Options> Opts;
  /// The pass builder a made session uses; null with a given session.
  llvm::PassBuilder *PB = nullptr;
  /// The session handed out for every module; null when one is made per module.
  Session *Given = nullptr;
};

/// The session of `M` from `MAM`, which must have SessionAnalysis registered;
/// it lives as long as the module's compile.
Session &sessionOf(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

/// The analysis managers the plugin's nested pipelines run against. They are
/// registered through the surrounding pipeline's PassBuilder, so that another
/// plugin's pipeline names resolve, and kept local, so that nothing the outer
/// pipeline cached reaches a pass that runs out of order. The session of the
/// module is registered first, so that it is the one they hand out.
struct LocalAnalysisManagers {
  /// The loop, function, CGSCC and module managers, cross-registered.
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  /// Registers `PB`'s analyses and the session `S`, which must outlive the
  /// managers.
  LocalAnalysisManagers(llvm::PassBuilder &PB, Session &S) {
    MAM.registerPass([&S] { return SessionAnalysis(S); });
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }
};

} // namespace poseidon
#endif // POSEIDON_SOLVE_SESSION_H

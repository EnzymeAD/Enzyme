//=- Session.h - the compile of one module --------------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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

#ifndef POSEIDON_SESSION_H
#define POSEIDON_SESSION_H

#include "Core/CostTable.h"
#include "Core/HerbieRun.h"
#include "Core/Options.h"
#include "Sites/LaunchDescriptors.h"
#include "Solvers.h"
#include "Techniques/Matmul/HostDispatch.h"
#include "Techniques/Matmul/MatmulInternal.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace poseidon {

class Session {
public:
  explicit Session(Options O, bool IsTwin = false);

  const Options Opts;
  // The session of the module as written, which FinalizePass compiles as a
  // build without a profile: it is not snapshotted and hands no launch over.
  const bool IsTwin;

  // Driver.cpp and the stages it drives. The canonical clones of the module's
  // site bodies, which finalizeModule drops once nothing calls them.
  llvm::SmallVector<llvm::WeakVH, 4> SiteClones;
  // The functions a solve rewrote a site in, with every function that calls
  // them: what the module as written does not have.
  llvm::SmallVector<llvm::WeakVH, 4> RewrittenFunctions;
  // Markers on the same body with the same target share one solve, as they
  // share one profile record: a body inlined at several call sites otherwise
  // pays the whole solve once per copy.
  std::map<std::pair<const llvm::Function *, double>, llvm::WeakVH> SolvedSites;
  // The module before the pass changed it, for FinalizePass to compile as
  // written.
  std::unique_ptr<llvm::Module> AsWrittenTwin;

  // Sites/SiteIdentity.cpp. The body each site clone was cut from.
  std::map<const llvm::Function *, llvm::Function *> SiteOrigins;
  // The site clone has no callers while it is optimized, so the caller values
  // of its parameters are kept here for the parameter-staging arm to resolve a
  // pointer parameter to its addrspace(3) global.
  std::map<const llvm::Function *, llvm::SmallVector<llvm::WeakTrackingVH, 12>>
      SiteArgs;

  // Pass/Explain.cpp. Where each site is called from.
  std::map<const llvm::Function *, llvm::DebugLoc> SiteCalls;

  // Solvers.cpp. The items of -poseidon-apply-rewrites and the sites they name.
  PinState Pins;

  // Core/CostTable.cpp and Model/CostModel.cpp.
  CostModelState CostModel;

  // Techniques/Herbie.cpp. The platform Herbie searches against, derived from
  // the cost model once it is resolved, and the digest of each Herbie binary
  // run.
  HerbiePlatform Platform;
  bool PlatformResolved = false;
  HerbieBinaryDigests HerbieDigests;

  // Profile/ProfileRead.cpp. Whether each profile directory's profiles name
  // their format, by directory.
  llvm::StringMap<bool> ProfileDirFormats;

  // Sites/LaunchDescriptors.cpp. By kernel name, so that the manifest and its
  // messages come out in one order whatever order the sites were solved in;
  // host dispatch first, then profile generation.
  std::map<std::string, Recorded> RecordedLaunches[2];
  std::map<std::string, ProfileHeader> ProfileHeaders;

  // Techniques/Matmul/HostDispatch.cpp. fpOptimize records the note keyed by
  // the optimized body clone; optimizeSiteBody retrieves it after fpOptimize
  // returns.
  NoteMap<GemmBodyNote> GemmBodies;
  // In the order the sites were split, so that of two sites of one wrapper the
  // same one is recorded last in every compile.
  llvm::MapVector<const llvm::Function *, PendingGemm> PendingGemms;

  // Techniques/Expansion.cpp. The limbs the last two-component applyExpansion
  // produced, and those every wider one produced since resetExpLimbs, for
  // getCompCost to re-root its cost walk on.
  llvm::SmallVector<llvm::Value *, 32> LastExpansionLimbs;
  llvm::SmallVector<llvm::Value *, 32> LastExpLimbs;

  // Techniques/Matmul. The tensor-core shapes the cost model prices, and the
  // warnings already given.
  std::optional<llvm::SmallVector<WmmaTarget, 16>> AvailableWmmaTargets;
  std::set<const void *> WarnedRectangularRows;
  std::set<std::string> ReportedDirectRows;
  std::set<std::string> ReportedInKernelRows;
  bool LegacyMemoEntriesNoted = false;
};

// The session of each module an analysis manager is asked about. Registered
// with the options of the compile, it makes a session per module; registered
// with a session, it hands out that one, for the managers a pass runs a
// nested pipeline of the session's module against.
class SessionAnalysis : public llvm::AnalysisInfoMixin<SessionAnalysis> {
public:
  struct Result {
    std::unique_ptr<Session> Owned;
    Session *S = nullptr;
    // The session lasts as long as the module's compile, whatever a pass
    // changes.
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  explicit SessionAnalysis(Options O);
  explicit SessionAnalysis(Session &S);
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);

private:
  friend llvm::AnalysisInfoMixin<SessionAnalysis>;
  static llvm::AnalysisKey Key;
  std::optional<Options> Opts;
  Session *Given = nullptr;
};

Session &sessionOf(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

} // namespace poseidon
#endif // POSEIDON_SESSION_H

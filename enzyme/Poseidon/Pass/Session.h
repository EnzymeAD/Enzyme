//===- Session.h - the compile of one module ------------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
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

#ifndef POSEIDON_SESSION_H
#define POSEIDON_SESSION_H

#include "Core/CostTable.h"
#include "Core/HerbieRun.h"
#include "Core/Options.h"
#include "Sites/LaunchDescriptors.h"
#include "Solve/Pins.h"
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

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace poseidon {

class Session {
public:
  Session(Options O, llvm::PassBuilder &PB, bool IsTwin = false)
      : Opts(std::move(O)), PB(PB), IsTwin(IsTwin),
        CostModel(Opts.Model.CostModel) {}

  const Options Opts;
  // The pass builder of the compile, which knows its target: what a rewrite
  // leaves behind is cleaned up with this builder's function simplification
  // pipeline, as the rest of the module was.
  llvm::PassBuilder &PB;
  // The session of the module as written, which FinalizePass compiles as a
  // build without a profile: it is not snapshotted and hands no launch over.
  const bool IsTwin;

  // The canonical clones of the module's site bodies.
  llvm::SmallVector<llvm::WeakVH, 4> SiteClones;
  // The functions a solve rewrote a site in, with every function that calls
  // them: what the module as written does not have.
  llvm::SmallVector<llvm::WeakVH, 4> RewrittenFunctions;
  // Markers on the same body with the same target share one solve, as they
  // share one profile record.
  std::map<std::pair<const llvm::Function *, double>, llvm::WeakVH> SolvedSites;
  // The module before the pass changed it, for FinalizePass to compile as
  // written.
  std::unique_ptr<llvm::Module> AsWrittenTwin;

  // Per site clone, the body it was cut from and the caller values of its
  // parameters: the clone has no callers while it is optimized, and the
  // parameter-staging arm resolves a pointer parameter to its addrspace(3)
  // global through them.
  struct SiteClone {
    llvm::Function *Origin;
    llvm::SmallVector<llvm::WeakTrackingVH, 12> Args;
  };
  std::map<const llvm::Function *, SiteClone> SiteCloneOf;
  // The identity (SiteIdentity.h) of each body the pass made for a site and
  // of each marker's target, taken before the optimizer could change its
  // linkage.
  llvm::ValueMap<llvm::Function *, std::string> SiteIdentities;

  // Where each site is called from.
  std::map<const llvm::Function *, llvm::DebugLoc> SiteCalls;

  // Whether the module's sites were warned about as compiled without a
  // profile action.
  bool WarnedWithoutProfile = false;

  // The items of -poseidon-apply-rewrites and the sites they name.
  PinState Pins;

  CostModelState CostModel;

  HerbieRunState Herbie;

  // By kernel name, so that the manifest and its messages come out in one
  // order whatever order the sites were solved in; host dispatch first, then
  // profile generation.
  std::map<std::string, Recorded> RecordedLaunches[2];
  std::map<std::string, ProfileHeader> ProfileHeaders;
  // What the device profiler's tables must hold for this device module: the
  // functions it profiles and the probes of the largest.
  unsigned ProfiledFunctions = 0, ProfiledSlots = 0;

  MatmulState Matmul;
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

  SessionAnalysis(Options O, llvm::PassBuilder &PB)
      : Opts(std::move(O)), PB(&PB) {}
  explicit SessionAnalysis(Session &S) : Given(&S) {}
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);

private:
  friend llvm::AnalysisInfoMixin<SessionAnalysis>;
  static llvm::AnalysisKey Key;
  std::optional<Options> Opts;
  llvm::PassBuilder *PB = nullptr;
  Session *Given = nullptr;
};

Session &sessionOf(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

// The analysis managers the plugin's nested pipelines run against. They are
// registered through the surrounding pipeline's PassBuilder, so that another
// plugin's pipeline names resolve, and kept local, so that nothing the outer
// pipeline cached reaches a pass that runs out of order. The session of the
// module is registered first, so that it is the one they hand out.
struct LocalAnalysisManagers {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
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
#endif // POSEIDON_SESSION_H

//===- PassInternal.h - declarations shared by the pass sources -----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// How every Poseidon pass that runs on a module reaches the module's session:
// SessionAnalysis, a module analysis that is never invalidated (Pipeline.cpp).
// The site actions of the Poseidon passes (Actions.cpp), and the module as
// written beside the rewritten one (AsWritten.cpp): the copy is compiled as a
// build without a profile would compile it, and takes from the rewritten
// module only the functions a rewrite reached.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PASS_PASS_INTERNAL_H
#define POSEIDON_PASS_PASS_INTERNAL_H

#include "Core/Session.h"
#include "Support/Options.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <utility>

namespace llvm {
class CallBase;
class CallInst;
class Function;
class Module;
} // namespace llvm

namespace poseidon {

struct Site;

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

/// Profile use: each of `sites` solved and its call made a call of what the
/// solve chose, appending the calls made to `calls`; in a compile without a
/// profile, the site as written. A marker becomes a call of its body, a kernel
/// a wrapper around its own outlined body.
llvm::Expected<bool>
optimizeSites(Session &S, llvm::ArrayRef<Site> sites,
              llvm::SmallVectorImpl<llvm::CallBase *> &calls);

/// Profile generation: instruments every site of M that is not deferred with
/// the profiling probes and rewrites its marker into an __enzyme_autodiff
/// request. Returns true if it instrumented a site.
llvm::Expected<bool> prepareModule(Session &S, llvm::Module &M);

/// Drop the canonicalized clones nothing calls any more. Runs once per module,
/// after every site is lowered and inlined. An -poseidon-pin item
/// that names no site of the module is an error.
llvm::Error finalizeModule(Session &S, llvm::Module &M);

/// A copy of `M`, in its context, whose values keep the use-list order of the
/// values they copy, which passes that iterate a value's users read: the
/// module as written, to compile as a build without the plugin would.
std::unique_ptr<llvm::Module> snapshotAsWritten(const llvm::Module &M);

/// Compiles `asWrittenModule`, the module as written, with the default pipeline
/// of `Phase` as a build without a profile, takes into it the functions in
/// `rewritten`, those a rewrite reached, and gives M its contents. M is left as
/// it is when the compile of `asWrittenModule` reports an error or a rewritten
/// function refers to what `asWrittenModule` does not hold alike.
void replaceWithAsWritten(llvm::Module &M,
                          std::unique_ptr<llvm::Module> asWrittenModule,
                          const Session &parent, llvm::PassBuilder &PB,
                          llvm::OptimizationLevel Level,
                          llvm::ThinOrFullLTOPhase Phase,
                          llvm::ArrayRef<llvm::Function *> rewritten);

} // namespace poseidon

#endif // POSEIDON_PASS_PASS_INTERNAL_H

//===- Session.h - the compile of one module ------------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One Session per module the pass compiles: the options of the compile and
// everything the passes record about the module while they run on it. What a
// later layer records is a state of its own type, which the session holds in a
// slot (AttachedStates), so that this header names none of those types.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_SESSION_H
#define POSEIDON_CORE_SESSION_H

#include "Support/Options.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"

#include <map>
#include <memory>
#include <utility>

namespace llvm {
class Function;
class PassBuilder;
} // namespace llvm

namespace poseidon {

/// The states attached to an owner, one of each type, each made on first use
/// and owned as long as the owner. A state type T is default-constructible and
/// declares `static inline char ID`, whose address keys it.
class AttachedStates {
public:
  /// The state of type T, default-constructed on first use.
  template <typename T> T &get() { return slot<T>(); }
  /// The state of type T, default-constructed on first use.
  template <typename T> const T &get() const { return slot<T>(); }

private:
  /// Owns one state of any type.
  struct Holder {
    virtual ~Holder();
  };
  /// Owns a state of type T.
  template <typename T> struct HolderOf final : Holder {
    T Value;
  };

  /// The state of type T, made on first use.
  template <typename T> T &slot() const {
    std::unique_ptr<Holder> &H = States[&T::ID];
    if (!H)
      H = std::make_unique<HolderOf<T>>();
    return static_cast<HolderOf<T> &>(*H).Value;
  }

  /// A state made on first use is part of the owner's value, so a const owner
  /// makes one too.
  mutable llvm::DenseMap<const void *, std::unique_ptr<Holder>> States;
};

/// What one module's compile keeps across Poseidon's passes: its options, the
/// sites found, and the state the solve, the techniques and the hand-over to
/// the host half build up.
class Session : public AttachedStates {
public:
  /// A session compiling with `O`; `PB` must outlive it.
  Session(Options O, llvm::PassBuilder &PB, bool IsAsWritten = false)
      : Opts(std::move(O)), PB(PB), IsAsWritten(IsAsWritten) {}

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

  /// Where each site is called from.
  std::map<const llvm::Function *, llvm::DebugLoc> SiteCalls;
};

} // namespace poseidon
#endif // POSEIDON_CORE_SESSION_H

//===- Technique.h - what a technique provides the pass -------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A technique proposes optimization units (Core/Unit.h) for a site; the units
// price their candidates and materialize the one the solver takes. The pass
// knows a technique only through this table: the kind of its units, the option
// that turns it on, and the steps of a compile it takes part in, each called
// for every registered technique in the order techniques() lists them, which
// is also the order of a site's units. A technique keeps its own state.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_TECHNIQUE_H
#define POSEIDON_CORE_TECHNIQUE_H

#include "Core/Unit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <memory>
#include <string>

namespace llvm {
class Function;
class Module;
class Value;
class raw_ostream;
} // namespace llvm

namespace poseidon {

class CacheKey;
class Session;
struct Options;
struct SiteState;
struct SiteProfileData;
struct Subgraph;

/// What a technique keeps for a site or for the compile of a module. A state
/// type derives from TechniqueState and declares `static inline char ID`, whose
/// address keys it.
struct TechniqueState {
  virtual ~TechniqueState();
};

/// The technique states of a site or of a compile, one of each type, each made
/// on first use and owned as long as its owner.
class TechniqueStates {
public:
  /// The state of type T, made empty on first use.
  template <typename T> T &get() {
    std::unique_ptr<TechniqueState> &State = States[&T::ID];
    if (!State)
      State = std::make_unique<T>();
    return static_cast<T &>(*State);
  }

private:
  llvm::DenseMap<const void *, std::unique_ptr<TechniqueState>> States;
};

/// A technique of the pass. Every step but BindUnits is optional (null).
/// WriteProfileHeader, Recognize, BeginSite and the Propose steps are called
/// only when Enabled says the compile proposes the technique's candidates,
/// RemarkUnavailable only when it does not, and the others for every
/// technique, since what they act on may come from another.
struct Technique {
  /// The kind of the technique's units.
  const UnitKindInfo &Unit;
  /// Whether this compile proposes the technique's candidates.
  bool (*Enabled)(const Options &O);
  /// Writes to the header of the profile of the site clone `Clone`, before
  /// its probes are added, the lines the technique reads back from it.
  void (*WriteProfileHeader)(const Options &O, llvm::Function &Clone,
                             llvm::raw_ostream &OS);
  /// Finds, before the site's FP subgraphs are built, the code the technique
  /// takes as a whole; the subgraphs leave it out. Whether it found any.
  bool (*Recognize)(Session &S, llvm::Function &F, const SiteProfileData &P,
                    SiteState &Site);
  /// Reshapes the FP subgraphs of the site `F`, once they are split, into the
  /// subgraphs its units are proposed for.
  void (*ShapeSubgraphs)(const Options &O, llvm::Function &F,
                         llvm::SmallVectorImpl<Subgraph> &Subgraphs);
  /// Prepares the proposals of a site that has FP subgraphs or what Recognize
  /// found.
  llvm::Error (*BeginSite)(Session &S, llvm::Function &F, SiteState &Site);
  /// Says that a site with FP subgraphs gets none of the technique's
  /// candidates because this compile cannot run the technique.
  void (*RemarkUnavailable)(Session &S, llvm::Function &F);
  /// Proposes the technique's units for the site's FP subgraph `Index`.
  llvm::Error (*ProposeForSubgraph)(Session &S, llvm::Function &F,
                                    SiteState &Site, Subgraph &Subgraph,
                                    size_t Index, const SiteProfileData &P);
  /// Proposes the technique's units for what Recognize found.
  llvm::Error (*ProposeForSite)(Session &S, SiteState &Site,
                                const SiteProfileData &P);
  /// Binds the site's units of the technique to the site, in the order they
  /// were proposed, and appends them to its list of units.
  void (*BindUnits)(Session &S, SiteState &Site);
  /// Adds to the cache key of a site's DP table what of this build, beyond its
  /// options and cost model, decides the technique's candidates.
  void (*AddToTableKey)(CacheKey &Key);
  /// The accuracy cost of the site as written that its units price, each
  /// output counted once; adds to `Unpriced` the outputs it leaves out for want
  /// of a price.
  double (*Baseline)(SiteState &Site, unsigned &Unpriced);
  /// Adds to `Why` why the technique proposed no unit for the FP subgraph
  /// `Subgraph` of `Site`, for its explanation.
  void (*ExplainAbsent)(const Options &O, const SiteState &Site,
                        const Subgraph &Subgraph,
                        llvm::SmallVectorImpl<std::string> &Why);
  /// Completes the site function `F` once the steps its solve applied to it
  /// are materialized, before it is cleaned up.
  void (*SiteApplied)(Session &S, llvm::Function &F);
  /// Records that the host launches the solved site `Body` through `Kernel`
  /// with `Args`, the values its parameters take there.
  void (*SiteLaunched)(Session &S, const llvm::Function &Body,
                       llvm::Function &Kernel,
                       llvm::ArrayRef<llvm::Value *> Args);
  /// Reports, once the module is done, what the technique did across sites.
  void (*Report)(Session &S);
  /// Rewrites the host module `M` of a compile that uses a profile from what
  /// the device half recorded; whether it changed M.
  llvm::Expected<bool> (*RewriteHost)(const Options &O, llvm::Module &M);
};

/// The techniques of the pass, in the order of a site's units.
llvm::ArrayRef<Technique> techniques();

} // namespace poseidon
#endif // POSEIDON_CORE_TECHNIQUE_H

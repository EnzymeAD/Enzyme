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
// knows a technique only through the Technique interface: the kind of its
// units, the option that turns it on, and the steps of a compile it takes part
// in, each called for every technique in the order techniques() lists them,
// which is also the order of a site's units. A technique keeps its own state in
// a slot of the SiteState or of the Session. Of the types a later phase
// defines, this header and Core/Unit.h name only those a step merely passes
// through: SiteProfileData, Explanation, UnitExplanation and Alternative.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_TECHNIQUE_H
#define POSEIDON_CORE_TECHNIQUE_H

#include "Core/FPNode.h"
#include "Core/Session.h"
#include "Core/Unit.h"
#include "Support/CacheKey.h"
#include "Support/Options.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace llvm {
class Function;
class Instruction;
class Module;
class Value;
class raw_ostream;
} // namespace llvm

namespace poseidon {

struct SiteProfileData;

/// One site's solve: its FP graph, its profile's scale, the units proposed for
/// it and what each technique keeps for it.
struct SiteState : FPGraph, AttachedStates {
  /// The accuracy target the site is solved against: the one written at the
  /// site, else -poseidon-tau; 0 for none.
  double target = 0.0;
  /// Sum of the profile's per-slot sumSens, times the kappa factor the accuracy
  /// costs carry; the denominator that turns an accuracy cost into a
  /// per-operation relative rounding level.
  double accScale = 0.0;
  /// What the profiled run spent in the site, in cost-model units: the
  /// execution count times the price of every operation the profile records.
  double profiledCost = 0.0;
  /// The factor the accuracy costs were multiplied by to put them on the
  /// application's scale: kappa over the summed sensitivity, 1 without a
  /// measured kappa.
  double scaleFactor = 1.0;
  /// The accumulating operations of the reduction loops a technique
  /// recognized, each of which -poseidon-subgraph-reductions admits as an FP
  /// subgraph of one operation.
  llvm::SmallPtrSet<llvm::Instruction *, 4> reductionOps;
  /// The site's FP subgraphs, the connected components of its operations.
  llvm::SmallVector<Subgraph, 1> subgraphs;
  /// The site's units, technique by technique in the order techniques() lists
  /// them, each bound to this site.
  llvm::SmallVector<SiteUnit *, 8> units;
  /// Whether the site's units are bound and listed in `units`: no technique
  /// proposes a unit for the site after that.
  bool unitsSealed = false;
};

/// A technique of the pass. Every step but unitKind, enabled and bindUnits
/// does nothing unless the technique overrides it. writeProfileHeader,
/// recognize, beginSite and the propose steps are called only when enabled
/// says the compile proposes the technique's candidates, remarkUnavailable
/// only when it does not, and the others for every technique, since what they
/// act on may come from another.
class Technique {
public:
  virtual ~Technique();

  /// The kind of the technique's units.
  virtual const UnitKindInfo &unitKind() const = 0;
  /// Whether this compile proposes the technique's candidates.
  virtual bool enabled(const Options &O) const = 0;
  /// Writes to the header of the profile of the site clone `Clone`, before
  /// its probes are added, the lines the technique reads back from it.
  virtual void writeProfileHeader(const Options &O, llvm::Function &Clone,
                                  llvm::raw_ostream &OS) const {}
  /// Finds, before the site's FP subgraphs are built, the code the technique
  /// takes as a whole; the subgraphs leave it out. Whether it found any.
  virtual bool recognize(Session &S, llvm::Function &F,
                         const SiteProfileData &P, SiteState &Site) const {
    return false;
  }
  /// Reshapes the FP subgraphs of the site `F`, once they are split, into the
  /// subgraphs its units are proposed for.
  virtual void
  shapeSubgraphs(const Options &O, llvm::Function &F,
                 llvm::SmallVectorImpl<Subgraph> &Subgraphs) const {}
  /// Prepares the proposals of a site that has FP subgraphs or what recognize
  /// found.
  virtual llvm::Error beginSite(Session &S, llvm::Function &F,
                                SiteState &Site) const {
    return llvm::Error::success();
  }
  /// Says that a site with FP subgraphs gets none of the technique's
  /// candidates because this compile cannot run the technique.
  virtual void remarkUnavailable(Session &S, llvm::Function &F) const {}
  /// Proposes the technique's units for the site's FP subgraph `Index`.
  virtual llvm::Error proposeForSubgraph(Session &S, llvm::Function &F,
                                         SiteState &Site, Subgraph &Subgraph,
                                         size_t Index,
                                         const SiteProfileData &P) const {
    return llvm::Error::success();
  }
  /// Proposes the technique's units for what recognize found.
  virtual llvm::Error proposeForSite(Session &S, SiteState &Site,
                                     const SiteProfileData &P) const {
    return llvm::Error::success();
  }
  /// Binds the site's units of the technique to the site, in the order they
  /// were proposed, and appends them to its list of units.
  virtual void bindUnits(Session &S, SiteState &Site) const = 0;
  /// Adds to the cache key of a site's DP table what of this build, beyond its
  /// options and cost model, decides the technique's candidates.
  virtual void addToTableKey(CacheKey &Key) const {}
  /// The accuracy cost of the site as written that its units price, each
  /// output counted once; adds to `Unpriced` the outputs it leaves out for want
  /// of a price.
  virtual double baseline(SiteState &Site, unsigned &Unpriced) const {
    return 0.0;
  }
  /// Adds to `Why` why the technique proposed no unit for the FP subgraph
  /// `Subgraph` of `Site`, for its explanation.
  virtual void explainAbsent(const Options &O, const SiteState &Site,
                             const Subgraph &Subgraph,
                             llvm::SmallVectorImpl<std::string> &Why) const {}
  /// Completes the site function `F` once the steps its solve applied to it
  /// are materialized, before it is cleaned up.
  virtual void siteApplied(Session &S, llvm::Function &F) const {}
  /// Records that the host launches the solved site `Body` through `Kernel`
  /// with `Args`, the values its parameters take there.
  virtual void siteLaunched(Session &S, const llvm::Function &Body,
                            llvm::Function &Kernel,
                            llvm::ArrayRef<llvm::Value *> Args) const {}
  /// Reports, once the module is done, what the technique did across sites.
  virtual void report(Session &S) const {}
  /// Rewrites the host module `M` of a compile that uses a profile from what
  /// the device half recorded; whether it changed M.
  virtual llvm::Expected<bool> rewriteHost(const Options &O,
                                           llvm::Module &M) const {
    return false;
  }

protected:
  /// Binds `Units`, the site's units of one technique, to the site in order
  /// and appends them to its list of units.
  template <typename UnitT>
  static void bindAll(Session &S, SiteState &Site,
                      llvm::SmallVectorImpl<UnitT> &Units) {
    for (size_t I = 0; I < Units.size(); ++I) {
      Units[I].bind(S, Site, I);
      Site.units.push_back(&Units[I]);
    }
  }
};

/// The techniques of the pass, in the order of a site's units.
llvm::ArrayRef<const Technique *> techniques();

} // namespace poseidon
#endif // POSEIDON_CORE_TECHNIQUE_H

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
// knows a technique only through this table: its names, the option that turns
// it on, and the steps of a site's collection it takes part in, each called
// for every registered technique in the order techniques() lists them, which
// is also the order of a site's units.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_TECHNIQUE_H
#define POSEIDON_CORE_TECHNIQUE_H

#include "Core/Unit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>

namespace llvm {
class Function;
} // namespace llvm

namespace poseidon {

class Session;
struct Options;
struct SiteState;
struct SiteProfileData;
struct Subgraph;

/// A technique of the pass. Every step but BindUnits is optional (null) and is
/// called only when Enabled says the compile proposes the technique's
/// candidates, RemarkUnavailable only when it does not.
struct Technique {
  /// The names of the technique's units; Unit.kind is its position in
  /// techniques().
  UnitKindInfo Unit;
  /// Whether this compile proposes the technique's candidates.
  bool (*Enabled)(const Options &O);
  /// Finds, before the site's FP subgraphs are built, the code the technique
  /// takes as a whole; the subgraphs leave it out.
  void (*Recognize)(Session &S, llvm::Function &F, const SiteProfileData &P,
                    SiteState &Site);
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
  /// Reports, once the module is done, what the technique did across sites.
  void (*Report)(Session &S);
};

/// The techniques of the pass, in UnitKind order.
llvm::ArrayRef<Technique> techniques();

} // namespace poseidon
#endif // POSEIDON_CORE_TECHNIQUE_H

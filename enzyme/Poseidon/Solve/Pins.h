//=- Pins.h - the items of -poseidon-apply-rewrites -------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// -poseidon-apply-rewrites names the steps a site applies instead of solving
// for them, and a solve prints the value that applies its steps again.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PINS_H
#define POSEIDON_PINS_H

#include "Core/Solver.h"
#include "Techniques/Unit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class Function;
class Module;
} // namespace llvm

namespace poseidon {

struct FunctionFPState;
class Session;

// One item of -poseidon-apply-rewrites: [<site>:]<unit>_<ordinal> or
// [<site>:]<unit>=<name>.
struct ApplyItem {
  std::string text;
  std::string site;
  UnitKind kind = UnitKind::R;
  size_t unit = 0;
  std::optional<size_t> ordinal;
  std::string name;
  // The keys of the module's sites `site` named. Sites are solved as the pass
  // reaches them, so that it named none is only known at the end of the
  // module.
  std::set<std::string> named;
};

struct PinState {
  std::string spec;
  std::vector<ApplyItem> items;
  std::vector<std::string> sites;
  // Two sites cut from one body share a key; printing their pin twice would
  // give a value that applies each item twice.
  std::set<std::string> printed;
};

// The steps the items of -poseidon-apply-rewrites name at the site of `st`.
llvm::SmallVector<SolutionStep> parseManualRewrites(Session &S,
                                                    FunctionFPState &st);

// A site of the module, for the items of -poseidon-apply-rewrites that name
// one; `body` is what the site's clone is made from. An item whose site two
// sites answer to is an error.
void noteSiteForPins(Session &S, const llvm::Function &body);

// Prints the value of -poseidon-apply-rewrites that applies `applied` at the
// site of `st` again.
void printPin(Session &S, const FunctionFPState &st,
              llvm::ArrayRef<SolutionStep> applied);

// At the end of a module: an item that names a site has to have named a site
// of the module, if the module has any.
void checkPinnedSites(Session &S, llvm::Module &M);

} // namespace poseidon
#endif // POSEIDON_PINS_H

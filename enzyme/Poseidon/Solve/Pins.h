//===- Pins.h - the items of -poseidon-pin ---------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// -poseidon-pin names the steps a site applies instead of solving
// for them, and a solve prints the value that applies its steps again.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SOLVE_PINS_H
#define POSEIDON_SOLVE_PINS_H

#include "Core/Unit.h"
#include "Solve/DP.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace llvm {
class Function;
class Module;
} // namespace llvm

namespace poseidon {

struct SiteState;
class Session;

/// One item of -poseidon-pin: [<site>:]<unit>_<ordinal> or
/// [<site>:]<unit>=<name>.
struct PinItem {
  /// The item as written.
  std::string text;
  /// The site the item names; empty when it applies to every site.
  std::string site;
  /// The kind of the unit it names.
  const UnitKindInfo *kind = nullptr;
  /// The index of the unit among the site's units of that kind.
  size_t unit = 0;
  /// The candidate it applies, by ordinal; none when it names one.
  std::optional<size_t> ordinal;
  /// The candidate it applies, by name; empty when it gives an ordinal.
  std::string name;
  /// The keys of the module's sites `site` named. Sites are solved as the pass
  /// reaches them, so that it named none is only known at the end of the
  /// module.
  std::set<std::string> named;
};

/// What the compile knows of its -poseidon-pin items and of the
/// sites it has noted against them.
struct PinState {
  static inline char ID;
  /// Whether `items` holds the items of the compile's -poseidon-pin.
  bool parsed = false;
  /// The items, in the order the flag gives them.
  std::vector<PinItem> items;
  /// The keys of the module's sites noted so far, in the order they were noted.
  std::vector<std::string> sites;
  /// Two sites cut from one body share a key; printing their pin twice would
  /// give a value that applies each item twice.
  std::set<std::string> printed;
};

/// The steps the items of -poseidon-pin name at the site of `st`, in
/// the order a solve of the site returns its steps.
llvm::Expected<llvm::SmallVector<SolutionStep>> parsePins(Session &S,
                                                          SiteState &st);

/// A site of the module, for the items of -poseidon-pin that name
/// one; `body` is what the site's clone is made from. An item whose site two
/// sites answer to is an error.
llvm::Error noteSiteForPins(Session &S, const llvm::Function &body);

/// The value of -poseidon-pin that applies `applied` at the site of
/// `st` again.
std::string pinValue(const Session &S, const SiteState &st,
                     llvm::ArrayRef<SolutionStep> applied);

/// Prints pinValue once per value.
void printPin(Session &S, const SiteState &st,
              llvm::ArrayRef<SolutionStep> applied);

/// At the end of a module: an item that names a site has to have named a site
/// of the module, if the module has any.
llvm::Error checkPinnedSites(Session &S, llvm::Module &M);

} // namespace poseidon
#endif // POSEIDON_SOLVE_PINS_H

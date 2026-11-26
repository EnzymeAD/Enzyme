//===- Pins.cpp - the items of -poseidon-pin ------------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pins.h"
#include "Optimize.h"
#include "Sites/SiteIdentity.h"
#include "Solve/Session.h"
#include "SolveSites.h"
#include "Support/Diagnostics.h"
#include "Support/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace poseidon {

static std::optional<PinItem> parseUnitRef(StringRef s) {
  PinItem item;
  const UnitKindInfo *kind =
      llvm::find_if(unitKinds(), [&](const UnitKindInfo &k) {
        return s.starts_with(k.letter);
      });
  if (kind == unitKinds().end())
    return std::nullopt;
  s = s.drop_front(kind->letter.size());
  item.kind = kind->kind;
  size_t digits = s.find_first_not_of("0123456789");
  if (digits == 0 || digits == StringRef::npos ||
      s.take_front(digits).getAsInteger(10, item.unit))
    return std::nullopt;
  s = s.drop_front(digits);
  if (s.consume_front("=")) {
    if (s.empty())
      return std::nullopt;
    item.name = s.str();
    return item;
  }
  size_t ordinal;
  if (!s.consume_front("_") || s.getAsInteger(10, ordinal))
    return std::nullopt;
  item.ordinal = ordinal;
  return item;
}

// A name may itself hold ':' (a Herbie precision annotation), so the unit is
// read first and a site prefix only when the item does not start with one.
static Expected<PinItem> parseApplyItem(StringRef text, const Function &site) {
  std::optional<PinItem> item = parseUnitRef(text);
  if (!item) {
    auto [site, rest] = text.split(':');
    if (!site.empty() && (item = parseUnitRef(rest)))
      item->site = site.str();
  }
  if (!item)
    return userError(site,
                     "-poseidon-pin: cannot read '" + Twine(text) +
                         "'; an item is [<site>:]<unit>_<ordinal> or "
                         "[<site>:]<unit>=<name>, <unit> being R<i>, P<i> or "
                         "M<i>");
  item->text = text.str();
  return *item;
}

// `site` is the site the items are read for, where a mistake in them is
// reported.
static Expected<std::vector<PinItem> *> applyItems(Session &S,
                                                   const Function &site) {
  PinState &P = S.Pins;
  if (!P.parsed) {
    P.parsed = true;
    SmallVector<StringRef> texts;
    StringRef(S.Opts.Solver.Pin)
        .split(texts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef text : texts) {
      Expected<PinItem> item = parseApplyItem(text.trim(), site);
      if (!item)
        return item.takeError();
      P.items.push_back(std::move(*item));
    }
  }
  return &P.items;
}

static bool namesSite(const PinItem &item, StringRef key,
                      ArrayRef<std::string> aliases) {
  return item.site.empty() || item.site == key ||
         is_contained(aliases, item.site);
}

static std::string listSites(const std::set<std::string> &keys) {
  std::string out;
  for (const std::string &key : keys) {
    out += "\n  " + key;
    std::string shown = llvm::demangle(key);
    if (shown != key)
      out += " (" + shown + ")";
  }
  return out;
}

// Sites are noted before they are solved, so a name two sites answer to is
// refused before the second one applies it.
Error noteSiteForPins(Session &S, const Function &body) {
  if (S.Opts.Solver.Pin.empty())
    return Error::success();
  PinState &P = S.Pins;
  std::string key = siteKeyOfBody(body);
  if (is_contained(P.sites, key))
    return Error::success();
  P.sites.push_back(key);
  SmallVector<std::string, 2> aliases = siteAliases(key);
  Expected<std::vector<PinItem> *> items = applyItems(S, body);
  if (!items)
    return items.takeError();
  for (PinItem &item : **items) {
    if (item.site.empty() || !namesSite(item, key, aliases))
      continue;
    item.named.insert(key);
    if (item.named.size() > 1)
      return userError(body, "-poseidon-pin item '" + Twine(item.text) +
                                 "' names the site '" + item.site +
                                 "', which more than one site of this module "
                                 "answers to; give one of them by its symbol:" +
                                 listSites(item.named));
  }
  return Error::success();
}

Error checkPinnedSites(Session &S, Module &M) {
  PinState &P = S.Pins;
  // The host half of a CUDA or HIP unit is given the items too, and the sites
  // they name are the device half's. A noted site read the items.
  if (!P.sites.empty())
    for (const PinItem &item : P.items)
      if (!item.site.empty() && item.named.empty())
        return userError(
            "-poseidon-pin item '" + Twine(item.text) + "' names the site '" +
            item.site + "', which is no site of this module; its sites are:" +
            listSites(std::set<std::string>(P.sites.begin(), P.sites.end())));
  P.sites.clear();
  P.printed.clear();
  for (PinItem &item : P.items)
    item.named.clear();
  return Error::success();
}

// The position of the candidate an item names in a unit's list. A candidate
// pricing dropped is not in the list, and is told apart from a name the unit
// never had.
static Expected<size_t> findNamed(const PinItem &item, const Function &F,
                                  StringRef site, StringRef unit,
                                  const SiteUnit &cands) {
  std::optional<size_t> found;
  for (size_t i = 0; i < cands.numCandidates(); ++i) {
    if (cands.candidateName(i) != item.name)
      continue;
    if (found)
      return userError(F, "-poseidon-pin: " + Twine(site) + ": " + unit +
                              " has more than one candidate named '" +
                              item.name + "'; give it as " + unit +
                              "_<ordinal>");
    found = i;
  }
  if (found)
    return *found;
  for (const auto &[name, why] : cands.droppedCandidates())
    if (name == item.name)
      return userError(F, "-poseidon-pin: " + Twine(site) + ": " + unit +
                              " has no candidate named '" + item.name +
                              "': pricing dropped it, because " + why);
  std::string names;
  for (size_t i = 0; i < cands.numCandidates(); ++i)
    names += "\n  " + cands.candidateName(i);
  if (names.empty())
    names = " none";
  return userError(F, "-poseidon-pin: " + Twine(site) + ": " + unit +
                          " has no candidate named '" + item.name +
                          "'; its candidates are:" + names);
}

Expected<SmallVector<SolutionStep>> parsePins(Session &S, SiteState &st) {
  const std::string key = siteKey(S.Sites, *st.F);
  const SmallVector<std::string, 2> aliases = siteAliases(key);
  SmallVector<SolutionStep> steps;
  SmallPtrSet<const SiteUnit *, 8> appliedUnits;
  const Function &F = *st.F;
  Expected<std::vector<PinItem> *> items = applyItems(S, F);
  if (!items)
    return items.takeError();
  for (const PinItem &item : **items) {
    if (!namesSite(item, key, aliases))
      continue;
    const std::string &id = item.text;
    const UnitKindInfo &kind = infoOf(item.kind);
    SiteUnit *unit = nullptr;
    size_t ofKind = 0;
    for (SiteUnit *u : st.units) {
      if (u->unitKind() != item.kind)
        continue;
      if (u->indexInKind() == item.unit)
        unit = u;
      ++ofKind;
    }
    const std::string name = (kind.letter + Twine(item.unit)).str();
    if (!unit) {
      std::string has = "none";
      if (ofKind == 1)
        has = (kind.letter + "0").str();
      else if (ofKind > 1)
        has = (kind.letter + "0 to " + kind.letter + Twine(ofKind - 1)).str();
      return userError(F, "-poseidon-pin item '" + Twine(id) +
                              "': there is no " + name + "; the site has " +
                              has);
    }
    size_t pos;
    if (item.ordinal) {
      std::optional<size_t> found = unit->positionOfOrdinal(*item.ordinal);
      if (!found && kind.pricingDrops)
        return userError(F, "-poseidon-pin item '" + Twine(id) + "': " + name +
                                " has no candidate " + Twine(*item.ordinal) +
                                " that pricing kept");
      if (!found)
        return userError(F, "-poseidon-pin item '" + Twine(id) + "': " + name +
                                " has no candidate " + Twine(*item.ordinal) +
                                "; it has " + Twine(unit->numCandidates()));
      pos = *found;
    } else {
      Expected<size_t> named = findNamed(item, F, key, name, *unit);
      if (!named)
        return named.takeError();
      pos = *named;
    }
    if (!appliedUnits.insert(unit).second)
      return userError(F, "-poseidon-pin item '" + Twine(id) + "': " + name +
                              " already has " + kind.appliedNoun +
                              " applied; give one item per unit");
    // The rule the solvers keep: a raise replaces the loop it reads, so no
    // other step may edit an instruction of that loop.
    for (const SolutionStep &prior : steps) {
      const SiteUnit &other = unitOf(prior);
      if ((unit->consumesFootprint() || other.consumesFootprint()) &&
          unit->footprintOverlaps(other))
        return userError(F, "-poseidon-pin item '" + Twine(id) + "': " + name +
                                " and " + other.info().letter +
                                Twine(other.indexInKind()) +
                                " change the same instructions, and one of "
                                "them replaces them; give one of the two");
    }
    if (S.Opts.Diagnostics.Print)
      llvm::errs() << "[poseidon] Selecting " << id << ": "
                   << unit->selectionText(pos) << '\n';
    steps.emplace_back(unit, pos);
  }
  // The steps apply in the order a solve would return them, so a pin builds
  // the program the solve that printed it built.
  const SmallVector<OptimizationUnit *> order =
      solveOrder(S.Opts, ArrayRef<SiteUnit *>(st.units));
  llvm::stable_sort(steps, [&](const SolutionStep &a, const SolutionStep &b) {
    return find(order, a.unit) < find(order, b.unit);
  });
  return steps;
}

std::string pinValue(const Session &S, const SiteState &st,
                     ArrayRef<SolutionStep> applied) {
  const std::string key = siteKey(S.Sites, *st.F);
  std::string value;
  for (const SolutionStep &step : applied) {
    const SiteUnit &unit = unitOf(step);
    std::string item = (unit.info().letter + Twine(unit.indexInKind()) + "=" +
                        unit.candidateName(step.candidateIndex))
                           .str();
    if (!value.empty())
      value += ",";
    value += key + ":" + item;
  }
  return value;
}

void printPin(Session &S, const SiteState &st, ArrayRef<SolutionStep> applied) {
  const std::string value = pinValue(S, st, applied);
  if (!value.empty() && !S.Pins.printed.insert(value).second)
    return;
  if (S.Opts.Diagnostics.Print)
    llvm::errs() << "[poseidon] pin: " << value << '\n';
}

} // namespace poseidon

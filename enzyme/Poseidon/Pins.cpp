//=- Pins.cpp - the items of -poseidon-apply-rewrites -----------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pins.h"
#include "Optimize.h"
#include "Pass/Session.h"
#include "Sites/SiteIdentity.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace poseidon {

static std::optional<ApplyItem> parseUnitRef(StringRef s) {
  ApplyItem item;
  const UnitKindInfo *kind =
      llvm::find_if(unitKinds(), [&](const UnitKindInfo &k) {
        return s.starts_with(k.idPrefix);
      });
  if (kind == unitKinds().end())
    return std::nullopt;
  s = s.drop_front(kind->idPrefix.size());
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
static ApplyItem parseApplyItem(StringRef text) {
  std::optional<ApplyItem> item = parseUnitRef(text);
  if (!item) {
    auto [site, rest] = text.split(':');
    if (!site.empty() && (item = parseUnitRef(rest)))
      item->site = site.str();
  }
  if (!item)
    report_fatal_error("[poseidon] -poseidon-apply-rewrites: cannot read '" +
                       Twine(text) +
                       "'; an item is [<site>:]<unit>_<ordinal> or "
                       "[<site>:]<unit>=<name>, <unit> being R<i>, PT<i> or "
                       "M<i>");
  item->text = text.str();
  return *item;
}

static std::vector<ApplyItem> &applyItems(Session &S) {
  PinState &P = S.Pins;
  if (P.spec != S.Opts.Solver.ApplyRewrites) {
    P.spec = S.Opts.Solver.ApplyRewrites;
    P.items.clear();
    SmallVector<StringRef> texts;
    StringRef(P.spec).split(texts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef text : texts)
      P.items.push_back(parseApplyItem(text.trim()));
  }
  return P.items;
}

static bool namesSite(const ApplyItem &item, StringRef key,
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
void noteSiteForPins(Session &S, const Function &body) {
  if (S.Opts.Solver.ApplyRewrites.empty())
    return;
  PinState &P = S.Pins;
  std::string key = siteKeyOfBody(body);
  if (is_contained(P.sites, key))
    return;
  P.sites.push_back(key);
  SmallVector<std::string, 2> aliases = siteAliases(key);
  for (ApplyItem &item : applyItems(S)) {
    if (item.site.empty() || !namesSite(item, key, aliases))
      continue;
    item.named.insert(key);
    if (item.named.size() > 1)
      report_fatal_error("[poseidon] -poseidon-apply-rewrites item '" +
                             Twine(item.text) + "' names the site '" +
                             item.site +
                             "', which more than one site of this module "
                             "answers to; give one of them by its symbol:" +
                             listSites(item.named),
                         /*gen_crash_diag=*/false);
  }
}

void checkPinnedSites(Session &S) {
  PinState &P = S.Pins;
  // The host half of a CUDA or HIP unit is given the items too, and the sites
  // they name are the device half's.
  if (!P.sites.empty())
    for (const ApplyItem &item : applyItems(S))
      if (!item.site.empty() && item.named.empty())
        report_fatal_error(
            "[poseidon] -poseidon-apply-rewrites item '" + Twine(item.text) +
                "' names the site '" + item.site +
                "', which is no site of this module. Its sites are:" +
                listSites(
                    std::set<std::string>(P.sites.begin(), P.sites.end())),
            /*gen_crash_diag=*/false);
  P.sites.clear();
  P.printed.clear();
  for (ApplyItem &item : P.items)
    item.named.clear();
}

// The position of the candidate an item names in a unit's list. A candidate
// pricing dropped is not in the list, and is told apart from a name the unit
// never had.
static size_t findNamed(const ApplyItem &item, StringRef site, StringRef unit,
                        const SiteUnit &cands) {
  std::optional<size_t> found;
  for (size_t i = 0; i < cands.numCandidates(); ++i) {
    if (cands.candidateName(i) != item.name)
      continue;
    if (found)
      report_fatal_error("[poseidon] " + Twine(site) + ": " + unit +
                             " has more than one candidate named '" +
                             item.name + "'; give it as " + unit + "_<ordinal>",
                         /*gen_crash_diag=*/false);
    found = i;
  }
  if (found)
    return *found;
  for (const auto &[name, why] : cands.droppedCandidates())
    if (name == item.name)
      report_fatal_error("[poseidon] " + Twine(site) + ": " + unit +
                             " has no candidate named '" + item.name +
                             "': pricing dropped it, because " + why,
                         /*gen_crash_diag=*/false);
  std::string names;
  for (size_t i = 0; i < cands.numCandidates(); ++i)
    names += "\n  " + cands.candidateName(i);
  if (names.empty())
    names = " none";
  report_fatal_error("[poseidon] " + Twine(site) + ": " + unit +
                         " has no candidate named '" + item.name +
                         "'. Its candidates are:" + names,
                     /*gen_crash_diag=*/false);
}

SmallVector<SolutionStep> parseManualRewrites(Session &S, FunctionFPState &st) {
  const std::string key = siteKey(S, *st.F);
  const SmallVector<std::string, 2> aliases = siteAliases(key);
  SmallVector<SolutionStep> steps;
  SmallPtrSet<const SiteUnit *, 8> appliedUnits;
  for (const ApplyItem &item : applyItems(S)) {
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
    if (!unit)
      report_fatal_error("[poseidon] " + Twine(kind.tag) + " index " +
                         Twine(item.unit) + " out of range (" + Twine(ofKind) +
                         " " + kind.tag + "s)");
    size_t pos;
    if (item.ordinal) {
      std::optional<size_t> found = unit->positionOfOrdinal(*item.ordinal);
      if (!found && kind.pricingDrops)
        report_fatal_error("[poseidon] " + Twine(kind.tag) + " " +
                           Twine(item.unit) + " has no candidate " +
                           Twine(*item.ordinal) + " that pricing kept ('" +
                           Twine(id) + "')");
      if (!found)
        report_fatal_error("[poseidon] candidate index " +
                           Twine(*item.ordinal) + " out of range for " +
                           kind.tag + " " + Twine(item.unit) + " (" +
                           Twine(unit->numCandidates()) + " candidates)");
      pos = *found;
    } else {
      pos =
          findNamed(item, key, (kind.idPrefix + Twine(item.unit)).str(), *unit);
    }
    if (!appliedUnits.insert(unit).second)
      report_fatal_error("[poseidon] " + Twine(kind.tag) + " " +
                         Twine(item.unit) + " already has " + kind.appliedNoun +
                         " applied (duplicate ID '" + Twine(id) + "')");
    llvm::errs() << "[poseidon] Selecting " << id << ": "
                 << unit->selectionText(pos) << "\n";
    steps.emplace_back(unit, pos);
  }
  return steps;
}

void printPin(Session &S, const FunctionFPState &st,
              ArrayRef<SolutionStep> applied) {
  const std::string key = siteKey(S, *st.F);
  std::string value;
  for (const SolutionStep &step : applied) {
    const SiteUnit &unit = unitOf(step);
    std::string item = (unit.info().idPrefix + Twine(unit.indexInKind()) + "=" +
                        unit.candidateName(step.candidateIndex))
                           .str();
    if (!value.empty())
      value += ",";
    value += key + ":" + item;
  }
  if (!value.empty() && !S.Pins.printed.insert(value).second)
    return;
  llvm::errs() << "[poseidon] pin: " << value << "\n";
}

} // namespace poseidon

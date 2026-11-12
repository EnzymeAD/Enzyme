//=- SiteTable.h - a site's DP table in the cache ---------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A site's solved DP table is cached in table.json under the key of everything
// it is computed from; a later compile with the same key binds the table to
// its own candidates instead of pricing and solving again.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_SITE_TABLE_H
#define POSEIDON_SITE_TABLE_H

#include "Core/CacheKey.h"
#include "Core/Solver.h"
#include "Techniques/Unit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace poseidon {

struct FunctionFPState;
struct Options;
class Session;

// A site's DP table in the cache: the key of what it is solved from and, when
// the cache holds a table under that key, the table bound to this compile's
// candidates.
struct SiteTable {
  CacheKey key;
  bool hit = false;
  CostMap costToAccuracyMap;
  SolutionMap costToSolutionMap;
  std::optional<double> baseline;
};

CacheKey solveKey(Session &S, const FunctionFPState &st,
                  llvm::ArrayRef<SiteUnit *> units,
                  llvm::StringRef profilePath);

bool readSiteTable(const Options &O, llvm::StringRef fnName, SiteTable &table,
                   llvm::ArrayRef<SiteUnit *> units);

void writeSiteTable(const Options &O, llvm::StringRef fnName,
                    const CacheKey &key, const CostMap &costToAccuracyMap,
                    const SolutionMap &costToSolutionMap, double baseline,
                    double profiledCost);

} // namespace poseidon
#endif // POSEIDON_SITE_TABLE_H

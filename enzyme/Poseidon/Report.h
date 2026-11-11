//=- Report.h - the DP table print and the report files ---------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What a single-site solve leaves for a reader: the DP table on stderr
// (-poseidon-show-table) and the files of -poseidon-report-path.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_REPORT_H
#define POSEIDON_REPORT_H

#include "Core/Solver.h"
#include "Techniques/Unit.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace poseidon {

class Session;

void printDPTable(const CostScale &scale, const CostMap &costToAccuracyMap,
                  const SolutionMap &costToSolutionMap);

void emitReport(Session &S, llvm::StringRef funcName, const CostScale &scale,
                const CostMap &costToAccuracyMap,
                const SolutionMap &costToSolutionMap,
                llvm::ArrayRef<SiteUnit *> units);

} // namespace poseidon
#endif // POSEIDON_REPORT_H

//===- EnzymeSummary.h - Per-function facts for separate differentiation --===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A summary of what one function does to floating-point data, computed from
// its body alone and without differentiating anything. Calls are kept
// symbolic (as edges naming the callee and parameter), so a caller's summary
// does not depend on its callees' bodies, and summaries can be computed per
// module, per translation unit or per Julia CodeInstance and combined later.
//
// The shape follows the per-method summary of ActivityAnalysis.jl: effects
// per argument and on globals, a flow matrix from arguments to arguments'
// memory and the return value, and flags for unknown effects and escapes.
//
// See EnzymeSummary.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_SUMMARY_H
#define ENZYME_SUMMARY_H

#include "PassUtils.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Support/JSON.h"

#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace llvm {
class Function;
class Module;
} // namespace llvm

/// What a function may do to the memory reachable from one of its arguments,
/// not counting what its callees do with it (see EnzymeFunctionSummary::Edges).
struct EnzymeArgEffects {
  /// May read floating-point data from it.
  bool ReadFP = false;
  /// May write floating-point data to it. A store of a constant counts: it
  /// overwrites a value that may have a derivative.
  bool WriteFP = false;
  /// May write data of any type to it.
  bool WriteAny = false;
  /// Memory reachable from it may become reachable from another argument's
  /// memory, the return value or a global (a row of PointsTo).
  bool Escape = false;
};

/// A pointer argument of a call site: the memory it points into, and whether
/// that memory may be written again after the call (loops included).
struct EnzymeCallArg {
  /// "a<i>" (reachable from argument i), "g<name>" (a global), "l" (local
  /// memory) or "u" (unknown or several); empty for a non-pointer argument.
  std::string Root;
  bool WrittenAfter = false;
};

struct EnzymeCallSite {
  std::string Callee;
  std::vector<EnzymeCallArg> Args;
};

/// The facts about one function that differentiating its callers needs,
/// computed from its own body. Pure: computing it does not change the IR.
struct EnzymeFunctionSummary {
  //===--------------------------------------------------------------------===//
  // Effects on data, before composing with callees
  //===--------------------------------------------------------------------===//

  /// One entry per argument of the function.
  std::vector<EnzymeArgEffects> Args;
  /// Globals the function may read or write floating-point data in, and
  /// globals it may write data of any type to.
  std::set<std::string> GlobalsReadFP, GlobalsWriteFP, GlobalsWriteAny;
  /// Flow[s][t]: data from source s may reach sink t.
  ///  Sources: argument i (its value, or memory reachable from it) for
  ///           0 <= i < n, then any global (index n).
  ///  Sinks:   memory reachable from argument i for 0 <= i < n, then the
  ///           return value (index n), then any global (index n + 1).
  /// What callees do with memory passed to them is not included: that is
  /// what Edges is for. Data without a root in the caller (passed by value,
  /// or held in local objects) that is given to a callee is assumed to reach
  /// everything the callee may write.
  /// Matches ActivityAnalysis.jl's ActivityDescriptor.matrix (rows and
  /// columns below n, and the return column) and escape_data (the globals
  /// column).
  std::vector<std::vector<bool>> Flow;
  /// PointsTo[s][t]: memory reachable from source s may become reachable
  /// from sink t (a pointer to it is stored there, or returned). Same shape
  /// as Flow; matches ActivityDescriptor.pts_matrix.
  std::vector<std::vector<bool>> PointsTo;
  /// The function has effects the summary cannot describe (indirect calls,
  /// atomics, pointers of unknown origin), so treat it as touching anything.
  bool Unknown = false;
  /// It may write memory of unknown origin.
  bool UnknownWrite = false;
  bool Frees = false;
  bool ReturnsFP = false;
  bool ReturnsPointer = false;
  /// (root, callee, parameter): memory rooted at root ("a<i>" or
  /// "g<name>") is reachable from what callee is given as that parameter.
  /// A global named "*" is one that cannot be named (a constant inttoptr
  /// address, as Julia emits for its objects) or memory a callee returned.
  std::set<std::tuple<std::string, std::string, unsigned>> Edges;
  /// Per call site, what each pointer argument points into and whether it
  /// may be overwritten after the call.
  std::vector<EnzymeCallSite> CallsAt;

  //===--------------------------------------------------------------------===//
  // Structure
  //===--------------------------------------------------------------------===//

  std::string Linkage;
  unsigned Insts = 0;
  unsigned IndirectCalls = 0;
  /// Direct callees, functions whose address is used, and globals used.
  std::set<std::string> Calls, Refs, Globals;
  bool TouchesFP = false;
  bool MemTransfer = false;
  bool Allocates = false;
  /// "enzyme_type" declared on each parameter and on the return value
  /// (empty if none).
  std::vector<std::string> ArgTypes;
  std::string RetType;
  bool Inactive = false;
  bool NoFree = false;
  bool NoEscapingAllocation = false;

  unsigned numArgs() const { return Args.size(); }
  unsigned globalsSource() const { return numArgs(); }
  unsigned returnSink() const { return numArgs(); }
  unsigned globalsSink() const { return numArgs() + 1; }

  llvm::json::Object toJSON() const;
};

/// Compute the summary of a function with a body.
EnzymeFunctionSummary summarizeFunction(llvm::Function &F);

/// The summary as a function analysis, for passes that want it cached and
/// invalidated by the pass manager.
class EnzymeFunctionSummaryAnalysis
    : public llvm::AnalysisInfoMixin<EnzymeFunctionSummaryAnalysis> {
  friend llvm::AnalysisInfoMixin<EnzymeFunctionSummaryAnalysis>;
  static llvm::AnalysisKey Key;

public:
  using Result = EnzymeFunctionSummary;
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &);
};

/// The facts about a whole module a thin-link step needs: every function's
/// summary, the __enzyme_* calls and registrations, and the globals.
llvm::json::Object summarizeModule(llvm::Module &M);

/// Prints the summary of each function with a body as one line of JSON:
/// -passes='print<enzyme-function-summary>'.
class EnzymeFunctionSummaryPrinterPass final
    : public PassParent<EnzymeFunctionSummaryPrinterPass> {
  friend PassParent<EnzymeFunctionSummaryPrinterPass>;
  static llvm::AnalysisKey Key;
  llvm::raw_ostream &OS;

public:
  explicit EnzymeFunctionSummaryPrinterPass(llvm::raw_ostream &OS) : OS(OS) {}
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
  static bool isRequired() { return true; }
};

/// The enzyme-summary pass: writes summarizeModule as JSON to
/// -enzyme-summary-out (stdout if empty).
class EnzymeSummaryNewPM final : public PassParent<EnzymeSummaryNewPM> {
  friend PassParent<EnzymeSummaryNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);
  static bool isRequired() { return true; }
};

#endif

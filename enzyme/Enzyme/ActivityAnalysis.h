//===- ActivityAnalysis.h - Declaration of Activity Analysis  -----------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file consists of two mutually recurive
// functions that compute this for values and instructions, respectively.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_ACTIVE_VAR_H
#define ENZYME_ACTIVE_VAR_H 1

#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Support/CommandLine.h"

#include "llvm/IR/InstVisitor.h"

#include "TypeAnalysis/TypeAnalysis.h"

extern "C" {
extern llvm::cl::opt<bool> EnzymePrintActivity;
extern llvm::cl::opt<bool> EnzymeNonmarkedGlobalsInactive;
}

class PreProcessCache;

// A map of MPI comm allocators (otherwise inactive) to the
// argument of the Comm* they allocate into.
extern const std::map<std::string, size_t> MPIInactiveCommAllocators;

/// Helper class to analyze the differential activity
class ActivityAnalyzer {
  PreProcessCache &PPC;

  /// Aliasing Information
  llvm::AAResults &AA;
  /// Library Information
  llvm::TargetLibraryInfo &TLI;

public:
  /// Whether the returns of the function being analyzed are active
  const bool ActiveReturns;

private:
  /// Direction of current analysis
  const uint8_t directions;
  /// Analyze up based off of operands
  static constexpr uint8_t UP = 1;
  /// Analyze down based off uses
  static constexpr uint8_t DOWN = 2;

  /// Instructions that don't propagate adjoints
  /// These instructions could return an active pointer, but
  /// do not propagate adjoints themselves
  llvm::SmallPtrSet<llvm::Instruction *, 4> ConstantInstructions;

  /// Instructions that could propagate adjoints
  llvm::SmallPtrSet<llvm::Instruction *, 20> ActiveInstructions;

  /// Values that do not contain derivative information, either
  /// directly or as a pointer to
  llvm::SmallPtrSet<llvm::Value *, 4> ConstantValues;

  /// Values that may contain derivative information
  llvm::SmallPtrSet<llvm::Value *, 2> ActiveValues;

public:
  /// Construct the analyzer from the a previous set of constant and active
  /// values and whether returns are active. The all arguments of the functions
  /// being analyzed must be in the set of constant and active values, lest an
  /// error occur during analysis
  ActivityAnalyzer(PreProcessCache &PPC, llvm::AAResults &AA_,
                   llvm::TargetLibraryInfo &TLI_,
                   const llvm::SmallPtrSetImpl<llvm::Value *> &ConstantValues,
                   const llvm::SmallPtrSetImpl<llvm::Value *> &ActiveValues,
                   bool ActiveReturns)
      : PPC(PPC), AA(AA_), TLI(TLI_), ActiveReturns(ActiveReturns),
        directions(UP | DOWN),
        ConstantValues(ConstantValues.begin(), ConstantValues.end()),
        ActiveValues(ActiveValues.begin(), ActiveValues.end()) {}

  /// Return whether this instruction is known not to propagate adjoints
  /// Note that instructions could return an active pointer, but
  /// do not propagate adjoints themselves
  bool isConstantInstruction(TypeResults &TR, llvm::Instruction *inst);

  /// Return whether this values is known not to contain derivative
  // information, either directly or as a pointer to
  bool isConstantValue(TypeResults &TR, llvm::Value *val);

private:
  std::map<llvm::Instruction *, std::set<llvm::Value *>>
      ReEvaluateValueIfInactiveInst;
  std::map<llvm::Value *, std::set<llvm::Value *>>
      ReEvaluateValueIfInactiveValue;

  std::map<llvm::Value *, std::set<llvm::Instruction *>>
      ReEvaluateInstIfInactiveValue;

  void InsertConstantInstruction(TypeResults &TR, llvm::Instruction *I);
  void InsertConstantValue(TypeResults &TR, llvm::Value *V);

  /// Create a new analyzer starting from an existing Analyzer
  /// This is used to perform inductive assumptions
  ActivityAnalyzer(ActivityAnalyzer &Other, uint8_t directions)
      : PPC(Other.PPC), AA(Other.AA), TLI(Other.TLI),
        ActiveReturns(Other.ActiveReturns), directions(directions),
        ConstantInstructions(Other.ConstantInstructions),
        ActiveInstructions(Other.ActiveInstructions),
        ConstantValues(Other.ConstantValues), ActiveValues(Other.ActiveValues) {
    assert(directions != 0);
    assert((directions & Other.directions) == directions);
    assert((directions & Other.directions) != 0);
  }

  /// Import known constants from an existing analyzer
  void insertConstantsFrom(TypeResults &TR, ActivityAnalyzer &Hypothesis) {
    for (auto I : Hypothesis.ConstantInstructions) {
      InsertConstantInstruction(TR, I);
    }
    for (auto V : Hypothesis.ConstantValues) {
      InsertConstantValue(TR, V);
    }
  }

  /// Import known data from an existing analyzer
  void insertAllFrom(TypeResults &TR, ActivityAnalyzer &Hypothesis,
                     llvm::Value *Orig) {
    insertConstantsFrom(TR, Hypothesis);
    for (auto I : Hypothesis.ActiveInstructions) {
      bool inserted = ActiveInstructions.insert(I).second;
      if (inserted && directions == 3) {
        ReEvaluateInstIfInactiveValue[Orig].insert(I);
      }
    }
    for (auto V : Hypothesis.ActiveValues) {
      bool inserted = ActiveValues.insert(V).second;
      if (inserted && directions == 3) {
        ReEvaluateValueIfInactiveValue[Orig].insert(V);
      }
    }
  }

  /// Is the use of value val as an argument of call CI known to be inactive
  bool isFunctionArgumentConstant(llvm::CallInst *CI, llvm::Value *val);

  /// Is the instruction guaranteed to be inactive because of its operands
  bool isInstructionInactiveFromOrigin(TypeResults &TR, llvm::Value *val);

public:
  enum class UseActivity {
    // No Additional use activity info
    None = 0,

    // Only consider loads of memory
    OnlyLoads = 1,

    // Ignore potentially active stores
    OnlyStores = 2
  };
  /// Is the value free of any active uses
  bool isValueInactiveFromUsers(TypeResults &TR, llvm::Value *val,
                                UseActivity UA,
                                llvm::Instruction **FoundInst = nullptr);

  /// Is the value potentially actively returned or stored
  bool isValueActivelyStoredOrReturned(TypeResults &TR, llvm::Value *val,
                                       bool outside = false);

private:
  /// StoredOrReturnedCache acts as an inductive cache of results for
  /// isValueActivelyStoredOrReturned
  std::map<llvm::Value *, bool> StoredOrReturnedCache;
};

#endif

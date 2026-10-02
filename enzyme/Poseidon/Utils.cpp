//=- Utils.cpp - Utility functions for Poseidon optimization pass ----------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for the Poseidon floating-point
// optimization pass.
//
//===----------------------------------------------------------------------===//

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <mpfr.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "llvm/Support/Debug.h"

#include "Flags.h"
#include "Types.h"
#include "Utils.h"

#define DEBUG_TYPE "poseidon"

using namespace llvm;

namespace poseidon {

static bool isTargetNVPTX(const Module &M) {
  return Triple(M.getTargetTriple()).isNVPTX();
}

// TODO: handle amd
bool isGPUMode(const Function &F) { return isTargetNVPTX(*F.getParent()); }

void simplifyFunction(Function &F, OptimizationLevel Level) {

  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  if (verifyFunction(F, &llvm::errs())) {
    F.print(llvm::errs());
    llvm_unreachable("Poseidon intermediate function failed verification");
  }

  FunctionPassManager FPM =
      PB.buildFunctionSimplificationPipeline(Level, ThinOrFullLTOPhase::None);
  (void)FPM.run(F, FAM);
}

const std::unordered_set<std::string> &libmFuncs() {
  static const std::unordered_set<std::string> LibmFuncs = {
      "sin",      "cos",   "tan",   "asin",     "acos",  "atan",   "atan2",
      "sinh",     "cosh",  "tanh",  "asinh",    "acosh", "atanh",  "exp",
      "log",      "sqrt",  "cbrt",  "pow",      "powi",  "fabs",   "fma",
      "hypot",    "expm1", "log1p", "ceil",     "floor", "erf",    "exp2",
      "lgamma",   "log10", "log2",  "rint",     "round", "tgamma", "trunc",
      "copysign", "fdim",  "fmod",  "remainder"};
  return LibmFuncs;
}

StringRef deviceMathName(StringRef fnName) {
  static const std::unordered_set<std::string> Bases = [] {
    std::unordered_set<std::string> bases = libmFuncs();
    // getLLValue emits these as llvm.minnum/llvm.maxnum, which NVPTX selects.
    bases.insert("fmin");
    bases.insert("fmax");
    return bases;
  }();

  StringRef base = fnName;
  if (!base.consume_front("__nv_"))
    return "";
  if (Bases.count(base.str()))
    return base;
  if (base.ends_with("f") && Bases.count(base.drop_back(1).str()))
    return base.drop_back(1);
  return "";
}

double getOneULP(double value) {
  assert(!std::isnan(value) && !std::isinf(value));

  double next = std::nextafter(value, std::numeric_limits<double>::infinity());
  double ulp = std::fabs(next - value);

  return ulp;
}

std::string getLibmFunctionForPrecision(StringRef funcName, Type *newType) {
  std::string baseName = funcName.str();

  std::string prefix;
  if (baseName.size() > 5 && baseName.substr(0, 5) == "__nv_") {
    prefix = "__nv_";
    baseName = baseName.substr(5);
  }

  if (!baseName.empty() && (baseName.back() == 'f' || baseName.back() == 'l')) {
    baseName.pop_back();
  }

  if (libmFuncs().count(baseName)) {
    if (newType->isHalfTy() || newType->isBFloatTy() || newType->isFloatTy()) {
      return prefix + baseName + "f";
    } else if (newType->isDoubleTy()) {
      return prefix + baseName;
    } else if (newType->isFP128Ty() || newType->isX86_FP80Ty()) {
      return prefix + baseName + "l";
    }
  }

  return "";
}

double stringToDouble(const std::string &str) {
  char *end;
  errno = 0;
  double result = std::strtod(str.c_str(), &end);

  if (errno == ERANGE) {
    if (result == HUGE_VAL) {
      result = std::numeric_limits<double>::infinity();
    } else if (result == -HUGE_VAL) {
      result = -std::numeric_limits<double>::infinity();
    }
  }

  return result; // Denormalized values are fine
}

void topoSort(const SetVector<Instruction *> &insts,
              SmallVectorImpl<Instruction *> &instsSorted) {
  SmallPtrSet<Instruction *, 8> visited;
  SmallPtrSet<Instruction *, 8> onStack;

  std::function<void(Instruction *)> dfsVisit = [&](Instruction *I) {
    if (visited.count(I))
      return;
    visited.insert(I);
    onStack.insert(I);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
    for (auto &op : operands) {
      if (isa<Instruction>(op)) {
        Instruction *oI = cast<Instruction>(op);
        if (insts.contains(oI)) {
          if (onStack.count(oI)) {
            llvm_unreachable(
                "topoSort: Cycle detected in instruction dependencies!");
          }
          dfsVisit(oI);
        }
      }
    }

    onStack.erase(I);
    instsSorted.push_back(I);
  };

  for (auto *I : insts) {
    if (!visited.count(I)) {
      dfsVisit(I);
    }
  }
}

void reverseTopoSort(const SetVector<Instruction *> &insts,
                     SmallVectorImpl<Instruction *> &instsSorted) {
  topoSort(insts, instsSorted);
  std::reverse(instsSorted.begin(), instsSorted.end());
}

void getUniqueArgs(const std::string &expr, SmallSet<std::string, 8> &args) {
  std::regex argPattern("v\\d+");

  std::sregex_iterator begin(expr.begin(), expr.end(), argPattern);
  std::sregex_iterator end;

  while (begin != end) {
    args.insert(begin->str());
    ++begin;
  }
}

void collectExprInsts(Value *V, const SetVector<Value *> &inputs,
                      SmallPtrSetImpl<Instruction *> &exprInsts,
                      SmallPtrSetImpl<Value *> &visited) {
  if (!V || inputs.contains(V) || visited.contains(V)) {
    return;
  }

  visited.insert(V);

  if (auto *I = dyn_cast<Instruction>(V)) {
    exprInsts.insert(I);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();

    for (auto &op : operands) {
      collectExprInsts(op, inputs, exprInsts, visited);
    }
  }
}

bool isExpansionBottleneck(Instruction *I, const Subgraph &subgraph) {
  if (subgraph.outputs.contains(I) && subgraph.outputs.size() <= 1) {
    return false;
  }

  unsigned internalUses = 0;
  for (auto *U : I->users()) {
    if (auto *UI = dyn_cast<Instruction>(U)) {
      if (subgraph.operations.contains(UI)) {
        ++internalUses;
      }
    }
  }

  if (internalUses < flags::MinUsesSplit) {
    return false;
  }

  SetVector<Instruction *> relevantTree;
  SmallVector<Instruction *, 16> worklist;
  worklist.push_back(I);
  relevantTree.insert(I);

  while (!worklist.empty()) {
    Instruction *current = worklist.pop_back_val();
    auto operands = isa<CallInst>(current) ? cast<CallInst>(current)->args()
                                           : current->operands();
    for (auto &op : operands) {
      if (subgraph.inputs.contains(op)) {
        continue;
      }
      if (auto *OpI = dyn_cast<Instruction>(op)) {
        if (subgraph.operations.contains(OpI) && !relevantTree.contains(OpI)) {
          worklist.push_back(OpI);
          relevantTree.insert(OpI);
        }
      }
    }
  }

  SetVector<Instruction *> movedOps;
  SetVector<Instruction *> keptSubtreeRoots;

  for (auto *op : relevantTree) {
    if (op == I) {
      movedOps.insert(op);
      continue;
    }

    bool hasExternalUse = false;
    for (auto *U : op->users()) {
      if (auto *UI = dyn_cast<Instruction>(U)) {
        if (!relevantTree.contains(UI)) {
          hasExternalUse = true;
          break;
        }
      }
    }

    if (hasExternalUse) {
      keptSubtreeRoots.insert(op);
    } else {
      movedOps.insert(op);
    }
  }

  for (auto *keptRoot : keptSubtreeRoots) {
    SetVector<Instruction *> keptUpstream;
    SmallVector<Instruction *, 16> keptWorklist;
    keptWorklist.push_back(keptRoot);
    keptUpstream.insert(keptRoot);

    while (!keptWorklist.empty()) {
      Instruction *current = keptWorklist.pop_back_val();
      auto operands = isa<CallInst>(current) ? cast<CallInst>(current)->args()
                                             : current->operands();
      for (auto &op : operands) {
        if (subgraph.inputs.contains(op)) {
          continue;
        }
        if (auto *OpI = dyn_cast<Instruction>(op)) {
          if (relevantTree.contains(OpI) && !keptUpstream.contains(OpI)) {
            keptWorklist.push_back(OpI);
            keptUpstream.insert(OpI);
            movedOps.remove(OpI);
          }
        }
      }
    }
  }

  bool isBottleneck = movedOps.size() >= flags::MinOpsSplit;
  if (flags::Print && isBottleneck) {
    llvm::errs() << "Bottleneck: " << *I << "\n";
    llvm::errs() << "Num of operations that would be moved: " << movedOps.size()
                 << " (>=" << flags::MinOpsSplit << ")\n";
    llvm::errs() << "Num of internal uses: " << internalUses
                 << " (>=" << flags::MinUsesSplit << ")\n";
    llvm::errs() << "Operations that would be moved:\n";
    for (auto *op : movedOps) {
      llvm::errs() << "\t" << *op << "\n";
    }
  }
  return isBottleneck;
}

SetVector<Value *>
findReachedInputs(const SetVector<Instruction *> &operations) {
  SetVector<Value *> reachedInputs;

  for (auto *I : operations) {
    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
    for (auto &op : operands) {
      if (auto *OpI = dyn_cast<Instruction>(op)) {
        if (operations.contains(OpI)) {
          continue;
        }
      }

      reachedInputs.insert(op);
    }
  }

  return reachedInputs;
}

void splitSubgraphAtBottleneck(Subgraph &currentSubgraph,
                               Instruction *bottleneck, Subgraph &newSubgraph,
                               Subgraph &remainingSubgraph) {

  SetVector<Instruction *> relevantTree;
  SmallVector<Instruction *, 16> worklist;
  worklist.push_back(bottleneck);
  relevantTree.insert(bottleneck);

  while (!worklist.empty()) {
    auto current = worklist.pop_back_val();
    auto operands = isa<CallInst>(current) ? cast<CallInst>(current)->args()
                                           : current->operands();
    for (auto &op : operands) {
      if (currentSubgraph.inputs.contains(op)) {
        continue;
      }
      if (auto *OpI = dyn_cast<Instruction>(op)) {
        if (currentSubgraph.operations.contains(OpI) &&
            !relevantTree.contains(OpI)) {
          worklist.push_back(OpI);
          relevantTree.insert(OpI);
        }
      }
    }
  }

  SetVector<Instruction *> movedOps;
  SetVector<Instruction *> keptSubtreeRoots;

  for (auto op : relevantTree) {
    if (op == bottleneck) {
      movedOps.insert(op);
      continue;
    }

    bool hasExternalUse = false;
    for (auto U : op->users()) {
      if (auto UI = dyn_cast<Instruction>(U)) {
        if (!relevantTree.contains(UI)) {
          hasExternalUse = true;
          break;
        }
      }
    }

    if (hasExternalUse) {
      keptSubtreeRoots.insert(op);
    } else {
      movedOps.insert(op);
    }
  }

  for (auto keptRoot : keptSubtreeRoots) {
    SetVector<Instruction *> keptUpstream;
    SmallVector<Instruction *, 16> keptWorklist;
    keptWorklist.push_back(keptRoot);
    keptUpstream.insert(keptRoot);

    while (!keptWorklist.empty()) {
      Instruction *current = keptWorklist.pop_back_val();
      auto operands = isa<CallInst>(current) ? cast<CallInst>(current)->args()
                                             : current->operands();
      for (auto &op : operands) {
        if (currentSubgraph.inputs.contains(op)) {
          continue;
        }
        if (auto *OpI = dyn_cast<Instruction>(op)) {
          if (relevantTree.contains(OpI) && !keptUpstream.contains(OpI)) {
            keptWorklist.push_back(OpI);
            keptUpstream.insert(OpI);
            movedOps.remove(OpI);
          }
        }
      }
    }
  }

  newSubgraph.operations = movedOps;
  newSubgraph.outputs.insert(bottleneck);
  newSubgraph.inputs = findReachedInputs(newSubgraph.operations);

  for (auto I : currentSubgraph.operations) {
    if (!movedOps.contains(I)) {
      remainingSubgraph.operations.insert(I);
    }
  }

  remainingSubgraph.outputs = currentSubgraph.outputs;
  if (currentSubgraph.outputs.contains(bottleneck)) {
    remainingSubgraph.outputs.remove(bottleneck);
  }

  remainingSubgraph.inputs = findReachedInputs(remainingSubgraph.operations);
  assert(remainingSubgraph.inputs.contains(bottleneck));
}

// The memory object a value is loaded from / stored to when that object is a
// staging buffer (pointer argument or addrspace(3) global), else nullptr.
static const Value *stagingObjectOf(const Value *Ptr) {
  const Value *Obj = getUnderlyingObject(Ptr);
  if (!Obj)
    return nullptr;
  if (isa<Argument>(Obj) && Obj->getType()->isPointerTy())
    return Obj;
  if (auto *GV = dyn_cast<GlobalVariable>(Obj))
    if (GV->getAddressSpace() == 3)
      return Obj;
  return nullptr;
}

void mergeSharedStagingSubgraphs(SmallVectorImpl<Subgraph> &subgraphs,
                                 Function &F) {
  if (subgraphs.size() < 2)
    return;

  SmallVector<SmallPtrSet<const Value *, 4>, 8> touched(subgraphs.size());
  for (size_t i = 0; i < subgraphs.size(); ++i) {
    auto add = [&](const Value *Ptr) {
      if (const Value *O = stagingObjectOf(Ptr))
        touched[i].insert(O);
    };
    for (Value *In : subgraphs[i].inputs)
      if (auto *LI = dyn_cast<LoadInst>(In))
        add(LI->getPointerOperand());
    for (Instruction *Op : subgraphs[i].operations)
      for (Value *V : Op->operands())
        if (auto *LI = dyn_cast<LoadInst>(V))
          add(LI->getPointerOperand());
    for (Instruction *Out : subgraphs[i].outputs)
      for (User *U : Out->users())
        if (auto *SI = dyn_cast<StoreInst>(U))
          if (SI->getValueOperand() == Out)
            add(SI->getPointerOperand());
  }

  SmallVector<unsigned, 8> parent(subgraphs.size());
  for (unsigned i = 0; i < parent.size(); ++i)
    parent[i] = i;
  std::function<unsigned(unsigned)> find = [&](unsigned x) -> unsigned {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](unsigned a, unsigned b) {
    a = find(a);
    b = find(b);
    if (a != b)
      parent[b] = a;
  };
  for (size_t i = 0; i < subgraphs.size(); ++i)
    for (size_t j = i + 1; j < subgraphs.size(); ++j)
      for (const Value *O : touched[i])
        if (touched[j].count(O)) {
          unite(i, j);
          break;
        }

  MapVector<unsigned, SmallVector<unsigned, 8>> groups;
  for (unsigned i = 0; i < subgraphs.size(); ++i)
    groups[find(i)].push_back(i);
  if (groups.size() == subgraphs.size())
    return; // nothing shares a buffer

  SmallVector<Subgraph, 8> merged;
  for (auto &kv : groups) {
    if (kv.second.size() == 1) {
      merged.push_back(subgraphs[kv.second.front()]);
      continue;
    }
    Subgraph M;
    for (unsigned idx : kv.second) {
      for (Instruction *Op : subgraphs[idx].operations)
        M.operations.insert(Op);
      for (Instruction *Out : subgraphs[idx].outputs)
        M.outputs.insert(Out);
    }
    // An input of one member may be produced by another member; those edges
    // are internal to the merged unit and must not be re-declared as inputs.
    for (unsigned idx : kv.second)
      for (Value *In : subgraphs[idx].inputs) {
        auto *I = dyn_cast<Instruction>(In);
        if (I && (M.operations.contains(I) || M.outputs.contains(I)))
          continue;
        M.inputs.insert(In);
      }
    if (flags::Print) {
      SmallPtrSet<const Value *, 4> objs;
      for (unsigned idx : kv.second)
        for (const Value *O : touched[idx])
          objs.insert(O);
      llvm::errs() << "[poseidon] merged " << kv.second.size()
                   << " FP subgraphs sharing " << objs.size()
                   << " staging buffer(s) into one optimization unit ("
                   << M.operations.size() << " operations, " << M.inputs.size()
                   << " inputs, " << M.outputs.size() << " outputs) in "
                   << F.getName() << "\n";
    }
    merged.push_back(std::move(M));
  }

  subgraphs.clear();
  for (auto &M : merged)
    subgraphs.push_back(std::move(M));
}

void splitSubgraphs(SmallVectorImpl<Subgraph> &subgraphs) {

  SmallVector<Subgraph, 8> resultSubgraphs;
  SmallVector<Subgraph, 8> workQueue;

  for (const auto &subgraph : subgraphs) {
    workQueue.push_back(subgraph);
  }

  while (!workQueue.empty()) {
    Subgraph currentSubgraph = workQueue.pop_back_val();

    if (currentSubgraph.operations.size() <= 2) {
      resultSubgraphs.push_back(currentSubgraph);
      continue;
    }

    SmallVector<Instruction *, 8> sortedOps;
    topoSort(currentSubgraph.operations, sortedOps);

    bool madeSplit = false;
    for (auto *I : sortedOps) {
      assert(currentSubgraph.operations.contains(I));

      if (isExpansionBottleneck(I, currentSubgraph)) {
        if (flags::Print) {
          llvm::errs() << "Bottleneck: " << *I << "\n";
          llvm::errs() << "currentSubgraph.inputs ("
                       << currentSubgraph.inputs.size() << "): ";
          for (auto *input : currentSubgraph.inputs) {
            llvm::errs() << "\t" << *input << "\n";
          }
          llvm::errs() << "\n";
          llvm::errs() << "currentSubgraph.operations ("
                       << currentSubgraph.operations.size() << "): ";
          for (auto *op : currentSubgraph.operations) {
            llvm::errs() << "\t" << *op << "\n";
          }
          llvm::errs() << "\n";
          llvm::errs() << "currentSubgraph.outputs ("
                       << currentSubgraph.outputs.size() << "): ";
          for (auto *output : currentSubgraph.outputs) {
            llvm::errs() << "\t" << *output << "\n";
          }
          llvm::errs() << "\n";
        }

        Subgraph newSubgraph, remainingSubgraph;
        splitSubgraphAtBottleneck(currentSubgraph, I, newSubgraph,
                                  remainingSubgraph);

        if (flags::Print) {
          llvm::errs() << "=== Splitting subgraph at bottleneck: " << *I
                       << " ===\n";

          llvm::errs() << "  New subgraph:\n";
          llvm::errs() << "    Inputs (" << newSubgraph.inputs.size() << "):\n";
          for (auto *input : newSubgraph.inputs) {
            llvm::errs() << "      " << *input << "\n";
          }
          llvm::errs() << "    Operations (" << newSubgraph.operations.size()
                       << "):\n";
          for (auto *op : newSubgraph.operations) {
            llvm::errs() << "      " << *op << "\n";
          }
          llvm::errs() << "    Outputs (" << newSubgraph.outputs.size()
                       << "):\n";
          for (auto *output : newSubgraph.outputs) {
            llvm::errs() << "      " << *output << "\n";
          }

          llvm::errs() << "  Remaining subgraph:\n";

          llvm::errs() << "    Inputs (" << remainingSubgraph.inputs.size()
                       << "):\n";
          for (auto *input : remainingSubgraph.inputs) {
            llvm::errs() << "      " << *input << "\n";
          }
          llvm::errs() << "    Operations ("
                       << remainingSubgraph.operations.size() << "):\n";
          for (auto *op : remainingSubgraph.operations) {
            llvm::errs() << "      " << *op << "\n";
          }
          llvm::errs() << "    Outputs (" << remainingSubgraph.outputs.size()
                       << "):\n";
          for (auto *output : remainingSubgraph.outputs) {
            llvm::errs() << "      " << *output << "\n";
          }
        }

        resultSubgraphs.push_back(newSubgraph);
        currentSubgraph = remainingSubgraph;
        madeSplit = true;
      }
    }

    if (madeSplit) {
      workQueue.push_back(currentSubgraph);
    } else {
      resultSubgraphs.push_back(currentSubgraph);
    }
  }

  if (flags::Print) {
    llvm::errs() << "=== Subgraph splitting complete ===\n";
    llvm::errs() << "  Original subgraphs: " << subgraphs.size() << "\n";
    llvm::errs() << "  Final subgraphs after splitting: "
                 << resultSubgraphs.size() << "\n";
  }

  subgraphs = std::move(resultSubgraphs);
}

} // namespace poseidon

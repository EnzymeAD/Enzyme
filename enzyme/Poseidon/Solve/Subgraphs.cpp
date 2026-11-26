//===- Subgraphs.cpp - the FP subgraphs of a site -------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/FPCore.h"
#include "Core/FPNode.h"
#include "Core/NumberFormat.h"
#include "Core/Technique.h"
#include "Sites/Canonicalize.h"
#include "Solve.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Support/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

using namespace llvm;

namespace poseidon {

// The dtype of a node of type T, which an optimizable instruction and its
// operands only ever have as float or double.
static const NumberFormatInfo *nodeFormat(Type *T) {
  const NumberFormatInfo *F = formatOf(T);
  assert(F &&
         (F->Kind == NumberFormat::FP32 || F->Kind == NumberFormat::FP64) &&
         "an optimizable instruction computes float or double");
  return F;
}

// Whether the profiling run executed `I`, an optimizable instruction of the
// canonical clone.
static bool executed(const Instruction &I, const ProfileMap &profileMap) {
  std::optional<size_t> slot = slotOf(&I);
  if (!slot)
    return true;
  auto it = profileMap.find(*slot);
  return it != profileMap.end() && it->second.exec != 0;
}

// Names in a missed-optimization remark the operations of F the profiling run
// never executed, which are left as written, by their source lines.
static void reportUnexecuted(const Function &F,
                             ArrayRef<const Instruction *> unexecuted) {
  std::set<std::pair<std::string, unsigned>> lines;
  for (const Instruction *I : unexecuted)
    if (const DebugLoc &DL = I->getDebugLoc())
      lines.insert({DL->getFilename().str(), DL.getLine()});
  std::string where;
  raw_string_ostream os(where);
  if (lines.empty())
    os << "no source location";
  ListSeparator sep;
  for (const auto &[file, line] : lines)
    os << sep << sys::path::filename(file) << ':' << line;
  remarkMissed(F, F.getSubprogram(), "Unexecuted",
               "Poseidon left " + Twine(unexecuted.size()) +
                   " floating-point operation(s) as written that the "
                   "profiling run never executed (" +
                   where + "); profile an input that runs them");
}

// The inputs of a subgraph of `operations`, in the order the operations read
// them: what they read that is neither one of them nor a floating-point
// literal.
static SetVector<Value *> inputsOf(const SetVector<Instruction *> &operations) {
  SetVector<Value *> inputs;
  for (Instruction *I : operations)
    for (Value *op : fpOperands(*I)) {
      auto *OpI = dyn_cast<Instruction>(op);
      if ((OpI && operations.contains(OpI)) || isa<ConstantFP>(op))
        continue;
      inputs.insert(op);
    }
  return inputs;
}

void buildSubgraphs(const Options &O, const ProfileMap &profileMap,
                    SiteState &st) {
  Function &F = *st.F;
  FPGraph &G = st;
  SmallVectorImpl<Subgraph> &subgraphs = st.subgraphs;
  auto &valueToNodeMap = G.valueToNodeMap;
  // An operation the run never executed is left as written, as an unexecuted
  // matrix product is: the profile gives it no range, which would leave its
  // whole subgraph unpriced. What an executed operation reads was computed on
  // the way to it, so an unexecuted one only ever ends a subgraph as the user
  // of its outputs.
  auto inSubgraph = [&](const Value &V) {
    return isOptimizable(V) && executed(cast<Instruction>(V), profileMap);
  };
  SmallVector<const Instruction *, 8> unexecuted;

  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] Starting Floodfill for " << F.getName() << '\n';

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isOptimizable(I)) {
        valueToNodeMap[&I] = G.create<FPLLValue>(&I, nullptr, nullptr);
        if (O.Diagnostics.Print)
          llvm::errs() << "[poseidon] not an optimizable operation: " << I
                       << '\n';
        continue;
      }

      valueToNodeMap[&I] = G.create<FPLLValue>(
          &I, fpcoreOperator(fpcoreName(I)), nodeFormat(I.getType()));
    }
  }

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isOptimizable(I))
        continue;
      auto node = valueToNodeMap[&I];
      for (auto &operand : fpOperands(I)) {
        if (!valueToNodeMap.count(operand)) {
          if (auto Arg = dyn_cast<Argument>(operand)) {
            valueToNodeMap[operand] =
                G.create<FPLLValue>(Arg, nullptr, nodeFormat(Arg->getType()));
          } else if (auto C = dyn_cast<ConstantFP>(operand)) {
            SmallString<10> value;
            C->getValueAPF().toString(value);
            valueToNodeMap[operand] =
                G.create<FPConst>(value.c_str(), nodeFormat(C->getType()));
          } else if (auto CI = dyn_cast<ConstantInt>(operand)) {
            // e.g., powi intrinsic has a constant int as its exponent
            double exponent = static_cast<double>(CI->getSExtValue());
            std::string doubleStr = std::to_string(exponent);
            valueToNodeMap[operand] = G.create<FPConst>(
                doubleStr.c_str(), &formatOf(NumberFormat::FP64));
          } else {
            llvm_unreachable("isOptimizable admits no other operand");
          }
        }
        node->addOperand(valueToNodeMap[operand]);
      }
    }
  }

  SmallSet<Value *, 8> processed;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!inSubgraph(I)) {
        if (isOptimizable(I))
          unexecuted.push_back(&I);
        continue;
      }

      if (processed.contains(&I))
        continue;

      SmallVector<Value *, 8> todo;
      SetVector<Instruction *> output_seen;
      SetVector<Instruction *> operation_seen;
      todo.push_back(&I);
      while (!todo.empty()) {
        auto cur = todo.pop_back_val();
        assert(valueToNodeMap.count(cur) && "Node not found in valueToNodeMap");

        auto I2 = cast<Instruction>(cur);

        if (operation_seen.contains(I2))
          continue;

        assert(!processed.contains(cur) &&
               "an operation is flood-filled into one subgraph only");

        operation_seen.insert(I2);
        processed.insert(cur);

        for (const auto &operand : fpOperands(*I2))
          if (inSubgraph(*operand))
            todo.push_back(operand);

        for (auto U : I2->users()) {
          if (auto I3 = dyn_cast<Instruction>(U)) {
            if (!inSubgraph(*I3))
              output_seen.insert(I2);
            else
              todo.push_back(I3);
          }
        }
      }

      if (!operation_seen.empty()) {
        SetVector<Value *> input_seen = inputsOf(operation_seen);
        if (O.Diagnostics.Print)
          llvm::errs() << "[poseidon] Found a subgraph with "
                       << operation_seen.size() << " operations and "
                       << input_seen.size() << " inputs and "
                       << output_seen.size() << " outputs\n";

        if (operation_seen.size() == 1 &&
            !(O.Candidates.SubgraphReductions &&
              st.reductionOps.contains(*operation_seen.begin()))) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[poseidon] Skipping trivial subgraph\n";
          continue;
        }
        if (operation_seen.size() == 1 && O.Diagnostics.Print)
          llvm::errs() << "[poseidon] Keeping 1-op reduction subgraph: "
                       << **operation_seen.begin() << '\n';

        subgraphs.emplace_back(input_seen, output_seen, operation_seen);
      }
    }
  }

  if (!unexecuted.empty())
    reportUnexecuted(F, unexecuted);

  if (O.Diagnostics.Print) {
    llvm::errs() << "[poseidon] Found " << subgraphs.size()
                 << " initial subgraphs in " << F.getName() << '\n';
  }
}

Error attachProfile(const Options &O, const ProfileMap &profileMap, FPGraph &G,
                    ArrayRef<Subgraph> subgraphs) {
  auto &valueToNodeMap = G.valueToNodeMap;
  for (auto &subgraph : subgraphs) {
    for (auto op : subgraph.operations) {
      std::optional<size_t> slot = slotOf(op);
      if (!slot)
        continue;
      auto it = profileMap.find(*slot);
      assert(it != profileMap.end() &&
             "an operation without a profile record is in no subgraph");
      const auto &profileInfo = it->second;

      auto node = valueToNodeMap[op];
      node->sens = profileInfo.sumSens;
      node->grad = profileInfo.sumAbsGrad;
      node->executions = profileInfo.exec;
      node->updateBounds(profileInfo.minRes, profileInfo.maxRes);

      for (const auto &[i, operand] : enumerate(fpOperands(*op))) {
        if (i >= profileInfo.minOperands.size()) {
          if (isa<Constant>(operand))
            continue;
          std::string name;
          raw_string_ostream(name) << *operand;
          return userError(*op, "the profile gives this operation " +
                                    Twine(profileInfo.minOperands.size()) +
                                    " operand range(s), and none for its "
                                    "operand " +
                                    Twine(i) + " (" + StringRef(name).trim() +
                                    "); re-run profile generation for this "
                                    "program");
        }
        valueToNodeMap[operand]->updateBounds(profileInfo.minOperands[i],
                                              profileInfo.maxOperands[i]);
      }
    }
  }
  return Error::success();
}

// The operations a split at I moves into a subgraph of their own: I and the
// operations of its operand tree within `subgraph` that neither have a user
// outside the tree nor feed one that has.
static SetVector<Instruction *> movedOperations(const Subgraph &subgraph,
                                                Instruction *I) {
  SetVector<Instruction *> relevantTree;
  SmallVector<Instruction *, 16> worklist;
  worklist.push_back(I);
  relevantTree.insert(I);

  while (!worklist.empty()) {
    Instruction *current = worklist.pop_back_val();
    for (auto &op : fpOperands(*current)) {
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
      for (auto &op : fpOperands(*current)) {
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
  return movedOps;
}

static bool isSplitPoint(const Options &O, Instruction *I,
                         const Subgraph &subgraph) {
  if (subgraph.outputs.contains(I) && subgraph.outputs.size() <= 1) {
    return false;
  }
  const bool forced =
      isa<CallInst>(I) &&
      is_contained(O.Candidates.SubgraphSplitCalls, fpcoreName(*I));

  unsigned internalUses = 0;
  for (auto *U : I->users()) {
    if (auto *UI = dyn_cast<Instruction>(U)) {
      if (subgraph.operations.contains(UI)) {
        ++internalUses;
      }
    }
  }

  if (!forced && (O.Candidates.SubgraphSplitUses == 0 ||
                  internalUses < O.Candidates.SubgraphSplitUses)) {
    return false;
  }

  SetVector<Instruction *> movedOps = movedOperations(subgraph, I);

  bool isBottleneck =
      forced || movedOps.size() >= O.Candidates.SubgraphSplitOps;
  if (O.Diagnostics.Print && isBottleneck) {
    llvm::errs() << "[poseidon] Bottleneck: " << *I << '\n';
    llvm::errs() << "  Num of operations that would be moved: "
                 << movedOps.size() << " (>=" << O.Candidates.SubgraphSplitOps
                 << ")\n";
    llvm::errs() << "  Num of internal uses: " << internalUses
                 << " (>=" << O.Candidates.SubgraphSplitUses << ")\n";
    llvm::errs() << "  Operations that would be moved:\n";
    for (auto *op : movedOps) {
      llvm::errs() << '\t' << *op << '\n';
    }
  }
  return isBottleneck;
}

static void splitSubgraphAtBottleneck(const Subgraph &currentSubgraph,
                                      Instruction *bottleneck,
                                      Subgraph &newSubgraph,
                                      Subgraph &remainingSubgraph) {
  SetVector<Instruction *> movedOps =
      movedOperations(currentSubgraph, bottleneck);

  newSubgraph.operations = movedOps;
  newSubgraph.outputs.insert(bottleneck);
  newSubgraph.inputs = inputsOf(newSubgraph.operations);

  for (auto I : currentSubgraph.operations) {
    if (!movedOps.contains(I)) {
      remainingSubgraph.operations.insert(I);
    }
  }

  remainingSubgraph.outputs = currentSubgraph.outputs;
  if (currentSubgraph.outputs.contains(bottleneck)) {
    remainingSubgraph.outputs.remove(bottleneck);
  }

  remainingSubgraph.inputs = inputsOf(remainingSubgraph.operations);
  assert(remainingSubgraph.inputs.contains(bottleneck) &&
         "the split-off bottleneck is an input of the remaining subgraph");
}

static void splitSubgraphs(const Options &O,
                           SmallVectorImpl<Subgraph> &subgraphs) {
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
      assert(currentSubgraph.operations.contains(I) &&
             "a split moves only operations upstream of the bottleneck");

      if (isSplitPoint(O, I, currentSubgraph)) {
        Subgraph newSubgraph, remainingSubgraph;
        splitSubgraphAtBottleneck(currentSubgraph, I, newSubgraph,
                                  remainingSubgraph);

        if (O.Diagnostics.Print) {
          llvm::errs() << "[poseidon] Splitting subgraph at bottleneck: " << *I
                       << '\n';

          llvm::errs() << "  New subgraph:\n";
          llvm::errs() << "    Inputs (" << newSubgraph.inputs.size() << "):\n";
          for (auto *input : newSubgraph.inputs) {
            llvm::errs() << "      " << *input << '\n';
          }
          llvm::errs() << "    Operations (" << newSubgraph.operations.size()
                       << "):\n";
          for (auto *op : newSubgraph.operations) {
            llvm::errs() << "      " << *op << '\n';
          }
          llvm::errs() << "    Outputs (" << newSubgraph.outputs.size()
                       << "):\n";
          for (auto *output : newSubgraph.outputs) {
            llvm::errs() << "      " << *output << '\n';
          }

          llvm::errs() << "  Remaining subgraph:\n";

          llvm::errs() << "    Inputs (" << remainingSubgraph.inputs.size()
                       << "):\n";
          for (auto *input : remainingSubgraph.inputs) {
            llvm::errs() << "      " << *input << '\n';
          }
          llvm::errs() << "    Operations ("
                       << remainingSubgraph.operations.size() << "):\n";
          for (auto *op : remainingSubgraph.operations) {
            llvm::errs() << "      " << *op << '\n';
          }
          llvm::errs() << "    Outputs (" << remainingSubgraph.outputs.size()
                       << "):\n";
          for (auto *output : remainingSubgraph.outputs) {
            llvm::errs() << "      " << *output << '\n';
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

  if (O.Diagnostics.Print)
    llvm::errs() << "  Final subgraphs after splitting: "
                 << resultSubgraphs.size() << '\n';

  subgraphs = std::move(resultSubgraphs);
}

void shapeSubgraphs(const Options &O, Function &F,
                    SmallVectorImpl<Subgraph> &subgraphs) {
  splitSubgraphs(O, subgraphs);

  if (O.Diagnostics.Print) {
    llvm::errs() << "[poseidon] After splitting, have " << subgraphs.size()
                 << " subgraphs in " << F.getName() << '\n';
  }

  // After splitting, so that a reshaped subgraph is never split again.
  for (const Technique *T : techniques())
    T->shapeSubgraphs(O, F, subgraphs);
}

} // namespace poseidon

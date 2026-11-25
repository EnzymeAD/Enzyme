//===- Subgraphs.cpp - the FP subgraphs of a site -------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/Options.h"
#include "IR/Utils.h"
#include "Model/FPCore.h"
#include "Optimize.h"
#include "Pass/Session.h"
#include "Sites/Canonicalize.h"
#include "Sites/SiteIdentity.h"
#include "Targets/Target.h"
#include "Techniques/Matmul/Matmul.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
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
static const FPFormatInfo *nodeFormat(Type *T) {
  const FPFormatInfo *F = formatOf(T);
  assert(F &&
         (F->Precision == PrecisionChangeType::FP32 ||
          F->Precision == PrecisionChangeType::FP64) &&
         "an optimizable instruction computes float or double");
  return F;
}

// Whether the profiling run executed `I`, an optimizable instruction of the
// canonical clone.
static bool executed(const Instruction &I, const ProfileMap &profileMap) {
  size_t idx;
  if (!tryReadProfIdxMetadata(&I, idx))
    return true;
  auto it = profileMap.find(idx);
  return it != profileMap.end() && it->second.exec != 0;
}

// Names the operations of F the profiling run never executed, which are left
// as written, by their source lines.
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
  diagnose(F, F.getSubprogram(),
           Twine(unexecuted.size()) +
               " floating-point operation(s) the profiling run never executed "
               "are left as written (" +
               where + "); profile an input that runs them to optimize them",
           DS_Warning);
}

void buildSubgraphs(
    const Options &O, Function &F, const ProfileMap &profileMap,
    const SmallVectorImpl<AbstractMatmul> &abstractMatmuls,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    SmallVectorImpl<Subgraph> &subgraphs) {
  // An operation the run never executed is left as written, as an unexecuted
  // matrix product is: the profile gives it no range, which would leave its
  // whole subgraph unpriced. What an executed operation reads was computed on
  // the way to it, so an unexecuted one only ever ends a subgraph as the user
  // of its outputs.
  auto inSubgraph = [&](const Value &V) {
    return isOptimizable(V) && executed(cast<Instruction>(V), profileMap);
  };
  SmallVector<const Instruction *, 8> unexecuted;
  SmallPtrSet<Instruction *, 32> reductionFMAs;
  if (O.Candidates.ReductionSubgraphs)
    for (const auto &am : abstractMatmuls)
      if (am.origin == AbstractMatmul::Origin::ScalarLoopReduction &&
          am.scalarLoop.fma)
        reductionFMAs.insert(am.scalarLoop.fma);

  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] Starting Floodfill for " << F.getName() << '\n';

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isOptimizable(I)) {
        valueToNodeMap[&I] = std::make_shared<FPLLValue>(&I, nullptr, nullptr);
        if (O.Diagnostics.Print)
          llvm::errs() << "[poseidon] not an optimizable operation: " << I
                       << '\n';
        continue;
      }

      valueToNodeMap[&I] = std::make_shared<FPLLValue>(
          &I, fpcoreOperator(getHerbieOperator(I)), nodeFormat(I.getType()));
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
            valueToNodeMap[operand] = std::make_shared<FPLLValue>(
                Arg, nullptr, nodeFormat(Arg->getType()));
          } else if (auto C = dyn_cast<ConstantFP>(operand)) {
            SmallString<10> value;
            C->getValueAPF().toString(value);
            valueToNodeMap[operand] = std::make_shared<FPConst>(
                value.c_str(), nodeFormat(C->getType()));
          } else if (auto CI = dyn_cast<ConstantInt>(operand)) {
            // e.g., powi intrinsic has a constant int as its exponent
            double exponent = static_cast<double>(CI->getSExtValue());
            std::string doubleStr = std::to_string(exponent);
            valueToNodeMap[operand] = std::make_shared<FPConst>(
                doubleStr.c_str(), &formatOf(PrecisionChangeType::FP64));
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
      SetVector<Value *> input_seen;
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

        for (const auto &operand : fpOperands(*I2)) {
          if (!inSubgraph(*operand)) {
            if (!isa<ConstantFP>(operand))
              input_seen.insert(operand);
          } else {
            todo.push_back(operand);
          }
        }

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
        if (O.Diagnostics.Print)
          llvm::errs() << "[poseidon] Found a subgraph with "
                       << operation_seen.size() << " operations and "
                       << input_seen.size() << " inputs and "
                       << output_seen.size() << " outputs\n";

        if (operation_seen.size() == 1 &&
            !reductionFMAs.contains(*operation_seen.begin())) {
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

void attachProfile(
    const Options &O, const ProfileMap &profileMap,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    ArrayRef<Subgraph> subgraphs) {
  for (auto &subgraph : subgraphs) {
    for (auto op : subgraph.operations) {
      size_t idx;
      if (!tryReadProfIdxMetadata(op, idx))
        continue;
      auto it = profileMap.find(idx);
      assert(it != profileMap.end() &&
             "an operation without a profile record is in no subgraph");
      const auto &profileInfo = it->second;

      auto node = valueToNodeMap[op];
      node->sens = profileInfo.sumSens;
      node->grad = profileInfo.sumGrad;
      node->executions = profileInfo.exec;
      node->updateBounds(profileInfo.minRes, profileInfo.maxRes);

      for (const auto &[i, operand] : enumerate(fpOperands(*op))) {
        if (i >= profileInfo.minOperands.size()) {
          if (isa<Constant>(operand))
            continue;
          std::string name;
          raw_string_ostream(name) << *operand;
          reportUserError(*op, "the profile gives this operation " +
                                   Twine(profileInfo.minOperands.size()) +
                                   " operand range(s), and its operand " +
                                   Twine(i) + " (" + StringRef(name).trim() +
                                   ") has none: the profile was not "
                                   "generated from this program. Re-run "
                                   "profile generation.");
        }
        valueToNodeMap[operand]->updateBounds(profileInfo.minOperands[i],
                                              profileInfo.maxOperands[i]);
      }
    }
  }
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

static bool isExpansionBottleneck(const Options &O, Instruction *I,
                                  const Subgraph &subgraph) {
  if (subgraph.outputs.contains(I) && subgraph.outputs.size() <= 1) {
    return false;
  }
  const bool forced = isa<CallInst>(I) && is_contained(O.Candidates.SplitCalls,
                                                       getHerbieOperator(*I));

  unsigned internalUses = 0;
  for (auto *U : I->users()) {
    if (auto *UI = dyn_cast<Instruction>(U)) {
      if (subgraph.operations.contains(UI)) {
        ++internalUses;
      }
    }
  }

  if (!forced && internalUses < O.Candidates.MinUsesSplit) {
    return false;
  }

  SetVector<Instruction *> movedOps = movedOperations(subgraph, I);

  bool isBottleneck = forced || movedOps.size() >= O.Candidates.MinOpsSplit;
  if (O.Diagnostics.Print && isBottleneck) {
    llvm::errs() << "[poseidon] Bottleneck: " << *I << '\n';
    llvm::errs() << "  Num of operations that would be moved: "
                 << movedOps.size() << " (>=" << O.Candidates.MinOpsSplit
                 << ")\n";
    llvm::errs() << "  Num of internal uses: " << internalUses
                 << " (>=" << O.Candidates.MinUsesSplit << ")\n";
    llvm::errs() << "  Operations that would be moved:\n";
    for (auto *op : movedOps) {
      llvm::errs() << '\t' << *op << '\n';
    }
  }
  return isBottleneck;
}

static SetVector<Value *>
findReachedInputs(const SetVector<Instruction *> &operations) {
  SetVector<Value *> reachedInputs;

  for (auto *I : operations) {
    for (auto &op : fpOperands(*I)) {
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

static void splitSubgraphAtBottleneck(const Subgraph &currentSubgraph,
                                      Instruction *bottleneck,
                                      Subgraph &newSubgraph,
                                      Subgraph &remainingSubgraph) {
  SetVector<Instruction *> movedOps =
      movedOperations(currentSubgraph, bottleneck);

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
  assert(remainingSubgraph.inputs.contains(bottleneck) &&
         "the split-off bottleneck is an input of the remaining subgraph");
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
    if (Target::of(*GV->getParent()).isShared(*GV))
      return Obj;
  return nullptr;
}

// The objects `objectOf` gives for the pointers a subgraph reads through the
// loads feeding it and writes through the stores of its outputs.
static SmallPtrSet<const Value *, 4>
touchedObjects(const Subgraph &subgraph,
               function_ref<const Value *(const Value *)> objectOf) {
  SmallPtrSet<const Value *, 4> touched;
  auto add = [&](const Value *Ptr) {
    if (const Value *O = objectOf(Ptr))
      touched.insert(O);
  };
  for (Value *In : subgraph.inputs)
    if (auto *LI = dyn_cast<LoadInst>(In))
      add(LI->getPointerOperand());
  for (Instruction *Op : subgraph.operations)
    for (Value *V : Op->operands())
      if (auto *LI = dyn_cast<LoadInst>(V))
        add(LI->getPointerOperand());
  for (Instruction *Out : subgraph.outputs)
    for (User *U : Out->users())
      if (auto *SI = dyn_cast<StoreInst>(U))
        if (SI->getValueOperand() == Out)
          add(SI->getPointerOperand());
  return touched;
}

// Merges every group of subgraphs that touch a common staging buffer (a
// pointer argument or an addrspace(3) global reached by a load feeding the
// subgraph or a store consuming it) into one unit, so the per-unit tier is the
// buffer's.
static void mergeSharedStagingSubgraphs(const Options &O,
                                        SmallVectorImpl<Subgraph> &subgraphs,
                                        Function &F) {
  if (subgraphs.size() < 2)
    return;

  SmallVector<SmallPtrSet<const Value *, 4>, 8> touched;
  for (const Subgraph &subgraph : subgraphs)
    touched.push_back(touchedObjects(subgraph, stagingObjectOf));

  SmallVector<unsigned, 8> parent(subgraphs.size());
  for (unsigned i = 0; i < parent.size(); ++i)
    parent[i] = i;
  auto find = [&](unsigned x) -> unsigned {
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
    if (O.Diagnostics.Print) {
      SmallPtrSet<const Value *, 4> objs;
      for (unsigned idx : kv.second)
        for (const Value *O : touched[idx])
          objs.insert(O);
      llvm::errs() << "[poseidon] merged " << kv.second.size()
                   << " FP subgraphs sharing " << objs.size()
                   << " staging buffer(s) into one optimization unit ("
                   << M.operations.size() << " operations, " << M.inputs.size()
                   << " inputs, " << M.outputs.size() << " outputs) in "
                   << F.getName() << '\n';
    }
    merged.push_back(std::move(M));
  }

  subgraphs.clear();
  for (auto &M : merged)
    subgraphs.push_back(std::move(M));
}

void hintSharedStaging(const Session &S, Function &F,
                       ArrayRef<Subgraph> subgraphs) {
  // The addrspace(3) global a pointer reaches: directly, or through the
  // parameter of F the site binds to it.
  auto sharedBufferOf = [&](const Value *Ptr) -> const Value * {
    const Value *Obj = getUnderlyingObject(Ptr);
    if (auto *A = dyn_cast_or_null<Argument>(Obj))
      if (Value *SA = siteArg(S.Sites, &F, A->getArgNo()))
        Obj = getUnderlyingObject(SA->stripPointerCasts());
    auto *GV = dyn_cast_or_null<GlobalVariable>(Obj);
    return GV && Target::of(*GV->getParent()).isShared(*GV) ? GV : nullptr;
  };
  MapVector<const Value *, unsigned> units;
  for (const Subgraph &subgraph : subgraphs)
    for (const Value *G : touchedObjects(subgraph, sharedBufferOf))
      ++units[G];
  std::string shared;
  raw_string_ostream os(shared);
  ListSeparator sep;
  for (const auto &[G, n] : units)
    if (n > 1)
      os << sep << '@' << G->getName() << " (" << n << " units)";
  if (shared.empty())
    return;
  diagnose(F, F.getSubprogram(),
           "more than one optimization unit reads or writes the shared "
           "buffer(s) " +
               shared +
               ", which a precision change narrows only when all its units "
               "pick one tier; -poseidon-merge-shared-staging makes them one "
               "unit",
           DS_Warning);
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

      if (isExpansionBottleneck(O, I, currentSubgraph)) {
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

  // After splitting, so that a merged unit is never split again.
  if (O.Candidates.MergeSharedStaging)
    mergeSharedStagingSubgraphs(O, subgraphs, F);
}

} // namespace poseidon

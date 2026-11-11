//===- Subgraphs.cpp - the FP subgraphs of a site -------------------------===//
#include "Subgraphs.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

#include "Core/Options.h"
#include "Model/FPCore.h"
#include "Sites/Canonicalize.h"
#include "Targets/Target.h"
#include "Techniques/Matmul/Matmul.h"
#include "Utils.h"

using namespace llvm;

namespace poseidon {

// An op without a profile record (-poseidon-loose-coverage) counts as zero.
static void aggressiveDCE(
    const Options &O, Function &F, SmallVectorImpl<Subgraph> &subgraphs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &nodes,
    const ProfileMap &profileMap, const std::unordered_set<size_t> &zeroGradIdx,
    const SmallVectorImpl<AbstractMatmul> &abstractMatmuls) {
  // In the order marked, which -poseidon-print lists them in.
  SmallSetVector<Value *, 32> critical;
  SmallVector<Value *, 16> worklist;
  auto mark = [&](Value *V) {
    if (V->getType()->isFloatingPointTy() && critical.insert(V))
      worklist.push_back(V);
  };
  for (Instruction &I : instructions(F))
    if (auto *fcmp = dyn_cast<FCmpInst>(&I)) {
      mark(fcmp->getOperand(0));
      mark(fcmp->getOperand(1));
    }
  for (const auto &am : abstractMatmuls) {
    if (am.scalarLoop.fma)
      critical.insert(am.scalarLoop.fma);
    if (am.scalarLoop.fmul)
      critical.insert(am.scalarLoop.fmul);
  }
  while (!worklist.empty()) {
    auto *inst = dyn_cast<Instruction>(worklist.pop_back_val());
    if (!inst)
      continue;
    auto operands =
        isa<CallInst>(inst) ? cast<CallInst>(inst)->args() : inst->operands();
    for (auto &op : operands)
      mark(op);
    if (auto *load = dyn_cast<LoadInst>(inst))
      for (User *U : load->getPointerOperand()->users())
        if (auto *store = dyn_cast<StoreInst>(U))
          mark(store->getValueOperand());
  }

  if (O.Diagnostics.Print) {
    llvm::errs() << "Critical values:\n";
    for (Value *V : critical)
      llvm::errs() << "\t" << *V << "\n";
  }

  auto zeroGrad = [&](Instruction *op) {
    size_t idx;
    if (!tryReadProfIdxMetadata(op, idx))
      return false;
    return zeroGradIdx.count(idx) || !profileMap.count(idx);
  };

  for (auto it = subgraphs.begin(); it != subgraphs.end();) {
    SmallVector<Instruction *, 32> toRemove;
    for (Instruction *op : it->operations)
      if (!critical.count(op) && zeroGrad(op))
        toRemove.push_back(op);
    for (Instruction *op : toRemove) {
      if (O.Diagnostics.Print)
        llvm::errs() << "Aggressive DCE: eliminating zero-gradient "
                     << "non-critical instruction: " << *op << "\n";
      op->replaceAllUsesWith(UndefValue::get(op->getType()));
      nodes.erase(op);
      it->operations.remove(op);
      it->outputs.remove(op);
      op->eraseFromParent();
    }
    if (it->outputs.empty()) {
      if (O.Diagnostics.Print)
        llvm::errs() << "Removing empty subgraph\n";
      it = subgraphs.erase(it);
    } else {
      ++it;
    }
  }

  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] After aggressive DCE, have " << subgraphs.size()
                 << " subgraphs in " << F.getName() << "\n";
}

void buildSubgraphs(
    const Options &O, Function &F,
    const SmallVectorImpl<AbstractMatmul> &abstractMatmuls,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    SmallVectorImpl<Subgraph> &subgraphs) {
  SmallPtrSet<Instruction *, 32> reductionFMAs;
  if (O.Candidates.ReductionSubgraphs)
    for (const auto &am : abstractMatmuls)
      if (am.origin == AbstractMatmul::Origin::ScalarLoopReduction &&
          am.scalarLoop.fma)
        reductionFMAs.insert(am.scalarLoop.fma);

  llvm::errs() << "[poseidon] Starting Floodfill for " << F.getName() << "\n";

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isOptimizable(I)) {
        valueToNodeMap[&I] = std::make_shared<FPLLValue>(&I, "__nh", "__nh");
        if (O.Diagnostics.Print)
          llvm::errs()
              << "Registered FPLLValue for non-isOptimizable instruction: " << I
              << "\n";
        continue;
      }

      std::string dtype;
      if (I.getType()->isFloatTy()) {
        dtype = "f32";
      } else if (I.getType()->isDoubleTy()) {
        dtype = "f64";
      } else {
        llvm_unreachable("Unexpected floating point type for instruction");
      }
      valueToNodeMap[&I] =
          std::make_shared<FPLLValue>(&I, getHerbieOperator(I), dtype);
    }
  }

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isOptimizable(I))
        continue;
      auto node = valueToNodeMap[&I];
      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I).args() : I.operands();
      for (auto &operand : operands) {
        if (!valueToNodeMap.count(operand)) {
          if (auto Arg = dyn_cast<Argument>(operand)) {
            std::string dtype;
            if (Arg->getType()->isFloatTy()) {
              dtype = "f32";
            } else if (Arg->getType()->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable("Unexpected floating point type for argument");
            }
            valueToNodeMap[operand] =
                std::make_shared<FPLLValue>(Arg, "__arg", dtype);
            if (O.Diagnostics.Print)
              llvm::errs() << "Registered FPNode for argument: " << *Arg
                           << "\n";
          } else if (auto C = dyn_cast<ConstantFP>(operand)) {
            SmallString<10> value;
            C->getValueAPF().toString(value);
            std::string dtype;
            if (C->getType()->isFloatTy()) {
              dtype = "f32";
            } else if (C->getType()->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable("Unexpected floating point type for constant");
            }
            valueToNodeMap[operand] =
                std::make_shared<FPConst>(value.c_str(), dtype);
            if (O.Diagnostics.Print)
              llvm::errs() << "Registered FPNode for " << dtype
                           << " constant: " << value << "\n";
          } else if (auto CI = dyn_cast<ConstantInt>(operand)) {
            // e.g., powi intrinsic has a constant int as its exponent
            double exponent = static_cast<double>(CI->getSExtValue());
            std::string dtype = "f64";
            std::string doubleStr = std::to_string(exponent);
            valueToNodeMap[operand] =
                std::make_shared<FPConst>(doubleStr.c_str(), dtype);
            if (O.Diagnostics.Print)
              llvm::errs() << "Registered FPNode for " << dtype
                           << " constant (casted from integer): " << doubleStr
                           << "\n";
          } else if (auto GV = dyn_cast<GlobalVariable>(operand)) {
            Type *elemType = GV->getValueType();

            assert(elemType->isFloatingPointTy() &&
                   "Global variable is not floating point type");
            std::string dtype;
            if (elemType->isFloatTy()) {
              dtype = "f32";
            } else if (elemType->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable(
                  "Unexpected floating point type for global variable");
            }
            valueToNodeMap[operand] =
                std::make_shared<FPLLValue>(GV, "__gv", dtype);
            if (O.Diagnostics.Print)
              llvm::errs() << "Registered FPNode for global variable: " << *GV
                           << "\n";
          } else {
            assert(0 && "Unknown operand");
          }
        }
        node->addOperand(valueToNodeMap[operand]);
      }
    }
  }

  SmallSet<Value *, 8> processed;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isOptimizable(I)) {
        if (O.Diagnostics.Print)
          llvm::errs() << "Skipping non-isOptimizable instruction: " << I
                       << "\n";
        continue;
      }

      if (processed.contains(&I)) {
        if (O.Diagnostics.Print)
          llvm::errs() << "Skipping already seen instruction: " << I << "\n";
        continue;
      }

      if (O.Diagnostics.Print)
        llvm::errs() << "Starting floodfill from: " << I << "\n";

      SmallVector<Value *, 8> todo;
      SetVector<Value *> input_seen;
      SetVector<Instruction *> output_seen;
      SetVector<Instruction *> operation_seen;
      todo.push_back(&I);
      while (!todo.empty()) {
        auto cur = todo.pop_back_val();
        assert(valueToNodeMap.count(cur) && "Node not found in valueToNodeMap");

        assert(isa<Instruction>(cur));
        auto I2 = cast<Instruction>(cur);

        if (operation_seen.contains(I2)) {
          if (O.Diagnostics.Print)
            llvm::errs() << "Skipping already seen instruction: " << *I2
                         << "\n";
          continue;
        }

        assert(!processed.contains(cur));

        if (O.Diagnostics.Print)
          llvm::errs() << "Insert to operation_seen and processed: " << *I2
                       << "\n";
        operation_seen.insert(I2);
        processed.insert(cur);

        auto operands =
            isa<CallInst>(I2) ? cast<CallInst>(I2)->args() : I2->operands();

        for (const auto &operand : operands) {
          if (!isOptimizable(*operand)) {
            if (O.Diagnostics.Print)
              llvm::errs() << "Non-isOptimizable input found: " << *operand
                           << "\n";

            if (!isa<ConstantFP>(operand))
              input_seen.insert(operand);
          } else {
            if (O.Diagnostics.Print)
              llvm::errs() << "Adding operand to todo list: " << *operand
                           << "\n";
            todo.push_back(operand);
          }
        }

        for (auto U : I2->users()) {
          if (auto I3 = dyn_cast<Instruction>(U)) {
            if (!isOptimizable(*I3)) {
              if (O.Diagnostics.Print)
                llvm::errs() << "Output instruction found: " << *I2 << "\n";
              output_seen.insert(I2);
            } else {
              if (O.Diagnostics.Print)
                llvm::errs() << "Adding user to todo list: " << *I3 << "\n";
              todo.push_back(I3);
            }
          }
        }
      }

      if (!operation_seen.empty()) {
        if (O.Diagnostics.Print) {
          llvm::errs() << "Found a subgraph with " << operation_seen.size()
                       << " operations and " << input_seen.size()
                       << " inputs and " << output_seen.size() << " outputs\n";

          llvm::errs() << "Inputs:\n";

          for (auto &input : input_seen) {
            llvm::errs() << *input << "\n";
          }

          llvm::errs() << "Outputs:\n";
          for (auto &output : output_seen) {
            llvm::errs() << *output << "\n";
          }

          llvm::errs() << "Operations:\n";
          for (auto &operation : operation_seen) {
            llvm::errs() << *operation << "\n";
          }
        }

        if (operation_seen.size() == 1 &&
            !reductionFMAs.contains(*operation_seen.begin())) {
          if (O.Diagnostics.Print)
            llvm::errs() << "Skipping trivial subgraph\n";
          continue;
        }
        if (operation_seen.size() == 1 && O.Diagnostics.Print)
          llvm::errs() << "Keeping 1-op reduction subgraph: "
                       << **operation_seen.begin() << "\n";

        subgraphs.emplace_back(input_seen, output_seen, operation_seen);
      }
    }
  }

  if (O.Diagnostics.Print) {
    llvm::errs() << "[poseidon] Found " << subgraphs.size()
                 << " initial subgraphs in " << F.getName() << "\n";
  }
}

void attachProfile(
    const Options &O, const ProfileMap &profileMap,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    ArrayRef<Subgraph> subgraphs) {
  for (auto &subgraph : subgraphs) {
    for (auto op : subgraph.operations) {
      if (auto MD = op->getMetadata("poseidon.prof.idx")) {
        if (auto C = dyn_cast<ConstantAsMetadata>(MD->getOperand(0))) {
          size_t idx = cast<ConstantInt>(C->getValue())->getZExtValue();
          auto it = profileMap.find(idx);

          if (it != profileMap.end()) {
            const auto &profileInfo = it->second;

            auto node = valueToNodeMap[op];
            node->sens = profileInfo.sumSens;
            node->grad = profileInfo.sumGrad;
            node->executions = profileInfo.exec;
            node->updateBounds(profileInfo.minRes, profileInfo.maxRes,
                               O.Diagnostics.Print);

            if (O.Diagnostics.Print) {
              llvm::errs() << "Range of " << *op << " is ["
                           << node->getLowerBound() << ", "
                           << node->getUpperBound() << "]\n";
              llvm::errs() << "Sensitivity score of " << *op
                           << " is: " << node->sens << "\n"
                           << "Gradient sum of " << *op << " is: " << node->grad
                           << "\n"
                           << "Execution count of " << *op
                           << " is: " << node->executions << "\n";
            }

            auto operands =
                isa<CallInst>(op) ? cast<CallInst>(op)->args() : op->operands();

            for (const auto &operand_ : enumerate(operands)) {
              auto &operand = operand_.value();
              auto i = operand_.index();

              if (i < profileInfo.minOperands.size()) {
                auto operandNode = valueToNodeMap[operand];
                operandNode->updateBounds(profileInfo.minOperands[i],
                                          profileInfo.maxOperands[i],
                                          O.Diagnostics.Print);
                if (O.Diagnostics.Print) {
                  llvm::errs() << "Range of " << *operand << " is ["
                               << operandNode->getLowerBound() << ", "
                               << operandNode->getUpperBound() << "]\n";
                }
              }
            }
          } else {
            if (!O.Profiling.LooseCoverage) {
              llvm::errs() << "FP Instruction " << *op
                           << " has no execution logged (idx=" << idx << ")!\n";
              llvm_unreachable("Unexecuted instruction found; set "
                               "-poseidon-loose-coverage "
                               "to suppress this error\n");
            }
            auto node = valueToNodeMap[op];
            node->sens = 0;
            node->grad = 0;
            node->executions = 0;
            if (O.Diagnostics.Print)
              llvm::errs() << "Sensitivity/gradient/executions of " << *op
                           << " not found in the log; using 0 for all\n";
          }
        }
      }
    }
  }
}

void shapeSubgraphs(
    const Options &O, Function &F, const ProfileMap &profileMap,
    const std::unordered_set<size_t> &zeroGradIdx,
    const SmallVectorImpl<AbstractMatmul> &abstractMatmuls,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    SmallVectorImpl<Subgraph> &subgraphs) {
  if (O.Solver.AggressiveDCE)
    aggressiveDCE(O, F, subgraphs, valueToNodeMap, profileMap, zeroGradIdx,
                  abstractMatmuls);

  splitSubgraphs(O, subgraphs);

  if (O.Diagnostics.Print) {
    llvm::errs() << "[poseidon] After splitting, have " << subgraphs.size()
                 << " subgraphs in " << F.getName() << "\n";
  }

  // After splitting so a merged unit is never re-split, and before the
  // frequency map below is built from the merged unit's instructions.
  if (O.Candidates.MergeSharedStaging) {
    mergeSharedStagingSubgraphs(O, subgraphs, F);
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] After staging-tier merge, have "
                   << subgraphs.size() << " subgraphs in " << F.getName()
                   << "\n";
  }

  if (O.Diagnostics.Print) {
    llvm::errs() << "\n=== Function IR after Subgraph Splitting ===\n";

    std::unordered_map<Instruction *, int> instToSubgraphIdx;
    for (size_t idx = 0; idx < subgraphs.size(); ++idx) {
      for (auto *inst : subgraphs[idx].operations) {
        instToSubgraphIdx[inst] = idx;
      }
      for (auto *inst : subgraphs[idx].outputs) {
        if (instToSubgraphIdx.find(inst) == instToSubgraphIdx.end()) {
          instToSubgraphIdx[inst] = idx;
        }
      }
    }

    for (auto &BB : F) {
      BB.printAsOperand(llvm::errs(), false);
      llvm::errs() << ":\n";
      for (auto &I : BB) {
        llvm::errs() << "  ";
        I.print(llvm::errs());

        auto it = instToSubgraphIdx.find(&I);
        if (it != instToSubgraphIdx.end()) {
          llvm::errs() << " ; [SG" << it->second << "]";
        }
        llvm::errs() << "\n";
      }
    }
    llvm::errs() << "=== End of Function IR ===\n\n";
  }
}

bool isExpansionBottleneck(const Options &O, Instruction *I,
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

  bool isBottleneck = forced || movedOps.size() >= O.Candidates.MinOpsSplit;
  if (O.Diagnostics.Print && isBottleneck) {
    llvm::errs() << "Bottleneck: " << *I << "\n";
    llvm::errs() << "Num of operations that would be moved: " << movedOps.size()
                 << " (>=" << O.Candidates.MinOpsSplit << ")\n";
    llvm::errs() << "Num of internal uses: " << internalUses
                 << " (>=" << O.Candidates.MinUsesSplit << ")\n";
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
    if (Target::of(*GV->getParent()).isShared(*GV))
      return Obj;
  return nullptr;
}

void mergeSharedStagingSubgraphs(const Options &O,
                                 SmallVectorImpl<Subgraph> &subgraphs,
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
                   << F.getName() << "\n";
    }
    merged.push_back(std::move(M));
  }

  subgraphs.clear();
  for (auto &M : merged)
    subgraphs.push_back(std::move(M));
}

void splitSubgraphs(const Options &O, SmallVectorImpl<Subgraph> &subgraphs) {

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

      if (isExpansionBottleneck(O, I, currentSubgraph)) {
        if (O.Diagnostics.Print) {
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

        if (O.Diagnostics.Print) {
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

  if (O.Diagnostics.Print) {
    llvm::errs() << "=== Subgraph splitting complete ===\n";
    llvm::errs() << "  Original subgraphs: " << subgraphs.size() << "\n";
    llvm::errs() << "  Final subgraphs after splitting: "
                 << resultSubgraphs.size() << "\n";
  }

  subgraphs = std::move(resultSubgraphs);
}

} // namespace poseidon

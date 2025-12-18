//===- Utils.cpp - Utilities for operation interfaces
//--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Interfaces/Utils.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <optional>

using namespace mlir;
using namespace mlir::enzyme;
namespace mlir {
namespace enzyme {
namespace oputils {

const std::set<std::string> &getNonCapturingFunctions() {
  static std::set<std::string> NonCapturingFunctions = {
      "free",           "printf",       "fprintf",       "scanf",
      "fscanf",         "gettimeofday", "clock_gettime", "getenv",
      "strrchr",        "strlen",       "sprintf",       "sscanf",
      "mkdir",          "fwrite",       "fread",         "memcpy",
      "cudaMemcpy",     "memset",       "cudaMemset",    "__isoc99_scanf",
      "__isoc99_fscanf"};
  return NonCapturingFunctions;
}

static bool isCaptured(Value v, Operation *potentialUser = nullptr,
                       bool *seenuse = nullptr) {
  SmallVector<Value> todo = {v};
  while (todo.size()) {
    Value v = todo.pop_back_val();
    for (auto u : v.getUsers()) {
      if (seenuse && u == potentialUser)
        *seenuse = true;
      if (isa<memref::LoadOp, LLVM::LoadOp, affine::AffineLoadOp>(u))
        continue;
      if (auto s = dyn_cast<memref::StoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto s = dyn_cast<affine::AffineStoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto s = dyn_cast<LLVM::StoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto sub = dyn_cast<LLVM::GEPOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<LLVM::BitcastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<LLVM::AddrSpaceCastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<func::ReturnOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemsetOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemcpyOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemmoveOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<memref::CastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<memref::DeallocOp>(u)) {
        continue;
      }
      if (auto cop = dyn_cast<LLVM::CallOp>(u)) {
        if (auto callee = cop.getCallee()) {
          if (getNonCapturingFunctions().count(callee->str()))
            continue;
        }
      }
      if (auto cop = dyn_cast<func::CallOp>(u)) {
        if (getNonCapturingFunctions().count(cop.getCallee().str()))
          continue;
      }
      return true;
    }
  }

  return false;
}

static Value getBase(Value v) {
  while (true) {
    if (auto s = v.getDefiningOp<LLVM::GEPOp>()) {
      v = s.getBase();
      continue;
    }
    if (auto s = v.getDefiningOp<LLVM::BitcastOp>()) {
      v = s.getArg();
      continue;
    }
    if (auto s = v.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
      v = s.getArg();
      continue;
    }
    if (auto s = v.getDefiningOp<memref::CastOp>()) {
      v = s.getSource();
      continue;
    }
    break;
  }
  return v;
}

static bool isStackAlloca(Value v) {
  return v.getDefiningOp<memref::AllocaOp>() ||
         v.getDefiningOp<memref::AllocOp>() ||
         v.getDefiningOp<LLVM::AllocaOp>();
}

bool mayAlias(Value v1, Value v2) {
  v1 = getBase(v1);
  v2 = getBase(v2);
  if (v1 == v2)
    return true;

  // We may now assume neither v1 nor v2 are subindices

  if (auto glob = v1.getDefiningOp<memref::GetGlobalOp>()) {
    if (auto Aglob = v2.getDefiningOp<memref::GetGlobalOp>()) {
      return glob.getName() == Aglob.getName();
    }
  }

  if (auto glob = v1.getDefiningOp<LLVM::AddressOfOp>()) {
    if (auto Aglob = v2.getDefiningOp<LLVM::AddressOfOp>()) {
      return glob.getGlobalName() == Aglob.getGlobalName();
    }
  }

  bool isAlloca[2];
  bool isGlobal[2];

  isAlloca[0] = isStackAlloca(v1);
  isGlobal[0] = v1.getDefiningOp<memref::GetGlobalOp>() ||
                v1.getDefiningOp<LLVM::AddressOfOp>();

  isAlloca[1] = isStackAlloca(v2);

  isGlobal[1] = v2.getDefiningOp<memref::GetGlobalOp>() ||
                v2.getDefiningOp<LLVM::AddressOfOp>();

  // Non-equivalent allocas/global's cannot conflict with each other
  if ((isAlloca[0] || isGlobal[0]) && (isAlloca[1] || isGlobal[1]))
    return false;

  BlockArgument barg1 = dyn_cast<BlockArgument>(v1);
  BlockArgument barg2 = dyn_cast<BlockArgument>(v2);

  FunctionOpInterface f1 =
      barg1 ? dyn_cast<FunctionOpInterface>(barg1.getOwner()->getParentOp())
            : nullptr;
  FunctionOpInterface f2 =
      barg2 ? dyn_cast<FunctionOpInterface>(barg2.getOwner()->getParentOp())
            : nullptr;

  bool isNoAlias1 =
      f1 ? !!f1.getArgAttr(barg1.getArgNumber(),
                           LLVM::LLVMDialect::getNoAliasAttrName())
         : false;
  bool isNoAlias2 =
      f2 ? !!f2.getArgAttr(barg2.getArgNumber(),
                           LLVM::LLVMDialect::getNoAliasAttrName())
         : false;

  if (!isCaptured(v1) && isNoAlias1)
    return false;
  if (!isCaptured(v2) && isNoAlias2)
    return false;

  bool isArg[2];
  isArg[0] = f1;
  isArg[1] = f2;

  // Stack allocations cannot have been passed as an argument.
  if ((isAlloca[0] && isArg[1]) || (isAlloca[1] && isArg[0]))
    return false;

  // Non captured base allocas cannot conflict with another base value.
  if (isAlloca[0] && !isCaptured(v1))
    return false;

  if (isAlloca[1] && !isCaptured(v2))
    return false;

  return true;
}

bool mayAlias(MemoryEffects::EffectInstance a, Value v2) {
  if (Value v = a.getValue()) {
    return mayAlias(v, v2);
  }
  return true;
}

bool mayAlias(MemoryEffects::EffectInstance &a,
              MemoryEffects::EffectInstance &b) {
  if (a.getResource()->getResourceID() != b.getResource()->getResourceID())
    return false;
  Value valA = a.getValue();
  Value valB = b.getValue();

  // unknown effects may always alias
  if (!valA || !valB) {
    return true;
  }

  auto valResult = oputils::mayAlias(valA, valB);
  return valResult;
}

bool isReadOnly(Operation *op) {
  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only reads from memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    if (!llvm::all_of(effects, [op](const MemoryEffects::EffectInstance &it) {
          return isa<MemoryEffects::Read>(it.getEffect());
        })) {
      return false;
    }
  }

  bool isRecursiveContainer =
      op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (isRecursiveContainer) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (!isReadOnly(&nestedOp))
            return false;
      }
    }
  }

  return true;
}

bool isReadNone(Operation *op) {
  bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (!isReadNone(&nestedOp))
            return false;
      }
    }
    return true;
  }

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    if (llvm::any_of(effects, [](const MemoryEffects::EffectInstance &it) {
          return isa<MemoryEffects::Read>(it.getEffect()) ||
                 isa<MemoryEffects::Write>(it.getEffect());
        })) {
      return false;
    }
    return true;
  }
  return false;
}

bool collectOpEffects(Operation *rootOp,
                      SmallVector<MemoryEffects::EffectInstance> &effects) {
  SmallVector<Operation *> effectingOps(1, rootOp);
  bool couldCollectEffects = true;

  while (!effectingOps.empty()) {
    Operation *op = effectingOps.pop_back_val();
    bool isRecursiveContainer =
        op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();

    if (isRecursiveContainer) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (Operation &nestedOp : block) {
            effectingOps.push_back(&nestedOp);
          }
        }
      }
    }

    if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> localEffects;
      effectInterface.getEffects(localEffects);
      llvm::append_range(effects, localEffects);
    } else if (!isRecursiveContainer) {
      // Handle specific operations which are not recursive containers, but
      // still may have memory effects(eg. autodiff calls, llvm calls to libc
      // functions). If it's none of these, then the operation may not have any
      // memory effects
      if (auto cop = dyn_cast<LLVM::CallOp>(op)) {
        if (auto callee = cop.getCallee()) {
          if (*callee == "scanf" || *callee == "__isoc99_scanf") {
            // Global read
            effects.emplace_back(
                MemoryEffects::Effect::get<MemoryEffects::Read>());

            bool first = true;
            for (auto &arg : cop.getArgOperandsMutable()) {
              if (first)
                effects.emplace_back(MemoryEffects::Read::get(), &arg);
              else
                effects.emplace_back(MemoryEffects::Write::get(), &arg,
                                     SideEffects::DefaultResource::get());
              first = false;
            }
          }
          if (*callee == "fscanf" || *callee == "__isoc99_fscanf") {
            // Global read
            effects.emplace_back(
                MemoryEffects::Effect::get<MemoryEffects::Read>());

            for (auto &&[idx, arg] :
                 llvm::enumerate(cop.getArgOperandsMutable())) {
              if (idx == 0) {
                effects.emplace_back(MemoryEffects::Read::get(), &arg,
                                     SideEffects::DefaultResource::get());
                effects.emplace_back(MemoryEffects::Write::get(), &arg,
                                     SideEffects::DefaultResource::get());
              } else if (idx == 1) {
                effects.emplace_back(MemoryEffects::Read::get(), &arg,
                                     SideEffects::DefaultResource::get());
              } else
                effects.emplace_back(MemoryEffects::Write::get(), &arg,
                                     SideEffects::DefaultResource::get());
            }
          }
          if (*callee == "printf") {
            // Global read
            effects.emplace_back(
                MemoryEffects::Effect::get<MemoryEffects::Write>());
            for (auto &arg : cop.getArgOperandsMutable()) {
              effects.emplace_back(MemoryEffects::Read::get(), &arg,
                                   SideEffects::DefaultResource::get());
            }
          }
          if (*callee == "free") {
            for (auto &arg : cop.getArgOperandsMutable()) {
              effects.emplace_back(MemoryEffects::Free::get(), &arg,
                                   SideEffects::DefaultResource::get());
            }
          }
          if (*callee == "strlen") {
            for (auto &arg : cop.getArgOperandsMutable()) {
              effects.emplace_back(MemoryEffects::Read::get(), &arg,
                                   SideEffects::DefaultResource::get());
            }
          }
        }
      } else {
        // TODO: handle AutoDiffOp, ForwardDiffOp and AutoDiffRegionOp effects.
        // Just conservatively add all effects for now

        // We need to be conservative here in case the op doesn't have the
        // interface and assume it can have any possible effect.

        effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
        effects.emplace_back(
            MemoryEffects::Effect::get<MemoryEffects::Write>());
        effects.emplace_back(
            MemoryEffects::Effect::get<MemoryEffects::Allocate>());
        effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
        couldCollectEffects = false;

        // no use in exploring other ops so break
        break;
      }
    }
  }
  return couldCollectEffects;
}

SmallVector<MemoryEffects::EffectInstance>
collectFnEffects(FunctionOpInterface fnOp) {
  SmallVector<MemoryEffects::EffectInstance> innerEffects;
  for (auto &blk : fnOp.getBlocks()) {
    for (auto &op : blk) {
      SmallVector<MemoryEffects::EffectInstance> opEffects;
      (void)collectOpEffects(&op, opEffects);
      innerEffects.append(opEffects.begin(), opEffects.end());
    }
  }

  return innerEffects;
}

MemoryEffects::EffectInstance getEffectOfVal(Value val,
                                             MemoryEffects::Effect *effect,
                                             SideEffects::Resource *resource) {

  if (auto valOR = dyn_cast<OpResult>(val))
    return MemoryEffects::EffectInstance(effect, valOR, resource);
  else if (auto valBA = dyn_cast<BlockArgument>(val)) {
    return MemoryEffects::EffectInstance(effect, valBA, resource);
  } else {
    llvm_unreachable("Provided Value is neither an argument nor a result of an "
                     "op. This is not allowed by SSA");
    return nullptr;
  }
}

} // namespace oputils
} // namespace enzyme
} // namespace mlir

//=- CostModel.cpp - the measured cost model and cost walks ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CostModel.h"
#include "Flags.h"
#include "Types.h"
#include "Utils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace llvm;

namespace poseidon {

static struct {
  std::unordered_set<std::string> scalar;
  std::unordered_set<std::string> matrix;
} SupportedTypes;

static std::string CostModelNativeArch;

// The CSV this compilation prices from, resolved once: the flag, the
// environment, this user's cache, then what the installation ships. Everything
// below reads it through costModelPath(), never the flag.
static std::string ResolvedCostModel;
static bool CostModelResolved = false;

static bool resolveFromRequest() {
  if (CostModelResolved)
    return true;
  if (!flags::CostModel.empty()) {
    ResolvedCostModel = flags::CostModel;
    CostModelResolved = true;
  } else if (const char *Env = std::getenv("POSEIDON_COST_MODEL")) {
    if (*Env) {
      ResolvedCostModel = Env;
      CostModelResolved = true;
    }
  }
  return CostModelResolved;
}

const std::string &costModelPath() {
  resolveFromRequest();
  return ResolvedCostModel;
}

static std::string userCacheDir() {
  if (const char *X = std::getenv("XDG_CACHE_HOME"))
    if (*X)
      return std::string(X) + "/poseidon";
  if (const char *H = std::getenv("HOME"))
    if (*H)
      return std::string(H) + "/.cache/poseidon";
  return std::string();
}

// One measured model per device: the file name carries the architecture it was
// measured on, so the module's target-cpu names it.
static void resolveForTarget(StringRef TargetCpu, const Function &F) {
  if (resolveFromRequest())
    return;
  if (TargetCpu.empty())
    report_fatal_error("Poseidon: optimizing " + Twine(F.getName()) +
                       " needs a measured cost model, and the module names no "
                       "target-cpu to look one up by; pass "
                       "-poseidon-cost-model=<csv>");
  std::string Prefix = ("cm_" + TargetCpu + "_").str();
  SmallVector<std::string, 4> Dirs;
  if (std::string C = userCacheDir(); !C.empty())
    Dirs.push_back(C);
  Dirs.push_back(POSEIDON_INSTALL_COST_MODELS);
  SmallVector<std::string, 4> Found;
  for (const std::string &Dir : Dirs) {
    std::error_code EC;
    for (sys::fs::directory_iterator It(Dir, EC), E; It != E && !EC;
         It.increment(EC)) {
      StringRef Name = sys::path::filename(It->path());
      if (Name.starts_with(Prefix) && Name.ends_with(".csv"))
        Found.push_back(It->path());
    }
    if (!Found.empty())
      break;
  }
  std::string Where;
  for (const std::string &Dir : Dirs)
    Where += (Where.empty() ? "" : ", ") + Dir;
  if (Found.empty())
    report_fatal_error("Poseidon: no cost model for target-cpu '" +
                       Twine(TargetCpu) + "' in " + Where +
                       "; measure this device with poseidon-calibrate, or pass "
                       "-poseidon-cost-model=<csv>");
  if (Found.size() > 1) {
    std::string List;
    for (const std::string &P : Found)
      List += (List.empty() ? "" : ", ") + P;
    report_fatal_error("Poseidon: several cost models for target-cpu '" +
                       Twine(TargetCpu) + "': " + List +
                       "; name the one to price from with "
                       "-poseidon-cost-model=<csv>");
  }
  ResolvedCostModel = Found.front();
  CostModelResolved = true;
}

static const std::unordered_set<std::string> DefaultScalarTypes = {
    "half", "bf16", "float", "double"};

const std::unordered_set<std::string> &getScalarTypes() {
  if (!costModelPath().empty()) {
    getCostModel();
    if (!SupportedTypes.scalar.empty())
      return SupportedTypes.scalar;
  }
  return DefaultScalarTypes;
}

const std::map<std::pair<std::string, std::string>, double> &getCostModel() {
  static std::map<std::pair<std::string, std::string>, double> CostModel;
  static bool Loaded = false;
  if (!Loaded) {
    std::ifstream CostFile(costModelPath());
    if (!CostFile.is_open()) {
      std::string msg =
          "Cost model file could not be opened: " + costModelPath();
      llvm_unreachable(msg.c_str());
    }
    std::string Line;
    while (std::getline(CostFile, Line)) {
      if (Line.empty())
        continue;
      if (Line[0] == '#') {
        // Parse "# scalar_types=...", "# matrix_types=...", "# native_arch=..."
        auto content = StringRef(Line).drop_front(1).trim();
        StringRef key, vals;
        std::tie(key, vals) = content.split('=');
        key = key.trim();
        vals = vals.trim();
        if (key == "native_arch") {
          CostModelNativeArch = vals.str();
          continue;
        }
        std::unordered_set<std::string> *target = nullptr;
        if (key == "scalar_types")
          target = &SupportedTypes.scalar;
        else if (key == "matrix_types")
          target = &SupportedTypes.matrix;
        if (target && !vals.empty()) {
          SmallVector<StringRef, 8> tokens;
          vals.split(tokens, ',', -1, false);
          for (auto t : tokens)
            target->insert(t.str());
        }
        continue;
      }
      std::istringstream SS(Line);
      std::string OpcodeStr, PrecisionStr, CostStr;
      if (!std::getline(SS, OpcodeStr, ',')) {
        llvm_unreachable(
            ("Unexpected line in custom cost model: " + Line).c_str());
      }
      if (!std::getline(SS, PrecisionStr, ',')) {
        llvm_unreachable(
            ("Unexpected line in custom cost model: " + Line).c_str());
      }
      if (!std::getline(SS, CostStr)) {
        llvm_unreachable(
            ("Unexpected line in custom cost model: " + Line).c_str());
      }
      CostModel[{OpcodeStr, PrecisionStr}] = std::stod(CostStr);
    }
    Loaded = true;
  }
  return CostModel;
}

// Hashes the parsed map rather than the file bytes so comments and row order
// stay out of the DP-cache key; a changed price invalidates the cached table.
uint64_t getCostModelFingerprint() {
  static uint64_t FP = 0;
  static bool Computed = false;
  if (Computed)
    return FP;
  Computed = true;
  if (costModelPath().empty())
    return FP;
  const auto &Model = getCostModel();
  uint64_t h = 1469598103934665603ULL;
  auto mix = [&](uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  auto mixStr = [&](const std::string &s) {
    for (unsigned char c : s)
      mix((uint64_t)c);
    mix(0x5eULL);
  };
  for (const auto &kv : Model) { // std::map => deterministic order
    mixStr(kv.first.first);
    mixStr(kv.first.second);
    uint64_t bits;
    double d = kv.second;
    std::memcpy(&bits, &d, sizeof(bits));
    mix(bits);
  }
  mixStr(CostModelNativeArch);
  FP = h;
  return FP;
}

void requireCostModel(const Function &F) {
  StringRef targetCpu = F.getFnAttribute("target-cpu").getValueAsString();
  resolveForTarget(targetCpu, F);
  getCostModel();
  if (CostModelNativeArch.empty())
    report_fatal_error("Poseidon: cost model " + Twine(costModelPath()) +
                       " carries no '# native_arch=' header, so it cannot be "
                       "matched against the compile target");
  if (targetCpu != CostModelNativeArch) {
    std::string msg =
        "Poseidon: cost model " + costModelPath() +
        " was measured on native_arch '" + CostModelNativeArch + "' but " +
        F.getName().str() + " compiles for target-cpu '" +
        (targetCpu.empty() ? std::string("<unset>") : targetCpu.str()) + "'";
    report_fatal_error(Twine(msg));
  }
}

const std::string &getCostModelNativeArch() {
  if (!costModelPath().empty())
    getCostModel();
  return CostModelNativeArch;
}

double queryCostModel(const std::string &OpcodeName,
                      const std::string &PrecisionName) {
  const auto &CostModel = getCostModel();
  auto Key = std::make_pair(OpcodeName, PrecisionName);
  auto It = CostModel.find(Key);
  if (It != CostModel.end())
    return It->second;

  std::string msg = "Custom cost model: entry not found for " + OpcodeName +
                    " @ " + PrecisionName;
  llvm::errs() << msg << "\n";
  llvm_unreachable(msg.c_str());
}

double queryCostModelOr(const std::string &OpcodeName,
                        const std::string &PrecisionName, double fallback) {
  const auto &CostModel = getCostModel();
  auto It = CostModel.find(std::make_pair(OpcodeName, PrecisionName));
  return (It != CostModel.end()) ? It->second : fallback;
}

double getInstructionCompCost(const Instruction *I) {
  if (!I->getType()->isFPOrFPVectorTy())
    return 0.0;

  std::string OpcodeName;
  switch (I->getOpcode()) {
  case Instruction::FNeg:
    OpcodeName = "fneg";
    break;
  case Instruction::FAdd:
    OpcodeName = "fadd";
    break;
  case Instruction::FSub:
    OpcodeName = "fsub";
    break;
  case Instruction::FMul:
    OpcodeName = "fmul";
    break;
  case Instruction::FDiv:
    OpcodeName = "fdiv";
    break;
  case Instruction::FCmp:
    OpcodeName = "fcmp";
    break;
  case Instruction::FPExt:
    OpcodeName = "fpext";
    break;
  case Instruction::FPTrunc:
    OpcodeName = "fptrunc";
    break;
  case Instruction::PHI:
  case Instruction::Select:
  case Instruction::Load:
    return 0;
  case Instruction::Call: {
    auto *Call = cast<CallInst>(I);
    if (auto *CalledFunc = Call->getCalledFunction()) {
      if (CalledFunc->isIntrinsic()) {
        switch (CalledFunc->getIntrinsicID()) {
        case Intrinsic::sin:
          OpcodeName = "sin";
          break;
        case Intrinsic::cos:
          OpcodeName = "cos";
          break;
#if LLVM_VERSION_MAJOR > 16
        case Intrinsic::tan:
          OpcodeName = "tan";
          break;
        case Intrinsic::asin:
          OpcodeName = "asin";
          break;
        case Intrinsic::acos:
          OpcodeName = "acos";
          break;
        case Intrinsic::atan:
          OpcodeName = "atan";
          break;
        case Intrinsic::atan2:
          OpcodeName = "atan2";
          break;
        case Intrinsic::sinh:
          OpcodeName = "sinh";
          break;
        case Intrinsic::cosh:
          OpcodeName = "cosh";
          break;
        case Intrinsic::tanh:
          OpcodeName = "tanh";
          break;
#endif
        case Intrinsic::exp:
          OpcodeName = "exp";
          break;
        case Intrinsic::log:
          OpcodeName = "log";
          break;
        case Intrinsic::sqrt:
        case Intrinsic::nvvm_sqrt_approx_f:
          OpcodeName = "sqrt";
          break;
        case Intrinsic::fabs:
          OpcodeName = "fabs";
          break;
        case Intrinsic::fma:
          OpcodeName = "fma";
          break;
        case Intrinsic::pow:
          OpcodeName = "pow";
          break;
        case Intrinsic::powi:
          OpcodeName = "powi";
          break;
        case Intrinsic::fmuladd:
          OpcodeName = "fmuladd";
          break;
        case Intrinsic::maxnum:
          OpcodeName = "maxnum";
          break;
        case Intrinsic::minnum:
          OpcodeName = "minnum";
          break;
        case Intrinsic::ceil:
          OpcodeName = "ceil";
          break;
        case Intrinsic::floor:
          OpcodeName = "floor";
          break;
        case Intrinsic::exp2:
          OpcodeName = "exp2";
          break;
        case Intrinsic::log10:
          OpcodeName = "log10";
          break;
        case Intrinsic::log2:
          OpcodeName = "log2";
          break;
        case Intrinsic::rint:
          OpcodeName = "rint";
          break;
        case Intrinsic::round:
          OpcodeName = "round";
          break;
        case Intrinsic::trunc:
          OpcodeName = "trunc";
          break;
        case Intrinsic::copysign:
          OpcodeName = "copysign";
          break;
        default: {
          std::string msg = "Custom cost model: unsupported intrinsic " +
                            CalledFunc->getName().str();
          llvm_unreachable(msg.c_str());
        }
        }
      } else if (StringRef deviceMath = deviceMathName(CalledFunc->getName());
                 !deviceMath.empty()) {
        OpcodeName = deviceMath.str();
        // PT materializes fmax/fmin as llvm.maxnum/llvm.minnum
        // (lib/Precision.cpp), so the priced op and the emitted op are the
        // same instruction and the same measured row serves both.
        if (OpcodeName == "fmax")
          OpcodeName = "maxnum";
        else if (OpcodeName == "fmin")
          OpcodeName = "minnum";
      } else {
        std::string FuncName = CalledFunc->getName().str();
        if (!FuncName.empty() &&
            (FuncName.back() == 'f' || FuncName.back() == 'l'))
          FuncName.pop_back();

        if (libmFuncs().count(FuncName))
          OpcodeName = FuncName;
        else {
          std::string msg =
              "Custom cost model: unknown function call " + FuncName;
          llvm_unreachable(msg.c_str());
        }
      }
    } else {
      llvm_unreachable("Custom cost model: unknown function call");
    }
    break;
  }
  default: {
    llvm::errs() << "Problematic instruction: " << *I << "\n";
    std::string msg = "Custom cost model: unexpected opcode " +
                      std::string(I->getOpcodeName());
    llvm_unreachable(msg.c_str());
  }
  }

  std::string PrecisionName;
  Type *Ty = I->getType();
  if (I->getOpcode() == Instruction::FCmp)
    Ty = I->getOperand(0)->getType();

  if (Ty->isBFloatTy())
    PrecisionName = "bf16";
  else if (Ty->isHalfTy())
    PrecisionName = "half";
  else if (Ty->isFloatTy())
    PrecisionName = "float";
  else if (Ty->isDoubleTy())
    PrecisionName = "double";
  else if (Ty->isX86_FP80Ty())
    PrecisionName = "fp80";
  else if (Ty->isFP128Ty())
    PrecisionName = "fp128";
  else {
    std::string msg = "Custom cost model: unsupported precision type!";
    llvm_unreachable(msg.c_str());
  }

  if (I->getOpcode() == Instruction::FPExt ||
      I->getOpcode() == Instruction::FPTrunc) {
    Type *SrcTy = I->getOperand(0)->getType();
    std::string SrcPrecisionName;
    if (SrcTy->isBFloatTy())
      SrcPrecisionName = "bf16";
    else if (SrcTy->isHalfTy())
      SrcPrecisionName = "half";
    else if (SrcTy->isFloatTy())
      SrcPrecisionName = "float";
    else if (SrcTy->isDoubleTy())
      SrcPrecisionName = "double";
    else if (SrcTy->isX86_FP80Ty())
      SrcPrecisionName = "fp80";
    else if (SrcTy->isFP128Ty())
      SrcPrecisionName = "fp128";
    else {
      std::string msg = "Custom cost model: unsupported precision type!";
      llvm_unreachable(msg.c_str());
    }

    OpcodeName += "_" + SrcPrecisionName + "_to_" + PrecisionName;
    PrecisionName = SrcPrecisionName;
  }

  return queryCostModel(OpcodeName, PrecisionName);
}

const std::unordered_set<std::string> &getPTFuncs() {
  static const std::unordered_set<std::string> PTFuncs = []() {
    std::unordered_set<std::string> funcs;
    for (const auto &func : libmFuncs()) {
      double costFP32 = queryCostModel(func, "float");
      double costFP64 = queryCostModel(func, "double");
      double costFPTrunc = queryCostModel("fptrunc_double_to_float", "double");
      double costFPExt = queryCostModel("fpext_float_to_double", "float");
      double costFPCast = costFPTrunc + costFPExt;
      if (costFP32 + costFPCast < costFP64)
        funcs.insert(func);
    }
    return funcs;
  }();
  return PTFuncs;
}

double computeMaxCost(BasicBlock *BB,
                      std::unordered_map<BasicBlock *, double> &MaxCost,
                      std::unordered_set<BasicBlock *> &Visited) {
  if (MaxCost.find(BB) != MaxCost.end())
    return MaxCost[BB];

  if (!Visited.insert(BB).second)
    return 0;

  double BBCost = 0;
  for (const Instruction &I : *BB) {
    if (I.isTerminator())
      continue;

    auto instCost = getInstructionCompCost(&I);

    BBCost += instCost;
  }

  double succCost = 0;

  if (!succ_empty(BB)) {
    double maxSuccCost = 0;
    for (BasicBlock *Succ : successors(BB)) {
      double succBBCost = computeMaxCost(Succ, MaxCost, Visited);
      if (succBBCost > maxSuccCost)
        maxSuccCost = succBBCost;
    }
    succCost = maxSuccCost;
  }

  double totalCost = BBCost + succCost;
  MaxCost[BB] = totalCost;
  Visited.erase(BB);
  return totalCost;
}

double getCompCost(Function *F) {
  std::unordered_map<BasicBlock *, double> MaxCost;
  std::unordered_set<BasicBlock *> Visited;

  BasicBlock *EntryBB = &F->getEntryBlock();
  double TotalCost = computeMaxCost(EntryBB, MaxCost, Visited);
  return TotalCost;
}

double getCompCost(const SmallVector<Value *> &outputs,
                   const SetVector<Value *> &inputs) {
  assert(!outputs.empty());
  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;
  double cost = 0;

  todo.insert(todo.end(), outputs.begin(), outputs.end());
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (inputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      // TODO: unfair to ignore branches when calculating cost
      cost += getInstructionCompCost(I);

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
      for (auto &operand : operands) {
        todo.push_back(operand);
      }
    }
  }

  return cost;
}

} // namespace poseidon

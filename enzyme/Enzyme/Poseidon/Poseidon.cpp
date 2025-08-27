#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/IR/CmpPredicate.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../Utils.h"
#include "Poseidon.h"
#include "PoseidonHerbieUtils.h"
#include "PoseidonPrecUtils.h"
#include "PoseidonProfUtils.h"
#include "PoseidonSolvers.h"
#include "PoseidonTypes.h"
#include "PoseidonUtils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "fp-opt"

extern "C" {
cl::opt<bool> FPProfileGenerate(
    "fpprofile-generate", cl::init(false), cl::Hidden,
    cl::desc("Generate instrumented program for FP profiling"));
cl::opt<std::string> FPProfileUse(
    "fpprofile-use", cl::Hidden, cl::value_desc("directory"), cl::ValueOptional,
    cl::desc("FP profile directory to read from for FP optimization"));
cl::opt<bool> FPOptPrint("fpopt-print", cl::init(false), cl::Hidden,
                         cl::desc("Print FPOpt debug info"));
cl::opt<bool> FPOptEnableHerbie(
    "fpopt-enable-herbie", cl::init(true), cl::Hidden,
    cl::desc("Use Herbie to rewrite floating-point expressions"));
cl::opt<bool> FPOptEnablePT(
    "fpopt-enable-pt", cl::init(true), cl::Hidden,
    cl::desc("Consider precision changes of floating-point expressions"));
cl::opt<std::string> FPOptCachePath("fpopt-cache-path", cl::init("cache"),
                                    cl::Hidden,
                                    cl::desc("Path to cache Herbie results"));
cl::opt<bool> FPOptEnableSolver(
    "fpopt-enable-solver", cl::init(true), cl::Hidden,
    cl::desc("Use the solver to select desirable rewrite candidates; when "
             "disabled, apply all Herbie's first choices"));
cl::opt<unsigned> FPOptMaxExprDepth(
    "fpopt-max-expr-depth", cl::init(100), cl::Hidden,
    cl::desc(
        "The maximum depth of expression construction; abort if exceeded"));
cl::opt<unsigned> FPOptMaxExprLength(
    "fpopt-max-expr-length", cl::init(10000), cl::Hidden,
    cl::desc("The maximum length of an expression; abort if exceeded"));
cl::opt<std::string> FPOptReductionEval(
    "fpopt-reduction-eval", cl::init("arithmean"), cl::Hidden,
    cl::desc("Which reduction result to use in candidate evaluation. "
             "Options are 'geomean', 'arithmean', and 'maxabs'"));
cl::opt<unsigned> FPOptMinUsesForSplit(
    "fpopt-min-uses-split", cl::init(3), cl::Hidden,
    cl::desc("Minimum number of uses of bottleneck node to trigger split"));
cl::opt<unsigned>
    FPOptMinOpsForSplit("fpopt-min-ops-split", cl::init(3), cl::Hidden,
                        cl::desc("Minimum number of upstream operations of "
                                 "bottleneck node to trigger split"));
cl::opt<bool>
    FPOptAggressiveDCE("fpopt-aggressive-dce", cl::init(true), cl::Hidden,
                       cl::desc("Aggressively eliminate zero gradient outputs "
                                "as dead code (non-conditional only)"));
}

bool Poseidonable(const llvm::Value &V) {
  const Instruction *I = dyn_cast<Instruction>(&V);
  if (!I)
    return false;

  switch (I->getOpcode()) {
  case Instruction::FNeg:
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    return I->getType()->isFloatTy() || I->getType()->isDoubleTy();
  case Instruction::Call: {
    const CallInst *CI = dyn_cast<CallInst>(I);
    if (CI && CI->getCalledFunction() &&
        (CI->getType()->isFloatTy() || CI->getType()->isDoubleTy())) {
      StringRef funcName = CI->getCalledFunction()->getName();
      return
          // LLVM intrinsics
          startsWith(funcName, "llvm.sin.") ||
          startsWith(funcName, "llvm.cos.") ||
          startsWith(funcName, "llvm.tan.") ||
          startsWith(funcName, "llvm.asin.") ||
          startsWith(funcName, "llvm.acos.") ||
          startsWith(funcName, "llvm.atan.") ||
          startsWith(funcName, "llvm.atan2.") ||
          startsWith(funcName, "llvm.sinh.") ||
          startsWith(funcName, "llvm.cosh.") ||
          startsWith(funcName, "llvm.tanh.") ||
          startsWith(funcName, "llvm.exp.") ||
          startsWith(funcName, "llvm.log.") ||
          startsWith(funcName, "llvm.sqrt.") ||
          startsWith(funcName, "llvm.pow.") ||
          startsWith(funcName, "llvm.powi.") ||
          startsWith(funcName, "llvm.fabs.") ||
          startsWith(funcName, "llvm.fma.") ||
          startsWith(funcName, "llvm.fmuladd.") ||
          startsWith(funcName, "llvm.maxnum.") ||
          startsWith(funcName, "llvm.minnum.") ||
          startsWith(funcName, "llvm.ceil.") ||
          startsWith(funcName, "llvm.floor.") ||
          startsWith(funcName, "llvm.exp2.") ||
          startsWith(funcName, "llvm.log10.") ||
          startsWith(funcName, "llvm.log2.") ||
          startsWith(funcName, "llvm.rint.") ||
          startsWith(funcName, "llvm.round.") ||
          startsWith(funcName, "llvm.trunc.") ||
          startsWith(funcName, "llvm.copysign.") ||
          startsWith(funcName, "llvm.fdim.") ||
          startsWith(funcName, "llvm.fmod.") ||

          // libm functions
          funcName == "sin" || funcName == "sinf" || funcName == "cos" ||
          funcName == "cosf" || funcName == "tan" || funcName == "tanf" ||
          funcName == "asin" || funcName == "asinf" || funcName == "acos" ||
          funcName == "acosf" || funcName == "atan" || funcName == "atanf" ||
          funcName == "atan2" || funcName == "atan2f" || funcName == "sinh" ||
          funcName == "sinhf" || funcName == "cosh" || funcName == "coshf" ||
          funcName == "tanh" || funcName == "tanhf" || funcName == "asinh" ||
          funcName == "asinhf" || funcName == "acosh" || funcName == "acoshf" ||
          funcName == "atanh" || funcName == "atanhf" || funcName == "sqrt" ||
          funcName == "sqrtf" || funcName == "cbrt" || funcName == "cbrtf" ||
          funcName == "pow" || funcName == "powf" || funcName == "exp" ||
          funcName == "expf" || funcName == "log" || funcName == "logf" ||
          funcName == "fabs" || funcName == "fabsf" || funcName == "fma" ||
          funcName == "fmaf" || funcName == "hypot" || funcName == "hypotf" ||
          funcName == "expm1" || funcName == "expm1f" || funcName == "log1p" ||
          funcName == "log1pf" || funcName == "ceil" || funcName == "ceilf" ||
          funcName == "floor" || funcName == "floorf" || funcName == "erf" ||
          funcName == "erff" || funcName == "exp2" || funcName == "exp2f" ||
          funcName == "lgamma" || funcName == "lgammaf" ||
          funcName == "log10" || funcName == "log10f" || funcName == "log2" ||
          funcName == "log2f" || funcName == "rint" || funcName == "rintf" ||
          funcName == "round" || funcName == "roundf" || funcName == "tgamma" ||
          funcName == "tgammaf" || funcName == "trunc" ||
          funcName == "truncf" || funcName == "copysign" ||
          funcName == "copysignf" || funcName == "fdim" ||
          funcName == "fdimf" || funcName == "fmod" || funcName == "fmodf" ||
          funcName == "remainder" || funcName == "remainderf";
    }
    return false;
  }
  default:
    return false;
  }
}

void setPoseidonMetadata(Function &F) {
  for (auto [idx, I] : enumerate(instructions(F))) {
    if (Poseidonable(I)) {
      I.setMetadata("enzyme_active", MDNode::get(I.getContext(), {}));
      I.setMetadata("enzyme_fpprofile_idx",
                    MDNode::get(I.getContext(),
                                {ConstantAsMetadata::get(ConstantInt::get(
                                    Type::getInt64Ty(I.getContext()), idx))}));
    }
  }
}

void preprocessForPoseidon(Function *F) {
  using namespace llvm::PatternMatch;

  // fmul + fadd -> fmuladd
  for (auto &BB : *F) {
    for (auto &I : make_early_inc_range(BB)) {
      Value *X, *Y, *Z;

      if (auto *FAdd = dyn_cast<BinaryOperator>(&I)) {
        if (!isa<FPMathOperator>(FAdd) || !FAdd->hasAllowReassoc() ||
            !FAdd->hasAllowContract())
          continue;

        // fadd (fmul X, Y), Z
        if (match(FAdd, m_FAdd(m_OneUse(m_FMul(m_Value(X), m_Value(Y))),
                               m_Value(Z)))) {
          IRBuilder<> B(FAdd);
          B.setFastMathFlags(FAdd->getFastMathFlags());

          Value *FMulAdd =
              B.CreateIntrinsic(Intrinsic::fmuladd, FAdd->getType(), {X, Y, Z});
          FAdd->replaceAllUsesWith(FMulAdd);
          FAdd->eraseFromParent();
        }
        // fadd Z, (fmul X, Y)
        else if (match(FAdd,
                       m_FAdd(m_Value(Z),
                              m_OneUse(m_FMul(m_Value(X), m_Value(Y)))))) {
          IRBuilder<> B(FAdd);
          B.setFastMathFlags(FAdd->getFastMathFlags());

          Value *FMulAdd =
              B.CreateIntrinsic(Intrinsic::fmuladd, FAdd->getType(), {X, Y, Z});
          FAdd->replaceAllUsesWith(FMulAdd);

          FAdd->eraseFromParent();
        }
      }
    }
  }

  for (auto &BB : *F) {
    for (auto &I : make_early_inc_range(BB)) {
      Value *X, *Y, *Z;

      if (auto *FSub = dyn_cast<BinaryOperator>(&I)) {
        if (!isa<FPMathOperator>(FSub) || !FSub->hasAllowReassoc() ||
            !FSub->hasAllowContract())
          continue;

        // Pattern: fsub (fmul X, Y), Z -> fmuladd(X, Y, -Z)
        if (match(FSub, m_FSub(m_OneUse(m_FMul(m_Value(X), m_Value(Y))),
                               m_Value(Z)))) {
          IRBuilder<> B(FSub);
          B.setFastMathFlags(FSub->getFastMathFlags());

          Value *NegZ = B.CreateFNeg(Z);
          Value *FMulAdd = B.CreateIntrinsic(Intrinsic::fmuladd,
                                             FSub->getType(), {X, Y, NegZ});
          FSub->replaceAllUsesWith(FMulAdd);
          FSub->eraseFromParent();
        }
        // Pattern: fsub Z, (fmul X, Y) -> fmuladd(-X, Y, Z)
        else if (match(FSub,
                       m_FSub(m_Value(Z),
                              m_OneUse(m_FMul(m_Value(X), m_Value(Y)))))) {
          IRBuilder<> B(FSub);
          B.setFastMathFlags(FSub->getFastMathFlags());

          Value *NegX = B.CreateFNeg(X);
          Value *FMulAdd = B.CreateIntrinsic(Intrinsic::fmuladd,
                                             FSub->getType(), {NegX, Y, Z});
          FSub->replaceAllUsesWith(FMulAdd);
          FSub->eraseFromParent();
        }
      }
    }
  }

  // fcmp + select -> fmax/fmin
  for (auto &BB : *F) {
    for (auto &I : make_early_inc_range(BB)) {
      if (auto *Select = dyn_cast<SelectInst>(&I)) {
        Value *Cond = Select->getCondition();
        Value *TrueVal = Select->getTrueValue();
        Value *FalseVal = Select->getFalseValue();

        if (!Select->getType()->isFloatingPointTy())
          continue;

        CmpPredicate Pred;
        Value *CmpLHS, *CmpRHS;

        if (match(Cond, m_FCmp(Pred, m_Value(CmpLHS), m_Value(CmpRHS)))) {
          IRBuilder<> B(Select);
          Value *Result = nullptr;

          // select (fcmp ogt X, 0.0), X, 0.0 -> maxnum(X, 0.0)
          if (Pred == FCmpInst::FCMP_OGT && match(CmpRHS, m_AnyZeroFP()) &&
              CmpLHS == TrueVal && match(FalseVal, m_AnyZeroFP())) {
            Result = B.CreateIntrinsic(
                Intrinsic::maxnum, CmpLHS->getType(),
                {CmpLHS, ConstantFP::get(CmpLHS->getType(), 0.0)});
          }
          // select (fcmp olt X, 0.0), 0.0, X -> maxnum(X, 0.0)
          else if (Pred == FCmpInst::FCMP_OLT && match(CmpRHS, m_AnyZeroFP()) &&
                   CmpLHS == FalseVal && match(TrueVal, m_AnyZeroFP())) {
            Result = B.CreateIntrinsic(
                Intrinsic::maxnum, CmpLHS->getType(),
                {CmpLHS, ConstantFP::get(CmpLHS->getType(), 0.0)});
          }
          // select (fcmp ogt X, Y), X, Y -> maxnum(X, Y)
          else if (Pred == FCmpInst::FCMP_OGT && CmpLHS == TrueVal &&
                   CmpRHS == FalseVal) {
            Result = B.CreateIntrinsic(Intrinsic::maxnum, CmpLHS->getType(),
                                       {CmpLHS, CmpRHS});
          }
          // select (fcmp olt X, Y), X, Y -> minnum(X, Y)
          else if (Pred == FCmpInst::FCMP_OLT && CmpLHS == TrueVal &&
                   CmpRHS == FalseVal) {
            Result = B.CreateIntrinsic(Intrinsic::minnum, CmpLHS->getType(),
                                       {CmpLHS, CmpRHS});
          }

          if (Result) {
            Select->replaceAllUsesWith(Result);
            Select->eraseFromParent();

            if (auto *FCmp = dyn_cast<FCmpInst>(Cond)) {
              if (FCmp->use_empty()) {
                FCmp->eraseFromParent();
              }
            }
          }
        }
      }
    }
  }
}

// Run (our choice of) floating point optimizations on function `F`.
// Return whether or not we change the function.
bool fpOptimize(Function &F, const TargetTransformInfo &TTI,
                double relErrorTol) {
  bool changed = false;

  const std::string functionName = F.getName().str();
  assert(!FPProfileUse.empty());
  SmallString<128> profilePathBuf(FPProfileUse);
  llvm::sys::path::append(profilePathBuf, F.getName() + ".fpprofile");
  const std::string profilePath = profilePathBuf.str().str();

  if (!FPOptCachePath.empty()) {
    if (auto EC = llvm::sys::fs::create_directories(FPOptCachePath, true))
      llvm::errs() << "Warning: Could not create cache directory: "
                   << EC.message() << "\n";
  }

  std::unordered_map<size_t, ProfileInfo> profileMap;
  if (!profilePath.empty()) {
    parseProfileFile(profilePath, profileMap);
    if (profileMap.empty()) {
      llvm::errs() << "Warning: No profile data found in " << profilePath
                   << "\n";
    }
  }

  int symbolCounter = 0;
  auto getNextSymbol = [&symbolCounter]() -> std::string {
    return "v" + std::to_string(symbolCounter++);
  };

  // Extract change:

  // E1) create map<Value, FPNode> for all instructions I, map[I] = FPLLValue(I)
  // E2) for all instructions, if Poseidonable(I), map[I] = FPNode(operation(I),
  // map[operands(I)])
  // E3) floodfill for all starting locations I to find all distinct graphs /
  // outputs.

  /*
  B1:
    x = sin(arg)

  B2:
    y = 1 - x * x


  -> result y = cos(arg)^2

B1:
  nothing

B2:
  costmp = cos(arg)
  y = costmp * costmp

  */

  std::unordered_map<Value *, std::shared_ptr<FPNode>> valueToNodeMap;
  std::unordered_map<std::string, Value *> symbolToValueMap;

  llvm::errs() << "FPOpt: Starting Floodfill for " << F.getName() << "\n";

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!Poseidonable(I)) {
        valueToNodeMap[&I] =
            std::make_shared<FPLLValue>(&I, "__nh", "__nh"); // Non-Poseidonable
        if (FPOptPrint)
          llvm::errs()
              << "Registered FPLLValue for non-Poseidonable instruction: " << I
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
      auto node = std::make_shared<FPLLValue>(&I, getHerbieOperator(I), dtype);

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
            if (FPOptPrint)
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
            if (FPOptPrint)
              llvm::errs() << "Registered FPNode for " << dtype
                           << " constant: " << value << "\n";
          } else if (auto CI = dyn_cast<ConstantInt>(operand)) {
            // e.g., powi intrinsic has a constant int as its exponent
            double exponent = static_cast<double>(CI->getSExtValue());
            std::string dtype = "f64";
            std::string doubleStr = std::to_string(exponent);
            valueToNodeMap[operand] =
                std::make_shared<FPConst>(doubleStr.c_str(), dtype);
            if (FPOptPrint)
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
            if (FPOptPrint)
              llvm::errs() << "Registered FPNode for global variable: " << *GV
                           << "\n";
          } else {
            assert(0 && "Unknown operand");
          }
        }
        node->addOperand(valueToNodeMap[operand]);
      }
      valueToNodeMap[&I] = node;
    }
  }

  SmallSet<Value *, 8> processed;
  SmallVector<Subgraph, 1> subgraphs;
  for (auto &BB : F) {
    for (auto &I : BB) {
      // Not a Poseidonable instruction, doesn't make sense to create graph node
      // out of.
      if (!Poseidonable(I)) {
        if (FPOptPrint)
          llvm::errs() << "Skipping non-Poseidonable instruction: " << I
                       << "\n";
        continue;
      }

      // Instruction is already in a set
      if (processed.contains(&I)) {
        if (FPOptPrint)
          llvm::errs() << "Skipping already seen instruction: " << I << "\n";
        continue;
      }

      if (FPOptPrint)
        llvm::errs() << "Starting floodfill from: " << I << "\n";

      SmallVector<Value *, 8> todo;
      SetVector<Value *> input_seen;
      SetVector<Instruction *> output_seen;
      SetVector<Instruction *> operation_seen;
      todo.push_back(&I);
      while (!todo.empty()) {
        auto cur = todo.pop_back_val();
        assert(valueToNodeMap.count(cur) && "Node not found in valueToNodeMap");

        // We now can assume that this is a Poseidonable expression
        // Since we can only herbify instructions, let's assert that
        assert(isa<Instruction>(cur));
        auto I2 = cast<Instruction>(cur);

        // Don't repeat any instructions we've already seen (to avoid loops
        // for phi nodes)
        if (operation_seen.contains(I2)) {
          if (FPOptPrint)
            llvm::errs() << "Skipping already seen instruction: " << *I2
                         << "\n";
          continue;
        }

        assert(!processed.contains(cur));

        if (FPOptPrint)
          llvm::errs() << "Insert to operation_seen and processed: " << *I2
                       << "\n";
        operation_seen.insert(I2);
        processed.insert(cur);

        auto operands =
            isa<CallInst>(I2) ? cast<CallInst>(I2)->args() : I2->operands();

        for (const auto &operand : operands) {
          if (!Poseidonable(*operand)) {
            if (FPOptPrint)
              llvm::errs() << "Non-Poseidonable input found: " << *operand
                           << "\n";

            // Don't mark constants as input `llvm::Value`s
            if (!isa<ConstantFP>(operand))
              input_seen.insert(operand);
          } else {
            if (FPOptPrint)
              llvm::errs() << "Adding operand to todo list: " << *operand
                           << "\n";
            todo.push_back(operand);
          }
        }

        for (auto U : I2->users()) {
          if (auto I3 = dyn_cast<Instruction>(U)) {
            if (!Poseidonable(*I3)) {
              if (FPOptPrint)
                llvm::errs() << "Output instruction found: " << *I2 << "\n";
              output_seen.insert(I2);
            } else {
              if (FPOptPrint)
                llvm::errs() << "Adding user to todo list: " << *I3 << "\n";
              todo.push_back(I3);
            }
          }
        }
      }

      // Don't bother with graphs without any Poseidonable operations
      if (!operation_seen.empty()) {
        if (FPOptPrint) {
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

        if (operation_seen.size() == 1) {
          if (FPOptPrint)
            llvm::errs() << "Skipping trivial subgraph\n";
          continue;
        }

        subgraphs.emplace_back(input_seen, output_seen, operation_seen);
      }
    }
  }

  if (FPOptPrint) {
    llvm::errs() << "FPOpt: Found " << subgraphs.size()
                 << " initial subgraphs in " << F.getName() << "\n";
  }

  if (FPOptAggressiveDCE) {
    auto subgraphIt = subgraphs.begin();
    while (subgraphIt != subgraphs.end()) {
      Subgraph &subgraph = *subgraphIt;

      // Don't remove zero gradient outputs that are used as predicates
      SmallVector<Instruction *, 4> deadOutputs;
      for (auto *output : subgraph.outputs) {
        if (valueToNodeMap[output]->grad == 0.) {
          bool isPredicate = false;
          for (auto *user : output->users()) {
            if (isa<FCmpInst>(user)) {
              isPredicate = true;
              break;
            }
          }
          if (!isPredicate) {
            if (FPOptPrint)
              llvm::errs() << "Aggressive DCE: found zero gradient output: "
                           << *output << "\n";
            deadOutputs.push_back(output);
          }
        }
      }

      for (auto *deadOutput : deadOutputs) {
        if (FPOptPrint)
          llvm::errs() << "Eliminating zero gradient output as dead code: "
                       << *deadOutput << "\n";

        if (!deadOutput->use_empty()) {
          deadOutput->replaceAllUsesWith(
              UndefValue::get(deadOutput->getType()));
        }

        subgraph.outputs.remove(deadOutput);
      }

      SmallVector<Instruction *, 8> sorted;
      reverseTopoSort(subgraph.operations, sorted);

      for (auto *inst : sorted) {
        if (inst->use_empty()) {
          if (FPOptPrint)
            llvm::errs() << "Aggressive DCE: eliminating unused instruction: "
                         << *inst << "\n";

          subgraph.operations.remove(inst);
          valueToNodeMap.erase(inst);
          inst->eraseFromParent();
        }
      }

      if (subgraph.outputs.empty()) {
        if (FPOptPrint)
          llvm::errs() << "Removing empty subgraph\n";
        subgraphIt = subgraphs.erase(subgraphIt);
      } else {
        ++subgraphIt;
      }
    }
  }

  if (FPOptPrint && FPOptAggressiveDCE) {
    llvm::errs() << "FPOpt: After aggressive DCE, have " << subgraphs.size()
                 << " subgraphs in " << F.getName() << "\n";
  }

  splitSubgraphs(subgraphs);

  if (FPOptPrint) {
    llvm::errs() << "FPOpt: After splitting, have " << subgraphs.size()
                 << " subgraphs in " << F.getName() << "\n";
  }

  for (auto &subgraph : subgraphs) {
    for (auto op : subgraph.operations) {
      if (auto MD = op->getMetadata("enzyme_fpprofile_idx")) {
        if (auto C = dyn_cast<ConstantAsMetadata>(MD->getOperand(0))) {
          size_t idx = cast<ConstantInt>(C->getValue())->getZExtValue();
          auto it = profileMap.find(idx);

          if (it != profileMap.end()) {
            const auto &profileInfo = it->second;

            auto node = valueToNodeMap[op];
            node->sens = profileInfo.sumSens;
            node->grad = profileInfo.sumGrad;
            node->executions = profileInfo.exec;
            node->updateBounds(profileInfo.minRes, profileInfo.maxRes);

            if (FPOptPrint) {
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
                                          profileInfo.maxOperands[i]);
                if (FPOptPrint) {
                  llvm::errs() << "Range of " << *operand << " is ["
                               << operandNode->getLowerBound() << ", "
                               << operandNode->getUpperBound() << "]\n";
                }
              }
            }
          } else {
            if (!FPOptLooseCoverage) {
              llvm::errs() << "FP Instruction " << *op
                           << " has no execution logged (idx=" << idx << ")!\n";
              llvm_unreachable("Unexecuted instruction found; set "
                               "-fpopt-loose-coverage "
                               "to suppress this error\n");
            }
            if (FPOptPrint)
              llvm::errs() << "Sensitivity of " << *op
                           << " not found in the log; using 0 instead\n";
          }
        }
      }
    }
  }

  // 1) Identify subgraphs of the computation which can be entirely represented
  // in herbie-style arithmetic
  // 2) Make the herbie FP-style expression by
  // converting llvm instructions into herbie string (FPNode ....)
  if (subgraphs.empty()) {
    if (FPOptPrint)
      llvm::errs() << "No subgraphs found\n";
    return false;
  }

  SmallVector<CandidateOutput, 4> COs;
  SmallVector<CandidateSubgraph, 4> CSs;

  int subgraphCounter = 0;

  for (auto &subgraph : subgraphs) {
    assert(subgraph.inputs.size() > 0 && "No inputs found for subgraph");
    if (FPOptEnableHerbie) {
      for (const auto &input : subgraph.inputs) {
        auto node = valueToNodeMap[input];
        if (node->op == "__const") {
          // Constants don't need a symbol
          continue;
        }

        if (!node->hasSymbol()) {
          node->symbol = getNextSymbol();
        }
        symbolToValueMap[node->symbol] = input;
        if (FPOptPrint)
          llvm::errs() << "assigning symbol: " << node->symbol << " to "
                       << *input << "\n";
      }

      std::vector<std::string> herbieInputs;
      std::vector<CandidateOutput> newCOs;
      int outputCounter = 0;

      assert(subgraph.outputs.size() > 0 && "No outputs found for subgraph");
      for (auto &output : subgraph.outputs) {
        // 3) run fancy opts
        double grad = valueToNodeMap[output]->grad;
        unsigned executions = valueToNodeMap[output]->executions;

        if (grad == 0.) {
          llvm::errs() << "Skipping zero gradient instruction: " << *output
                       << "\n";
          continue;
        }

        std::string expr = valueToNodeMap[output]->toFullExpression(
            valueToNodeMap, subgraph.inputs);

        // Skip trivial expressions with only one operation
        auto parenCount = std::count(expr.begin(), expr.end(), '(');
        assert(parenCount > 0);
        if (parenCount == 1) {
          if (FPOptPrint)
            llvm::errs() << "Skipping Herbie for simple expression: " << expr
                         << "\n";
          continue;
        }

        SmallSet<std::string, 8> args;
        getUniqueArgs(expr, args);

        std::string properties = ":herbie-conversions ([binary64 binary32])";
        if (valueToNodeMap[output]->dtype == "f32") {
          properties += " :precision binary32";
        } else if (valueToNodeMap[output]->dtype == "f64") {
          properties += " :precision binary64";
        } else {
          llvm_unreachable("Unexpected dtype");
        }

        std::string precondition =
            getPrecondition(args, valueToNodeMap, symbolToValueMap);
        properties += " :pre " + precondition;

        CandidateOutput CO(subgraph, output, expr, grad, executions, TTI);
        properties += " :name \"" + std::to_string(outputCounter++) + "\"";

        std::string argStr;
        for (const auto &arg : args) {
          if (!argStr.empty())
            argStr += " ";
          argStr += arg;
        }

        std::string herbieInput =
            "(FPCore (" + argStr + ") " + properties + " " + expr + ")";
        if (FPOptPrint)
          llvm::errs() << "Herbie input:\n" << herbieInput << "\n";

        if (herbieInput.length() > FPOptMaxExprLength) {
          llvm::errs() << "WARNING: Skipping Herbie optimization for "
                       << *output
                       << " since expression length exceeds limit of "
                       << FPOptMaxExprLength << "\n";
          continue;
        }

        herbieInputs.push_back(herbieInput);
        newCOs.push_back(CO);
      }

      if (!herbieInputs.empty()) {
        if (!improveViaHerbie(herbieInputs, newCOs, F.getParent(), TTI,
                              valueToNodeMap, symbolToValueMap,
                              subgraphCounter)) {
          if (FPOptPrint)
            llvm::errs() << "Failed to optimize expressions using Herbie!\n";
        }

        COs.insert(COs.end(), newCOs.begin(), newCOs.end());
      }
    }

    if (FPOptEnablePT) {
      // Sort `cs.operations` by the gradient and construct
      // `PrecisionChange`s.
      CandidateSubgraph CS(subgraph, TTI);
      auto *o0 = subgraph.outputs[0];
      CS.executions = valueToNodeMap[o0]->executions;

      const SmallVector<PrecisionChangeType> precTypes{
          PrecisionChangeType::FP32,
          PrecisionChangeType::FP64,
      };

      const auto &PTFuncs = getPTFuncs();

      // Check if we have a cached DP table
      std::string cacheFilePath = FPOptCachePath + "/table.json";
      bool skipEvaluation = FPOptSolverType == "dp" &&
                            !FPOptCachePath.empty() &&
                            llvm::sys::fs::exists(cacheFilePath);

      SetVector<FPLLValue *> operations;
      for (auto *I : subgraph.operations) {
        assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
               "Corrupted FPNode for original instructions");
        auto node = cast<FPLLValue>(valueToNodeMap[I].get());
        if (PTFuncs.count(node->op) != 0) {
          operations.insert(node);
        }
      }

      // Prioritize operations with low sensitivity scores
      SmallVector<FPLLValue *> sortedOps(operations.begin(), operations.end());
      llvm::sort(sortedOps, [](const auto &a, const auto &b) {
        return a->sens < b->sens;
      });

      // Create PrecisionChanges for 0-10%, 0-20%, ..., up to 0-100%
      for (int percent = 10; percent <= 100; percent += 10) {
        size_t numToChange = sortedOps.size() * percent / 100;

        if (FPOptPrint && numToChange > 0) {
          llvm::errs() << "Created PrecisionChange for " << percent
                       << "% of Funcs (" << numToChange << ")\n";
          double minSens = sortedOps[0]->sens;
          double maxSens = sortedOps[numToChange - 1]->sens;
          llvm::errs() << "Sensitivity score range: [" << minSens << ", "
                       << maxSens << "]\n";
        }

        for (auto prec : precTypes) {
          std::string precStr = getPrecisionChangeTypeString(prec).str();
          std::string desc =
              "Funcs 0% -- " + std::to_string(percent) + "% -> " + precStr;

          SetVector<FPLLValue *> nodesToChange(sortedOps.begin(),
                                               sortedOps.begin() + numToChange);
          PrecisionChange change(
              nodesToChange,
              getPrecisionChangeType(subgraph.outputs[0]->getType()), prec);

          SmallVector<PrecisionChange, 1> changes{std::move(change)};
          PTCandidate candidate{std::move(changes), desc};

          if (!skipEvaluation) {
            candidate.CompCost = getCompCost(subgraph, TTI, candidate);
          }

          CS.candidates.push_back(std::move(candidate));
        }
      }

      SetVector<FPLLValue *> allOperations;
      for (auto *I : subgraph.operations) {
        assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
               "Corrupted FPNode for original instructions");
        auto node = cast<FPLLValue>(valueToNodeMap[I].get());
        allOperations.insert(node);
      }

      // Prioritize operations with low sensitivity scores
      SmallVector<FPLLValue *> sortedAllOps(allOperations.begin(),
                                            allOperations.end());
      llvm::sort(sortedAllOps, [](const auto &a, const auto &b) {
        return a->sens < b->sens;
      });

      // Create PrecisionChanges for 0-10%, 0-20%, ..., up to 0-100%
      for (int percent = 10; percent <= 100; percent += 10) {
        size_t numToChange = sortedAllOps.size() * percent / 100;

        if (FPOptPrint && numToChange > 0) {
          llvm::errs() << "Created PrecisionChange for " << percent
                       << "% of all operations (" << numToChange << ")\n";
          double minSens = sortedAllOps[0]->sens;
          double maxSens = sortedAllOps[numToChange - 1]->sens;
          llvm::errs() << "Sensitivity score range: [" << minSens << ", "
                       << maxSens << "]\n";
        }

        for (auto prec : precTypes) {
          std::string precStr = getPrecisionChangeTypeString(prec).str();
          std::string desc =
              "All 0% -- " + std::to_string(percent) + "% -> " + precStr;

          SetVector<FPLLValue *> nodesToChange(
              sortedAllOps.begin(), sortedAllOps.begin() + numToChange);
          PrecisionChange change(
              nodesToChange,
              getPrecisionChangeType(subgraph.outputs[0]->getType()), prec);

          SmallVector<PrecisionChange, 1> changes{std::move(change)};
          PTCandidate candidate{std::move(changes), desc};

          if (!skipEvaluation) {
            candidate.CompCost = getCompCost(subgraph, TTI, candidate);
          }

          CS.candidates.push_back(std::move(candidate));
        }
      }

      if (!skipEvaluation) {
        setUnifiedAccuracyCost(CS, valueToNodeMap, symbolToValueMap);
      }

      CSs.push_back(std::move(CS));
    }
    llvm::errs() << "##### Finished synthesizing candidates for "
                 << ++subgraphCounter << " of " << subgraphs.size()
                 << " subgraphs! #####\n";
  }

  // Perform rewrites
  if (FPOptPrint) {
    if (FPOptEnableHerbie) {
      for (auto &CO : COs) {
        llvm::errs() << "\n################################\n";
        llvm::errs() << "Initial AccuracyCost: " << CO.initialAccCost << "\n";
        llvm::errs() << "Initial ComputationCost: " << CO.initialCompCost
                     << "\n";
        llvm::errs() << "Initial HerbieCost: " << CO.initialHerbieCost << "\n";
        llvm::errs() << "Initial HerbieAccuracy: " << CO.initialHerbieAccuracy
                     << "\n";
        llvm::errs() << "Initial Expression: " << CO.expr << "\n";
        llvm::errs() << "Grad: " << CO.grad << "\n\n";
        llvm::errs() << "Candidates:\n";
        llvm::errs() << "Δ AccCost\t\tΔ "
                        "CompCost\t\tHerbieCost\t\tAccuracy\t\tExpression\n";
        llvm::errs() << "--------------------------------\n";
        for (size_t i = 0; i < CO.candidates.size(); ++i) {
          auto &candidate = CO.candidates[i];
          llvm::errs() << CO.getAccCostDelta(i) << "\t\t"
                       << CO.getCompCostDelta(i) << "\t\t"
                       << candidate.herbieCost << "\t\t"
                       << candidate.herbieAccuracy << "\t\t" << candidate.expr
                       << "\n";
        }
        llvm::errs() << "################################\n\n";
      }
    }
    if (FPOptEnablePT) {
      for (auto &CS : CSs) {
        llvm::errs() << "\n################################\n";
        llvm::errs() << "Initial AccuracyCost: " << CS.initialAccCost << "\n";
        llvm::errs() << "Initial ComputationCost: " << CS.initialCompCost
                     << "\n";
        llvm::errs() << "Candidates:\n";
        llvm::errs() << "Δ AccCost\t\tΔ CompCost\t\tDescription\n"
                     << "---------------------------\n";
        for (size_t i = 0; i < CS.candidates.size(); ++i) {
          auto &candidate = CS.candidates[i];
          llvm::errs() << CS.getAccCostDelta(i) << "\t\t"
                       << CS.getCompCostDelta(i) << "\t\t" << candidate.desc
                       << "\n";
        }
        llvm::errs() << "################################\n\n";
      }
    }
  }

  if (!FPOptEnableSolver) {
    if (FPOptEnableHerbie) {
      for (auto &CO : COs) {
        CO.apply(0, valueToNodeMap, symbolToValueMap);
        changed = true;
      }
    }
  } else {
    if (FPOptSolverType == "greedy") {
      changed =
          accuracyGreedySolver(COs, CSs, valueToNodeMap, symbolToValueMap);
    } else if (FPOptSolverType == "dp") {
      changed = accuracyDPSolver(COs, CSs, valueToNodeMap, symbolToValueMap);
    } else {
      llvm::errs() << "FPOpt: Unknown solver type: " << FPOptSolverType << "\n";
      return false;
    }
  }

  llvm::errs() << "FPOpt: Finished optimizing " << F.getName() << "\n";

  // Cleanup
  if (changed) {
    for (auto &subgraph : subgraphs) {
      if (subgraph.outputs_rewritten != subgraph.outputs.size()) {
        if (FPOptPrint)
          llvm::errs() << "Skip erasing a subgraph: only rewrote "
                       << subgraph.outputs_rewritten << " of "
                       << subgraph.outputs.size() << " outputs\n";
        continue; // Intermediate operations cannot be erased safely
      }
      for (auto *I : subgraph.operations) {
        if (FPOptPrint)
          llvm::errs() << "Erasing: " << *I << "\n";
        if (!I->use_empty()) {
          I->replaceAllUsesWith(UndefValue::get(I->getType()));
        }
        I->eraseFromParent();
      }
    }

    llvm::errs() << "FPOpt: Finished cleaning up " << F.getName() << "\n";
  }

  if (FPOptPrint) {
    llvm::errs() << "FPOpt: Finished Optimization\n";
    // F.print(llvm::errs());
  }

  return changed;
}

namespace {} // namespace

char FPOpt::ID = 0;

FPOpt::FPOpt() : FunctionPass(ID) {}

void FPOpt::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetTransformInfoWrapperPass>();
  FunctionPass::getAnalysisUsage(AU);
}

bool FPOpt::runOnFunction(Function &F) {
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  return fpOptimize(F, TTI);
}

static RegisterPass<FPOpt>
    X("fp-opt", "Run Enzyme/Poseidon Floating point optimizations");

FunctionPass *createFPOptPass() { return new FPOpt(); }

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddFPOptPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFPOptPass());
}

FPOptNewPM::Result FPOptNewPM::run(llvm::Module &M,
                                   llvm::ModuleAnalysisManager &MAM) {
  bool changed = false;
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (auto &F : M) {
    if (!F.isDeclaration()) {
      const auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
      changed |= fpOptimize(F, TTI);
    }
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey FPOptNewPM::Key;

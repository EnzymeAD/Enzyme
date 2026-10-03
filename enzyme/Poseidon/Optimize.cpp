#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
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

#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "Canonicalize.h"
#include "CostModel.h"
#include "Evaluators.h"
#include "Expansion.h"
#include "Flags.h"
#include "Herbie.h"
#include "HostDispatch.h"
#include "Matmul.h"
#include "Optimize.h"
#include "Precision.h"
#include "ProfileRead.h"
#include "RaiseWMMA.h"
#include "Solvers.h"
#include "Staging.h"
#include "Types.h"
#include "Utils.h"

// The one filename rule, shared verbatim with both FP profiler runtimes that
// write what this file reads.
#include "FPProfileName.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "poseidon"

namespace poseidon {

static constexpr int kThreeTierStep = 20;

static std::map<const Function *, Function *> &siteOriginMap() {
  static std::map<const Function *, Function *> m;
  return m;
}
void noteSiteOrigin(Function *clone, Function *orig) {
  siteOriginMap()[clone] = orig;
}

// Keyed on the SOURCE body, not on the per-call clone: profileSite shares one
// instrumented clone between a body's markers, so the run writes one record
// per body, while here a body reached from several marker calls is cloned once
// per call and LLVM uniques the second and later clone names. dFEM inlines one
// integrator's marker into every registered Q1D specialization, so a body with
// many marker calls is the normal case.
std::string siteProfileStem(const Function &clone) {
  auto it = siteOriginMap().find(&clone);
  if (it != siteOriginMap().end() && it->second)
    return profileNameStem(("preprocess_" + it->second->getName()).str());
  return profileNameStem(clone.getName().str());
}

bool redirectNoopSite(Function *clone) {
  auto it = siteOriginMap().find(clone);
  if (it == siteOriginMap().end() || !it->second)
    return false;
  Function *orig = it->second;
  if (orig == clone)
    return false;
  SmallVector<CallBase *, 4> callers;
  for (User *U : clone->users())
    if (auto *CB = dyn_cast<CallBase>(U))
      if (CB->getCalledFunction() == clone)
        callers.push_back(CB);
  for (CallBase *CB : callers)
    CB->setCalledFunction(orig);
  if (!callers.empty())
    llvm::errs() << "[poseidon] no rewrite applied for " << clone->getName()
                 << "; wrapper call restored to original body "
                 << orig->getName() << " (bit-transparent no-op)\n";
  return !callers.empty();
}

bool isOptimizable(const llvm::Value &V) {
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
    if (!CI)
      return false;

    const Function *Callee = CI->getCalledFunction();
    if (!Callee)
      return false;

    // GPU: the CUDA libdevice entry points Poseidon handles.
    if (!deviceMathName(Callee->getName()).empty())
      return true;

    if (CI->getType()->isFloatTy() || CI->getType()->isDoubleTy()) {
      StringRef funcName = Callee->getName();
      return
          // LLVM intrinsics
          funcName.starts_with("llvm.sin.") ||
          funcName.starts_with("llvm.cos.") ||
          funcName.starts_with("llvm.tan.") ||
          funcName.starts_with("llvm.asin.") ||
          funcName.starts_with("llvm.acos.") ||
          funcName.starts_with("llvm.atan.") ||
          funcName.starts_with("llvm.atan2.") ||
          funcName.starts_with("llvm.sinh.") ||
          funcName.starts_with("llvm.cosh.") ||
          funcName.starts_with("llvm.tanh.") ||
          funcName.starts_with("llvm.exp.") ||
          funcName.starts_with("llvm.log.") ||
          funcName.starts_with("llvm.sqrt.") ||
          funcName.starts_with("llvm.pow.") ||
          funcName.starts_with("llvm.powi.") ||
          funcName.starts_with("llvm.fabs.") ||
          funcName.starts_with("llvm.fma.") ||
          funcName.starts_with("llvm.fmuladd.") ||
          funcName.starts_with("llvm.maxnum.") ||
          funcName.starts_with("llvm.minnum.") ||
          funcName.starts_with("llvm.ceil.") ||
          funcName.starts_with("llvm.floor.") ||
          funcName.starts_with("llvm.exp2.") ||
          funcName.starts_with("llvm.log10.") ||
          funcName.starts_with("llvm.log2.") ||
          funcName.starts_with("llvm.rint.") ||
          funcName.starts_with("llvm.round.") ||
          funcName.starts_with("llvm.trunc.") ||
          funcName.starts_with("llvm.copysign.") ||
          funcName.starts_with("llvm.fdim.") ||
          funcName.starts_with("llvm.fmod.") ||

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

void setSlotMetadata(Function &F) {
  // The fpprofile idx must be invariant between the profile-generate and
  // profile-use compiles, which differ in their non-optimizable instructions
  // (profiler instrumentation shifts absolute positions). Index by rank among
  // optimizable ops only, so the idx stays synchronized across both compiles.
  size_t slotIdx = 0;
  for (Instruction &I : instructions(F)) {
    if (isOptimizable(I)) {
      I.setMetadata("enzyme_active", MDNode::get(I.getContext(), {}));
      I.setMetadata(
          "poseidon.prof.idx",
          MDNode::get(I.getContext(),
                      {ConstantAsMetadata::get(ConstantInt::get(
                          Type::getInt64Ty(I.getContext()), slotIdx))}));
      ++slotIdx;
    }
  }
}

void preprocess(Function *F) {
  using namespace llvm::PatternMatch;

  // fmul + fadd -> fmuladd
  for (auto &BB : *F) {
    for (auto &I : make_early_inc_range(BB)) {
      Value *X, *Y, *Z;

      if (auto *FAdd = dyn_cast<BinaryOperator>(&I)) {
        // `contract` alone licenses fusing a*b+c into one rounded fmuladd;
        // `reassoc` is about regrouping across additions and is not needed.
        // CUDA-default -ffp-contract=fast sets contract but not reassoc.
        if (!isa<FPMathOperator>(FAdd) || !FAdd->hasAllowContract())
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
        if (!isa<FPMathOperator>(FSub) || !FSub->hasAllowContract())
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

namespace {

struct TierPair {
  PrecisionChangeType hi;
  PrecisionChangeType lo;
};

struct TierTriple {
  PrecisionChangeType hi;
  PrecisionChangeType mid;
  PrecisionChangeType lo;
};

static const TierPair kCanonicalPairs[] = {
    {PrecisionChangeType::FP64, PrecisionChangeType::FP32},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion2},
    {PrecisionChangeType::FP64, PrecisionChangeType::FP16},
    {PrecisionChangeType::FP64, PrecisionChangeType::BF16},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::FP32},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::FP16},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::BF16},
    {PrecisionChangeType::FP32, PrecisionChangeType::FP16},
    {PrecisionChangeType::FP32, PrecisionChangeType::BF16},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion3},
    {PrecisionChangeType::Expansion3, PrecisionChangeType::Expansion2},
    {PrecisionChangeType::Expansion3, PrecisionChangeType::FP32},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion4},
    {PrecisionChangeType::Expansion4, PrecisionChangeType::Expansion3},
    {PrecisionChangeType::Expansion4, PrecisionChangeType::Expansion2},
    {PrecisionChangeType::Expansion4, PrecisionChangeType::FP32},
};

static const TierTriple kCanonicalTriples[] = {
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion2,
     PrecisionChangeType::FP32},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion2,
     PrecisionChangeType::FP16},
    {PrecisionChangeType::FP64, PrecisionChangeType::Expansion2,
     PrecisionChangeType::BF16},
    {PrecisionChangeType::FP64, PrecisionChangeType::FP32,
     PrecisionChangeType::FP16},
    {PrecisionChangeType::FP64, PrecisionChangeType::FP32,
     PrecisionChangeType::BF16},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::FP32,
     PrecisionChangeType::FP16},
    {PrecisionChangeType::Expansion2, PrecisionChangeType::FP32,
     PrecisionChangeType::BF16},
};

static bool precAllowed(PrecisionChangeType t, bool gpuMode,
                        const std::unordered_set<std::string> &hwScalar) {
  if (unsigned nComp = expansionComponents(t)) {
    if (!gpuMode)
      return false;
    if (!flags::EnableMultifloat)
      return false;
    if (nComp > flags::ExpansionComponents)
      return false;
  }
  if (t == PrecisionChangeType::FP16 && (!gpuMode || !hwScalar.count("half")))
    return false;
  if (t == PrecisionChangeType::BF16 && (!gpuMode || !hwScalar.count("bf16")))
    return false;
  return true;
}

static bool tierPairAllowed(TierPair tp, bool gpuMode,
                            const std::unordered_set<std::string> &hwScalar) {
  return precAllowed(tp.hi, gpuMode, hwScalar) &&
         precAllowed(tp.lo, gpuMode, hwScalar);
}

static bool tierTripleAllowed(TierTriple tr, bool gpuMode,
                              const std::unordered_set<std::string> &hwScalar) {
  return precAllowed(tr.hi, gpuMode, hwScalar) &&
         precAllowed(tr.mid, gpuMode, hwScalar) &&
         precAllowed(tr.lo, gpuMode, hwScalar);
}

static std::string fmtPrecPct(PrecisionChangeType t, int pct) {
  std::string s = getPrecisionChangeTypeString(t).str();
  s += "(";
  s += std::to_string(pct);
  s += "%)";
  return s;
}

} // namespace

// Demote FP64 PHIs whose every incoming is fpext-from-float or a
// float-representable constant. InstCombine folds fptrunc(fpext(x)) -> x but
// stops at a loop-carried PHI, so an FP32 accumulator the solver chose would
// otherwise keep its fptrunc/fpext roundtrip.
bool demoteFPCastPHIs(llvm::Function &F) {
  using namespace llvm;
  LLVMContext &Ctx = F.getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);

  auto floatExtSrc = [&](Value *V) -> Value * {
    if (auto *FE = dyn_cast<FPExtInst>(V))
      if (FE->getOperand(0)->getType()->isFloatTy())
        return FE->getOperand(0);
    return nullptr;
  };
  auto floatReprConst = [&](Value *V, Constant *&out) -> bool {
    if (isa<UndefValue>(V)) {
      out = UndefValue::get(FloatTy);
      return true;
    }
    auto *CFP = dyn_cast<ConstantFP>(V);
    if (!CFP || !CFP->getType()->isDoubleTy())
      return false;
    APFloat v = CFP->getValueAPF();
    bool losesInfo = false;
    v.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
    if (losesInfo)
      return false;
    out = ConstantFP::get(Ctx, v);
    return true;
  };

  SmallVector<PHINode *, 16> phis;
  SmallPtrSet<Value *, 16> inSet;
  for (BasicBlock &BB : F)
    for (PHINode &P : BB.phis())
      if (P.getType()->isDoubleTy()) {
        phis.push_back(&P);
        inSet.insert(&P);
      }
  if (phis.empty())
    return false;

  bool shrunk = true;
  while (shrunk) {
    shrunk = false;
    for (PHINode *P : phis) {
      if (!inSet.count(P))
        continue;
      for (Value *IV : P->incoming_values()) {
        Constant *c = nullptr;
        if (inSet.count(IV) || floatExtSrc(IV) || floatReprConst(IV, c))
          continue;
        inSet.erase(P);
        shrunk = true;
        break;
      }
    }
  }
  SmallVector<PHINode *, 16> D;
  for (PHINode *P : phis)
    if (inSet.count(P))
      D.push_back(P);
  if (D.empty())
    return false;

  // Confirm the closure invariant holds for every surviving PHI before mutating
  // anything; bail cleanly rather than leave a half-rewritten function.
  for (PHINode *P : D)
    for (Value *IV : P->incoming_values()) {
      Constant *c = nullptr;
      if (!(inSet.count(IV) || floatExtSrc(IV) || floatReprConst(IV, c)))
        return false;
    }

  DenseMap<PHINode *, PHINode *> nmap;
  for (PHINode *P : D) {
    PHINode *NP = PHINode::Create(FloatTy, P->getNumIncomingValues(),
                                  P->getName() + ".f32", P->getIterator());
    nmap[P] = NP;
  }
  for (PHINode *P : D) {
    PHINode *NP = nmap[P];
    for (unsigned i = 0; i < P->getNumIncomingValues(); ++i) {
      Value *IV = P->getIncomingValue(i);
      Constant *c = nullptr;
      Value *fv = nullptr;
      if (auto *IVP = dyn_cast<PHINode>(IV))
        if (nmap.count(IVP))
          fv = nmap[IVP];
      if (!fv)
        if (Value *s = floatExtSrc(IV))
          fv = s;
      if (!fv && floatReprConst(IV, c))
        fv = c;
      assert(fv && "demoteFPCastPHIs: non-leaf incoming survived closure");
      NP->addIncoming(fv, P->getIncomingBlock(i));
    }
  }
  for (PHINode *P : D) {
    PHINode *NP = nmap[P];
    BasicBlock *BB = NP->getParent();
    IRBuilder<> B(BB, BB->getFirstInsertionPt());
    P->replaceAllUsesWith(B.CreateFPExt(NP, DoubleTy));
  }
  for (PHINode *P : D)
    P->eraseFromParent();
  return true;
}

// An op without a profile record (-poseidon-loose-coverage) counts as zero.
static void
aggressiveDCE(Function &F, SmallVectorImpl<Subgraph> &subgraphs,
              std::unordered_map<Value *, std::shared_ptr<FPNode>> &nodes,
              const std::unordered_map<size_t, ProfileInfo> &profileMap,
              const std::unordered_set<size_t> &zeroGradIdx,
              const SmallVectorImpl<AbstractMatmul> &abstractMatmuls) {
  SmallPtrSet<Value *, 32> critical;
  SmallVector<Value *, 16> worklist;
  auto mark = [&](Value *V) {
    if (V->getType()->isFloatingPointTy() && critical.insert(V).second)
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

  if (flags::Print) {
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
      if (flags::Print)
        llvm::errs() << "Aggressive DCE: eliminating zero-gradient "
                     << "non-critical instruction: " << *op << "\n";
      op->replaceAllUsesWith(UndefValue::get(op->getType()));
      nodes.erase(op);
      it->operations.remove(op);
      it->outputs.remove(op);
      op->eraseFromParent();
    }
    if (it->outputs.empty()) {
      if (flags::Print)
        llvm::errs() << "Removing empty subgraph\n";
      it = subgraphs.erase(it);
    } else {
      ++it;
    }
  }

  if (flags::Print)
    llvm::errs() << "[poseidon] After aggressive DCE, have " << subgraphs.size()
                 << " subgraphs in " << F.getName() << "\n";
}

bool collectFPCandidates(Function &F, double errorTol, double confidence,
                         unsigned sampleLogBits, FunctionFPState &st) {
  st.F = &F;
  st.errTol = errorTol;
  st.confidence = confidence;
  st.sampleLogBits = sampleLogBits;
  requireCostModel(F);

  if (isGPUMode(F))
    llvm::errs() << "[poseidon] GPU mode active for " << F.getName() << "\n";

  assert(!flags::ProfileUse.empty());
  SmallString<128> profilePathBuf(flags::ProfileUse);
  llvm::sys::path::append(profilePathBuf, siteProfileStem(F) + ".fpprofile");
  const std::string profilePath = profilePathBuf.str().str();

  if (!flags::Cache.empty()) {
    if (auto EC = llvm::sys::fs::create_directories(flags::Cache, true))
      llvm::errs() << "Warning: Could not create cache directory: "
                   << EC.message() << "\n";
  }

  // Names the site by what it computes, not by the kernel it was cut from: the
  // profile is only readable against the canonical form it was numbered
  // against, and the Herbie result cache is keyed on the same digest so that
  // renaming the kernel does not invalidate a shipped result.
  const std::string canonicalHash = poseidon::canonicalFormHash(F);
  if (flags::Print)
    llvm::errs() << "[poseidon] canonical form hash of " << F.getName()
                 << " is " << canonicalHash << "\n";

  std::unordered_map<size_t, ProfileInfo> profileMap;
  FunctionProfileHeader profileHeader;
  std::unordered_set<size_t> zeroGradIdx;
  if (!profilePath.empty()) {
    parseProfileFile(profilePath, profileMap, &profileHeader);
    if (profileMap.empty()) {
      llvm::errs() << "Warning: No profile data found in " << profilePath
                   << "\n";
    }
    // Before the floor below, which would hide the exact zeros.
    if (flags::AggressiveDCE)
      for (const auto &kv : profileMap) {
        const ProfileInfo &p = kv.second;
        if ((p.sumAbsGrad >= 0.0 ? p.sumAbsGrad : p.sumGrad) == 0.0)
          zeroGradIdx.insert(kv.first);
      }
    // Degenerate-adjoint guard, before anything reads a gradient: the floor is
    // written back into the profile records so the node table, the Herbie
    // candidate pricing, the precision-tuning per-output costs and the
    // scalar-reduction matmul gradD cells all see one consistent weight.
    applyGradientFloor(profileMap, F.getName());
    if (profileHeader.canonicalHash.empty()) {
      llvm::errs() << "Warning: " << profilePath
                   << " has no CanonicalHash field; the canonical form of "
                   << F.getName() << " is not verified against the profile\n";
    } else if (profileHeader.canonicalHash != canonicalHash) {
      report_fatal_error(
          Twine("Poseidon: the profile of ") + F.getName() +
          " was recorded against a different canonical form (profile "
          "CanonicalHash " +
          profileHeader.canonicalHash + ", this compile " + canonicalHash +
          "). Its slot indices do not name the same instructions; re-run "
          "profile generation.");
    }
    if (flags::Print && profileHeader.launchCount > 0) {
      llvm::errs() << "[poseidon] profile header for " << F.getName()
                   << ": MaxBlockDims=(" << profileHeader.maxBlockDim[0] << ","
                   << profileHeader.maxBlockDim[1] << ","
                   << profileHeader.maxBlockDim[2]
                   << ") LaunchCount=" << profileHeader.launchCount << "\n";
    }
  }

  auto &abstractMatmuls = st.abstractMatmuls;

  // These stash a ScalarEvolution* (and const SCEV* pointers it owns) into each
  // ScalarLoopHandle for the materializer's SCEVExpander, dereferenced during
  // materialization, so they must outlive it and live in `st`. Declared SE-last
  // so it is destroyed first (it references the others).
  auto &raiseDT = st.raiseDT;
  auto &raiseLI = st.raiseLI;
  auto &raiseAC = st.raiseAC;
  auto &raiseTLII = st.raiseTLII;
  auto &raiseTLI = st.raiseTLI;
  auto &raiseSE = st.raiseSE;
  if (isGPUMode(F) && flags::RaiseWMMA) {
    raiseDT.emplace(F);
    raiseLI.emplace(*raiseDT);
    raiseAC.emplace(F);
    raiseTLII.emplace(Triple(F.getParent()->getTargetTriple()));
    raiseTLI.emplace(*raiseTLII, &F);
    raiseSE.emplace(F, *raiseTLI, *raiseAC, *raiseDT, *raiseLI);
    findScalarLoopMatmuls(F, *raiseSE, *raiseLI, profileHeader,
                          abstractMatmuls);
    // Runtime-shape dense GEMMs (finite-element partial-assembly reduces). Runs
    // after the constant-trip recognizer and skips whatever that one claimed,
    // so previously recognized sites keep their exact candidate sets.
    findHostGemmLoopNests(F, *raiseSE, *raiseLI, profileHeader, profileMap,
                          abstractMatmuls);
  }

  if (flags::Print && !abstractMatmuls.empty()) {
    llvm::errs() << "[poseidon] Found " << abstractMatmuls.size()
                 << " AbstractMatmul(s) for " << F.getName() << "\n";
    auto dimStr = [](unsigned d) -> std::string {
      return d == ~0u ? std::string("?") : std::to_string(d);
    };
    for (const auto &m : abstractMatmuls) {
      llvm::errs() << "  Matmul[" << m.id << "]: " << dimStr(m.M) << "x"
                   << dimStr(m.N) << "x" << dimStr(m.K)
                   << " a=" << fpKindName(m.aType)
                   << " b=" << fpKindName(m.bType)
                   << " acc=" << fpKindName(m.accType)
                   << " d=" << fpKindName(m.dType) << " origin="
                   << (m.origin == AbstractMatmul::Origin::HostGemmLoopNest
                           ? "HostGemmLoopNest"
                           : "ScalarLoopReduction")
                   << "\n";
      if (m.origin == AbstractMatmul::Origin::HostGemmLoopNest)
        llvm::errs() << "    (profile-scale shape; runtime descriptor printed "
                        "by the [hostgemm] recognizer above)\n";
      else
        llvm::errs() << "    (profile dump skipped for ScalarLoopReduction)\n";
    }
  }

  int symbolCounter = 0;
  auto getNextSymbol = [&symbolCounter]() -> std::string {
    return "v" + std::to_string(symbolCounter++);
  };

  auto &valueToNodeMap = st.valueToNodeMap;
  auto &symbolToValueMap = st.symbolToValueMap;

  llvm::errs() << "[poseidon] Starting Floodfill for " << F.getName() << "\n";

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isOptimizable(I)) {
        valueToNodeMap[&I] = std::make_shared<FPLLValue>(&I, "__nh", "__nh");
        if (flags::Print)
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
            if (flags::Print)
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
            if (flags::Print)
              llvm::errs() << "Registered FPNode for " << dtype
                           << " constant: " << value << "\n";
          } else if (auto CI = dyn_cast<ConstantInt>(operand)) {
            // e.g., powi intrinsic has a constant int as its exponent
            double exponent = static_cast<double>(CI->getSExtValue());
            std::string dtype = "f64";
            std::string doubleStr = std::to_string(exponent);
            valueToNodeMap[operand] =
                std::make_shared<FPConst>(doubleStr.c_str(), dtype);
            if (flags::Print)
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
            if (flags::Print)
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
  auto &subgraphs = st.subgraphs;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!isOptimizable(I)) {
        if (flags::Print)
          llvm::errs() << "Skipping non-isOptimizable instruction: " << I
                       << "\n";
        continue;
      }

      if (processed.contains(&I)) {
        if (flags::Print)
          llvm::errs() << "Skipping already seen instruction: " << I << "\n";
        continue;
      }

      if (flags::Print)
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
          if (flags::Print)
            llvm::errs() << "Skipping already seen instruction: " << *I2
                         << "\n";
          continue;
        }

        assert(!processed.contains(cur));

        if (flags::Print)
          llvm::errs() << "Insert to operation_seen and processed: " << *I2
                       << "\n";
        operation_seen.insert(I2);
        processed.insert(cur);

        auto operands =
            isa<CallInst>(I2) ? cast<CallInst>(I2)->args() : I2->operands();

        for (const auto &operand : operands) {
          if (!isOptimizable(*operand)) {
            if (flags::Print)
              llvm::errs() << "Non-isOptimizable input found: " << *operand
                           << "\n";

            if (!isa<ConstantFP>(operand))
              input_seen.insert(operand);
          } else {
            if (flags::Print)
              llvm::errs() << "Adding operand to todo list: " << *operand
                           << "\n";
            todo.push_back(operand);
          }
        }

        for (auto U : I2->users()) {
          if (auto I3 = dyn_cast<Instruction>(U)) {
            if (!isOptimizable(*I3)) {
              if (flags::Print)
                llvm::errs() << "Output instruction found: " << *I2 << "\n";
              output_seen.insert(I2);
            } else {
              if (flags::Print)
                llvm::errs() << "Adding user to todo list: " << *I3 << "\n";
              todo.push_back(I3);
            }
          }
        }
      }

      if (!operation_seen.empty()) {
        if (flags::Print) {
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
          if (flags::Print)
            llvm::errs() << "Skipping trivial subgraph\n";
          continue;
        }

        subgraphs.emplace_back(input_seen, output_seen, operation_seen);
      }
    }
  }

  if (flags::Print) {
    llvm::errs() << "[poseidon] Found " << subgraphs.size()
                 << " initial subgraphs in " << F.getName() << "\n";
  }

  // Profile read must happen before aggressive DCE as it requires gradients
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
            node->updateBounds(profileInfo.minRes, profileInfo.maxRes);

            if (flags::Print) {
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
                if (flags::Print) {
                  llvm::errs() << "Range of " << *operand << " is ["
                               << operandNode->getLowerBound() << ", "
                               << operandNode->getUpperBound() << "]\n";
                }
              }
            }
          } else {
            if (!flags::LooseCoverage) {
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
            if (flags::Print)
              llvm::errs() << "Sensitivity/gradient/executions of " << *op
                           << " not found in the log; using 0 for all\n";
          }
        }
      }
    }
  }

  if (flags::AggressiveDCE)
    aggressiveDCE(F, subgraphs, valueToNodeMap, profileMap, zeroGradIdx,
                  abstractMatmuls);

  splitSubgraphs(subgraphs);

  if (flags::Print) {
    llvm::errs() << "[poseidon] After splitting, have " << subgraphs.size()
                 << " subgraphs in " << F.getName() << "\n";
  }

  if (flags::Print) {
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

  if (subgraphs.empty() && abstractMatmuls.empty()) {
    if (flags::Print)
      llvm::errs() << "No subgraphs or matmuls found\n";
    return false;
  }

  auto &COs = st.COs;
  auto &CSs = st.CSs;

  int subgraphCounter = 0;

  for (auto &subgraph : subgraphs) {
    assert(subgraph.inputs.size() > 0 && "No inputs found for subgraph");

    if (flags::EnableHerbie) {
      for (const auto &input : subgraph.inputs) {
        auto node = valueToNodeMap[input];
        if (node->op == "__const") {
          continue;
        }

        if (!node->hasSymbol()) {
          node->symbol = getNextSymbol();
        }
        symbolToValueMap[node->symbol] = input;
        if (flags::Print)
          llvm::errs() << "assigning symbol: " << node->symbol << " to "
                       << *input << "\n";
      }

      std::vector<std::string> herbieInputs;
      std::vector<CandidateOutput> newCOs;

      assert(subgraph.outputs.size() > 0 && "No outputs found for subgraph");
      for (auto &output : subgraph.outputs) {
        double grad = valueToNodeMap[output]->grad;
        unsigned executions = valueToNodeMap[output]->executions;

        if (grad == 0.) {
          llvm::errs() << "Skipping zero gradient instruction: " << *output
                       << "\n";
          continue;
        }

        std::string expr = valueToNodeMap[output]->toFullExpression(
            valueToNodeMap, subgraph.inputs);

        if (expr.length() > flags::MaxExprLength) {
          llvm::errs() << "WARNING: Skipping Herbie optimization for "
                       << *output << " since expression length "
                       << expr.length() << " exceeds limit of "
                       << flags::MaxExprLength << "\n";
          continue;
        }

        auto parenCount = std::count(expr.begin(), expr.end(), '(');
        assert(parenCount > 0);
        if (parenCount == 1) {
          if (flags::Print)
            llvm::errs() << "Skipping Herbie for simple expression: " << expr
                         << "\n";
          continue;
        }

        SmallSet<std::string, 8> args;
        getUniqueArgs(expr, args);

        std::string precondition =
            getPrecondition(args, valueToNodeMap, symbolToValueMap);

        std::string argStr;
        for (const auto &arg : args) {
          if (!argStr.empty())
            argStr += " ";
          argStr += arg;
        }

        for (const char *prec : {"binary64", "binary32"}) {
          std::string properties = ":herbie-conversions ([binary64 binary32])";
          properties += std::string(" :precision ") + prec;
          properties += " :pre " + precondition;

          CandidateOutput CO(subgraph, output, expr, grad, executions);
          properties += " :name \"" + std::to_string(newCOs.size()) + "\"";

          std::string herbieInput =
              "(FPCore (" + argStr + ") " + properties + " " + expr + ")";
          if (flags::Print)
            llvm::errs() << "Herbie input:\n" << herbieInput << "\n";

          herbieInputs.push_back(herbieInput);
          newCOs.push_back(CO);
        }
      }

      if (!herbieInputs.empty()) {
        if (!improveViaHerbie(herbieInputs, newCOs, F.getParent(),
                              valueToNodeMap, symbolToValueMap, subgraphCounter,
                              F.getName(), canonicalHash)) {
          if (flags::Print)
            llvm::errs() << "Failed to optimize expressions using Herbie!\n";
        }

        COs.insert(COs.end(), newCOs.begin(), newCOs.end());
      }
    }

    if (flags::EnablePT) {
      auto *o0 = subgraph.outputs[0];
      const unsigned o0Exec = valueToNodeMap[o0]->executions;

      CandidateSubgraph CS(subgraph);
      CS.executions = o0Exec;

      // Synthetic output-boundary casts execute at most once per thread per
      // launch, while the body executes CS.executions times; derive the scale
      // from the profile header (1 for elementwise subgraphs).
      if (CS.executions > 0 && profileHeader.launchCount > 0) {
        double thr = 1.0;
        for (int d = 0; d < 3; ++d)
          thr *= std::max<uint32_t>(1, profileHeader.maxBlockDim[d]);
        for (int d = 0; d < 3; ++d)
          thr *= std::max<uint32_t>(1, profileHeader.maxGridDim[d]);
        subgraph.outBoundaryFreqScale =
            std::min(1.0, thr * (double)profileHeader.launchCount /
                              (double)CS.executions);
      }

      // Keyed by THIS function: a cached table for an earlier site in the same
      // compilation unit carries no deltas for these candidates, so skipping
      // on the file's mere existence would leave them unevaluated.
      bool skipEvaluation = dpCacheHasFunction(F.getName());

      const auto &PTFuncs = getPTFuncs();
      SetVector<FPLLValue *> funcsSet, allSet;
      for (auto *I : subgraph.operations) {
        assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
               "Corrupted FPNode for original instructions");
        auto node = cast<FPLLValue>(valueToNodeMap[I].get());
        allSet.insert(node);
        if (PTFuncs.count(node->op) != 0) {
          funcsSet.insert(node);
          llvm::errs() << "[poseidon] PT Function identified: " << *I << "\n";
        }
      }
      SmallVector<FPLLValue *> sortedFuncs(funcsSet.begin(), funcsSet.end());
      SmallVector<FPLLValue *> sortedAllOps(allSet.begin(), allSet.end());
      auto bySens = [](const auto &a, const auto &b) {
        return a->sens < b->sens;
      };
      llvm::sort(sortedFuncs, bySens);
      llvm::sort(sortedAllOps, bySens);

      const bool gpuMode = isGPUMode(F);
      static const std::unordered_set<std::string> kEmptyScalars;
      const std::unordered_set<std::string> &hwScalar =
          gpuMode ? getScalarTypes() : kEmptyScalars;
      PrecisionChangeType curr =
          getPrecisionChangeType(subgraph.outputs[0]->getType());

      auto emitCandidate =
          [&](SmallVectorImpl<std::pair<PrecisionChangeType,
                                        SetVector<FPLLValue *>>> &assignment,
              std::string desc) {
            SmallVector<PrecisionChange, 3> changes;
            for (auto &kv : assignment) {
              if (kv.first != curr && !kv.second.empty())
                changes.emplace_back(kv.second, curr, kv.first);
            }
            if (changes.empty())
              return;
            PTCandidate cand{std::move(changes), std::move(desc)};
            if (!skipEvaluation)
              cand.CompCost = getCompCost(subgraph, cand);
            CS.candidates.push_back(std::move(cand));
          };

      auto sweepTwoTier = [&](TierPair tp, ArrayRef<FPLLValue *> sortedAsc,
                              StringRef label) {
        const size_t N = sortedAsc.size();
        const int step = std::max(5, flags::TwoTierStep.getValue());
        size_t prev = N + 1;
        for (int pct = 0; pct <= 100 - step; pct += step) {
          size_t k = N * pct / 100;
          if (k == prev)
            continue;
          prev = k;

          SetVector<FPLLValue *> hiOps(sortedAsc.end() - k, sortedAsc.end());
          SetVector<FPLLValue *> loOps(sortedAsc.begin(), sortedAsc.end() - k);

          if (flags::Print) {
            llvm::errs() << "Created " << label
                         << " two-tier PT candidate: " << fmtPrecPct(tp.hi, pct)
                         << " + " << fmtPrecPct(tp.lo, 100 - pct) << " (N=" << N
                         << ")\n";
          }
          std::string desc = label.str();
          if (!desc.empty())
            desc += " ";
          desc += fmtPrecPct(tp.hi, pct);
          desc += " + ";
          desc += fmtPrecPct(tp.lo, 100 - pct);

          SmallVector<std::pair<PrecisionChangeType, SetVector<FPLLValue *>>, 2>
              assignment;
          assignment.emplace_back(tp.hi, std::move(hiOps));
          assignment.emplace_back(tp.lo, std::move(loOps));
          emitCandidate(assignment, std::move(desc));
        }
      };

      auto sweepThreeTier = [&](TierTriple tr, ArrayRef<FPLLValue *> sortedAsc,
                                StringRef label) {
        const size_t N = sortedAsc.size();
        const int step = kThreeTierStep;
        for (int pctHi = step; pctHi <= 100 - 2 * step; pctHi += step) {
          for (int pctHiMid = pctHi + step; pctHiMid <= 100 - step;
               pctHiMid += step) {
            size_t k0 = N * pctHi / 100;
            size_t k1 = N * pctHiMid / 100;
            if (k0 == 0 || k0 >= k1 || k1 >= N)
              continue;

            SetVector<FPLLValue *> hiOps(sortedAsc.end() - k0, sortedAsc.end());
            SetVector<FPLLValue *> midOps(sortedAsc.end() - k1,
                                          sortedAsc.end() - k0);
            SetVector<FPLLValue *> loOps(sortedAsc.begin(),
                                         sortedAsc.end() - k1);

            if (flags::Print) {
              llvm::errs() << "Created " << label
                           << " three-tier PT candidate: "
                           << fmtPrecPct(tr.hi, pctHi) << " + "
                           << fmtPrecPct(tr.mid, pctHiMid - pctHi) << " + "
                           << fmtPrecPct(tr.lo, 100 - pctHiMid) << " (N=" << N
                           << ")\n";
            }
            std::string desc = label.str();
            if (!desc.empty())
              desc += " ";
            desc += fmtPrecPct(tr.hi, pctHi);
            desc += " + ";
            desc += fmtPrecPct(tr.mid, pctHiMid - pctHi);
            desc += " + ";
            desc += fmtPrecPct(tr.lo, 100 - pctHiMid);

            SmallVector<std::pair<PrecisionChangeType, SetVector<FPLLValue *>>,
                        3>
                assignment;
            assignment.emplace_back(tr.hi, std::move(hiOps));
            assignment.emplace_back(tr.mid, std::move(midOps));
            assignment.emplace_back(tr.lo, std::move(loOps));
            emitCandidate(assignment, std::move(desc));
          }
        }
      };

      for (TierPair tp : kCanonicalPairs) {
        if (!tierPairAllowed(tp, gpuMode, hwScalar))
          continue;
        sweepTwoTier(tp, sortedAllOps, "All");
        if (!sortedFuncs.empty())
          sweepTwoTier(tp, sortedFuncs, "Funcs");
      }

      if (flags::EnableThreeTier) {
        for (TierTriple tr : kCanonicalTriples) {
          if (!tierTripleAllowed(tr, gpuMode, hwScalar))
            continue;
          sweepThreeTier(tr, sortedAllOps, "All");
        }
      }

      if (!skipEvaluation) {
        setUnifiedAccuracyCost(CS, valueToNodeMap, symbolToValueMap);
      }

      // Zero candidates for a nonempty subgraph means every tier was filtered
      // out (a configuration problem), which must not pass as a no-op frontier.
      if (CS.candidates.empty() && !subgraph.operations.empty()) {
        llvm::errs() << "[poseidon] WARNING: subgraph with "
                     << subgraph.operations.size() << " FP operations in "
                     << F.getName()
                     << " synthesized ZERO precision-tuning candidates (all "
                        "tier combinations filtered); the solve cannot "
                        "propose any rewrite here. Check the cost model's "
                        "scalar_types header.\n";
        if (flags::StrictMode)
          report_fatal_error(
              "Poseidon strict mode: nonempty FP subgraph synthesized zero "
              "candidates with precision tuning enabled");
      }

      CSs.push_back(std::move(CS));
    }
    llvm::errs() << "##### Finished synthesizing candidates for "
                 << ++subgraphCounter << " of " << subgraphs.size()
                 << " subgraphs! #####\n";
  }

  auto &CMs = st.CMs;
  generateMatmulCandidates(abstractMatmuls, profileMap, st.confidence,
                           st.sampleLogBits, CMs);

  // Put this site's error on the application's scale. kappa is the measured
  // response of the declared quantity of interest to computing this site with a
  // relative error eps on every operation, and the profile's own first-order
  // prediction for the same noise is eps times the site's total sensitivity, in
  // the units the accuracy costs are already in; dividing the two cancels the
  // within-site propagation that both carry and leaves the downstream condition
  // per unit of accuracy cost. A site whose probe run diverged gets a large
  // finite factor instead, so no candidate of it is ever traded against
  // precision.
  {
    double siteSumSens = 0.0;
    for (const auto &kv : profileMap)
      siteSumSens += kv.second.sumSens;

    double factor = 1.0;
    if (profileHeader.hasKappa) {
      double sumSens = siteSumSens;
      if (sumSens == 0.0)
        report_fatal_error(
            Twine("Poseidon: the profile of ") + F.getName() +
            " carries a measured condition number but a zero total "
            "sensitivity, "
            "so the accuracy costs cannot be put on the scale the condition "
            "number was measured against; re-run profile generation.");
      factor = profileHeader.kappaMark == "diverged"
                   ? 1e30
                   : profileHeader.kappa / sumSens;
      llvm::errs() << "[poseidon] site " << F.getName()
                   << ": kappa=" << profileHeader.kappa << " ("
                   << profileHeader.kappaMark << ") sumsens=" << sumSens
                   << " factor=" << factor << "\n";
    }
    // The accuracy costs below are scaled by `factor`, so the level a tolerance
    // is compared against has to be scaled the same way; with a kappa this
    // reduces to kappa itself, without one to the site's total sensitivity.
    st.accScale = factor * siteSumSens;
    if (factor != 1.0) {
      for (auto &co : COs) {
        co.initialAccCost *= factor;
        for (auto &cand : co.candidates)
          cand.accuracyCost *= factor;
      }
      for (auto &cs : CSs) {
        cs.initialAccCost *= factor;
        for (auto &[node, cost] : cs.perOutputInitialAccCost)
          cost *= factor;
        for (auto &cand : cs.candidates) {
          cand.accuracyCost *= factor;
          for (auto &[node, cost] : cand.perOutputAccCost)
            cost *= factor;
        }
      }
      for (auto &cm : CMs) {
        cm.initialAccCost *= factor;
        for (auto &cand : cm.candidates)
          cand.accuracyCost *= factor;
      }
    }
  }

  if (flags::Print) {
    if (flags::EnableHerbie) {
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
    if (flags::EnablePT) {
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

  if (flags::Print) {
    for (const auto &cm : CMs) {
      llvm::errs() << "Matmul[" << cm.matmul->id
                   << "] candidates (initial cost=" << cm.initialCompCost
                   << ", executions=" << cm.executions
                   << ", globalM=" << cm.matmul->globalM
                   << ", globalN=" << cm.matmul->globalN
                   << ", globalK=" << cm.matmul->globalK
                   << ", K=" << cm.matmul->K
                   << ", gridCTAs=" << cm.matmul->gridCTAs << "):\n";
      if (cm.candidates.empty()) {
        llvm::errs() << "  (no candidates emitted)\n";
        continue;
      }
      for (size_t i = 0; i < cm.candidates.size(); ++i) {
        const auto &opt = cm.candidates[i];
        llvm::errs() << "  " << matmulOptionLabel(opt)
                     << ": Δcost=" << cm.getCompCostDelta(i)
                     << " ΔaccCost=" << cm.getAccCostDelta(i) << "\n";
      }
    }
  }

  return true;
}

// Split out of fpOptimize so the joint solver can collect across functions,
// solve once, then materialize each function with its own steps.
bool materializeFPSolution(Function &F, FunctionFPState &st,
                           ArrayRef<SolutionStep> steps) {
  auto &valueToNodeMap = st.valueToNodeMap;
  auto &symbolToValueMap = st.symbolToValueMap;
  auto &subgraphs = st.subgraphs;
  bool changed = applySolution(steps, valueToNodeMap, symbolToValueMap);

  llvm::errs() << "[poseidon] Finished optimizing " << F.getName() << "\n";

  if (changed) {
    for (auto &subgraph : subgraphs) {
      if (subgraph.outputs_rewritten != subgraph.outputs.size()) {
        if (flags::Print)
          llvm::errs() << "Skip erasing a subgraph: only rewrote "
                       << subgraph.outputs_rewritten << " of "
                       << subgraph.outputs.size() << " outputs\n";
        continue; // Intermediate operations cannot be erased safely
      }
      for (auto *I : subgraph.operations) {
        if (flags::Print)
          llvm::errs() << "Erasing: " << *I << "\n";
        if (!I->use_empty()) {
          I->replaceAllUsesWith(UndefValue::get(I->getType()));
        }
        I->eraseFromParent();
      }
    }

    llvm::errs() << "[poseidon] Finished cleaning up " << F.getName() << "\n";
  }

  if (changed)
    applyStagingNarrowing(F, /*announce=*/true);
  if (changed && demoteFPCastPHIs(F)) {
    if (flags::Print)
      llvm::errs() << "[poseidon] demoted fpcast-sandwiched FP64 PHIs in "
                   << F.getName() << "\n";
  }

  simplifyFunction(F, OptimizationLevel::O3);

  // Mark the Poseidon-transformed function `alwaysinline` so the downstream
  // inliner folds it back into the caller. The source-level `noinline` on the
  // user's matmul body is needed UPSTREAM so the function survives as a
  // distinct symbol Poseidon can clone, but the call boundary on the
  // transformed clone costs ~1.5x wallclock on GPU (NVPTX passes pointer args
  // via per-thread R-regs across calls, forcing R2UR broadcasts and tanking
  // uniformity analysis).
  //
  // The two `remove` calls are required by the IR verifier: `alwaysinline`
  // conflicts with `noinline` / `optimizenone`.
  if (changed) {
    F.removeFnAttr(Attribute::NoInline);
    F.removeFnAttr(Attribute::OptimizeNone);
    F.addFnAttr(Attribute::AlwaysInline);
  }

  if (flags::Print) {
    llvm::errs() << "[poseidon] Finished Optimization\n";
    F.print(llvm::errs());
  }

  return changed;
}

bool fpOptimize(Function &F, double errorTol, double siteConfidence) {
  // Matrix-product selection is only trustworthy with dynamic-range-aware
  // sampling (uniform [min,max] under-prices fixed-point Ozaki on
  // wide-dynamic-range matrices), so a tolerance -- the site's or the flag's --
  // defaults the sampling knob to a wide log-uniform range; an explicit
  // -poseidon-sample-log-bits still overrides. Read only by the matmul accuracy
  // model, so a site with no matrix product is unaffected.
  unsigned sampleLogBits = flags::SampleLogBits;
  if ((flags::Tau > 0.0 || errorTol > 0.0) && sampleLogBits == 0)
    sampleLogBits = 40;

  // The confidence level the matrix-product accuracy model reads its domain
  // error off, under the same precedence as the tolerance: what the site wrote
  // wins over the whole-build flag.
  const double confidence =
      siteConfidence > 0.0 ? siteConfidence : (double)flags::Confidence;
  if (siteConfidence > 0.0 && flags::Confidence.getNumOccurrences())
    llvm::errs() << "[poseidon] " << F.getName()
                 << ": the site's own confidence " << siteConfidence
                 << " overrides -poseidon-confidence=" << flags::Confidence
                 << "\n";
  if (flags::Print)
    llvm::errs() << "[poseidon] " << F.getName()
                 << ": accuracy target confidence " << confidence
                 << (siteConfidence > 0.0 ? " (site)"
                                          : " (-poseidon-confidence)")
                 << "\n";

  FunctionFPState st;
  if (!collectFPCandidates(F, errorTol, confidence, sampleLogBits, st))
    return false;

  // A tolerance written AT THE SITE is the user interface; the global flag is a
  // whole-build override for a source that carries none. When both are given
  // the site value wins, because it is the one the source asked for.
  double budget = errorTol > 0.0 ? errorTol : (double)flags::Tau;
  if (errorTol > 0.0 && flags::Tau > 0.0)
    llvm::errs() << "[poseidon] " << F.getName()
                 << ": the site's own accuracy target " << errorTol
                 << " overrides -poseidon-tau=" << flags::Tau << "\n";

  SmallVector<SolutionStep> steps;
  if (!flags::ApplyRewrites.empty()) {
    steps = parseManualRewrites(flags::ApplyRewrites, st.COs, st.CSs, st.CMs);
  } else if (errorTol > 0.0 && !st.CMs.empty()) {
    // A site tolerance is a RELATIVE error target, and for a matrix product
    // that is what errorBudgetSelector compares against, exactly as the global
    // flag does. Whatever elementwise work the products do not cover is then
    // solved by the normalized DP under the same number.
    steps = errorBudgetSelector(st.CMs, budget, confidence);
    if (!st.COs.empty() || !st.CSs.empty()) {
      SmallVector<CandidateMatmul, 4> noCMs;
      SmallVector<SolutionStep> elem =
          accuracyDPSolver(F, st.COs, st.CSs, noCMs, st.valueToNodeMap,
                           st.symbolToValueMap, budget, st.accScale);
      for (const SolutionStep &e : elem) {
        // The two selectors ran independently, so the footprint rule the DP
        // applies internally is applied between their results here. Never
        // silent: dropping a rewrite the DP priced changes what the site emits.
        if (stepConflictsWithMatmulSteps(e, steps)) {
          llvm::errs() << "[poseidon] " << F.getName()
                       << ": an elementwise rewrite the DP selected edits "
                          "instructions a selected matrix-product raise "
                          "consumes; keeping the raise and dropping the "
                          "rewrite\n";
          continue;
        }
        steps.push_back(e);
      }
    }
  } else if (flags::Tau > 0.0 && !st.CMs.empty()) {
    steps = errorBudgetSelector(st.CMs, budget, confidence);
  } else {
    steps = accuracyDPSolver(F, st.COs, st.CSs, st.CMs, st.valueToNodeMap,
                             st.symbolToValueMap, budget, st.accScale);
  }
  return materializeFPSolution(F, st, steps);
}

bool solveJointly(Module &M, FunctionAnalysisManager &FAM) {
  SmallVector<Function *, 4> marked;
  for (Function &F : M)
    if (!F.empty() && F.hasFnAttribute("poseidon-joint-errtol"))
      marked.push_back(&F);
  if (marked.empty())
    return false;

  llvm::errs() << "Poseidon JOINT: collecting candidates from " << marked.size()
               << " marked function(s)\n";

  std::vector<std::unique_ptr<FunctionFPState>> states;
  double jointErrTol = 0.0;
  for (Function *F : marked) {
    double errTol = 0.0;
    F->getFnAttribute("poseidon-joint-errtol")
        .getValueAsString()
        .getAsDouble(errTol);
    F->removeFnAttr("poseidon-joint-errtol");
    if (errTol > jointErrTol)
      jointErrTol = errTol;

    // Each site keeps the confidence it was annotated with: the joint DP shares
    // one BUDGET, not one accuracy model.
    double confidence = (double)flags::Confidence;
    if (F->hasFnAttribute("poseidon-joint-confidence")) {
      F->getFnAttribute("poseidon-joint-confidence")
          .getValueAsString()
          .getAsDouble(confidence);
      F->removeFnAttr("poseidon-joint-confidence");
    }

    auto st = std::make_unique<FunctionFPState>();
    if (collectFPCandidates(*F, errTol, confidence, flags::SampleLogBits,
                            *st)) {
      states.push_back(std::move(st));
    } else {
      llvm::errs() << "Poseidon JOINT: nothing to optimize in " << F->getName()
                   << "\n";
      redirectNoopSite(F);
    }
  }
  if (states.empty())
    return false;

  SmallVector<FunctionFPState *, 4> raw;
  for (auto &s : states)
    raw.push_back(s.get());

  SmallVector<SolutionStep> steps;
  if (!flags::ApplyRewrites.empty()) {
    for (auto *st : raw) {
      auto s =
          parseManualRewrites(flags::ApplyRewrites, st->COs, st->CSs, st->CMs);
      steps.append(s.begin(), s.end());
    }
  } else {
    steps = jointAccuracyDPSolver(raw, jointErrTol);
  }

  std::unordered_map<const void *, FunctionFPState *> owner;
  for (auto *st : raw) {
    for (auto &CO : st->COs)
      owner[&CO] = st;
    for (auto &CS : st->CSs)
      owner[&CS] = st;
    for (auto &CM : st->CMs)
      owner[&CM] = st;
  }
  std::unordered_map<FunctionFPState *, SmallVector<SolutionStep>> perState;
  for (auto &step : steps) {
    const void *p =
        std::visit([](auto *ptr) -> const void * { return ptr; }, step.item);
    auto it = owner.find(p);
    assert(it != owner.end() && "joint step references an unknown candidate");
    perState[it->second].push_back(step);
  }

  // A site whose step list is empty gets its wrapper call restored to the
  // original body: the preprocessed clone is not guaranteed to lower
  // bit-identically, and an unrewritten site must be numerically transparent.
  bool changed = false;
  for (auto *st : raw) {
    bool ch = materializeFPSolution(*st->F, *st, perState[st]);
    if (!ch)
      redirectNoopSite(st->F);
    changed |= ch;
  }
  // The joint materialize has stashed the host-dispatch notes; emit the
  // deferred descriptors now.
  flushPendingGemmDispatches();
  return changed;
}

} // namespace poseidon

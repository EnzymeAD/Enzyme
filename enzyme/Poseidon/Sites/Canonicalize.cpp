//===- Canonicalize.cpp - the canonical form of a site --------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Canonicalize.h"

#include "Enzyme/Utils.h"
#include "SiteIdentity.h"
#include "Support/Utils.h"
#include "Targets/MathLibrary.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

namespace poseidon {

static bool isOptimizableOperation(const Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::FNeg:
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    return I->getType()->isFloatTy() || I->getType()->isDoubleTy();
  case Instruction::Call: {
    const Function *Callee = cast<CallInst>(I)->getCalledFunction();
    if (!Callee)
      return false;
    // The libdevice and ocml entry points Poseidon handles.
    if (deviceMathName(Callee->getName()))
      return true;
    if (!I->getType()->isFloatTy() && !I->getType()->isDoubleTy())
      return false;
    switch (Callee->getIntrinsicID()) {
    case Intrinsic::sin:
    case Intrinsic::cos:
    case Intrinsic::tan:
    case Intrinsic::asin:
    case Intrinsic::acos:
    case Intrinsic::atan:
    case Intrinsic::atan2:
    case Intrinsic::sinh:
    case Intrinsic::cosh:
    case Intrinsic::tanh:
    case Intrinsic::exp:
    case Intrinsic::log:
    case Intrinsic::sqrt:
    case Intrinsic::pow:
    case Intrinsic::powi:
    case Intrinsic::fabs:
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
    case Intrinsic::maxnum:
    case Intrinsic::minnum:
    case Intrinsic::ceil:
    case Intrinsic::floor:
    case Intrinsic::exp2:
    case Intrinsic::log10:
    case Intrinsic::log2:
    case Intrinsic::rint:
    case Intrinsic::round:
    case Intrinsic::trunc:
    case Intrinsic::copysign:
      return true;
    case Intrinsic::not_intrinsic:
      return isOptimizableLibmCall(Callee->getName());
    default:
      return false;
    }
  }
  default:
    return false;
  }
}

/// An undef or poison operand has no value to sample and may take a different
/// one at each use; a constant expression has no literal.
static bool isModelledOperand(const Value *V) {
  Type *T = V->getType();
  if (isa<Argument, ConstantFP>(V))
    return T->isFloatTy() || T->isDoubleTy();
  if (isa<Instruction>(V))
    return T->isFloatingPointTy() || T->isIntegerTy();
  return isa<ConstantInt>(V);
}

bool isOptimizable(const llvm::Value &V) {
  const auto *I = dyn_cast<Instruction>(&V);
  if (!I || !isOptimizableOperation(I))
    return false;
  const auto *CI = dyn_cast<CallInst>(I);
  return all_of(CI ? CI->args() : I->operands(),
                [](const Use &U) { return isModelledOperand(U.get()); });
}

std::optional<size_t> slotOf(const Instruction *I) {
  MDNode *md = I ? I->getMetadata(kSlotMD) : nullptr;
  if (!md)
    return std::nullopt;
  return mdconst::extract<ConstantInt>(md->getOperand(0))->getZExtValue();
}

size_t requireSlot(const Instruction *I) {
  std::optional<size_t> slot = slotOf(I);
  if (!slot)
    report_fatal_error("Poseidon: an instruction of a canonical clone has no "
                       "slot");
  return *slot;
}

void setSlotMetadata(Function &F) {
  // A slot is the rank among optimizable instructions, which the
  // profile-generate and profile-use compiles share; the instrumentation shifts
  // absolute positions.
  size_t slotIdx = 0;
  for (Instruction &I : instructions(F)) {
    if (isOptimizable(I)) {
      I.setMetadata("enzyme_active", MDNode::get(I.getContext(), {}));
      I.setMetadata(
          kSlotMD,
          MDNode::get(I.getContext(),
                      {ConstantAsMetadata::get(ConstantInt::get(
                          Type::getInt64Ty(I.getContext()), slotIdx))}));
      ++slotIdx;
    }
  }
}

void addCanonicalPreInline(FunctionPassManager &FPM, OptimizationLevel Level) {
  if (Level == OptimizationLevel::O0)
    return;
  FPM.addPass(Float2IntPass());
  FPM.addPass(LowerConstantIntrinsicsPass());
  LoopPassManager LPM;
  LPM.addPass(LoopRotatePass(/*EnableHeaderDuplication=*/true, false));
  LPM.addPass(LoopDeletionPass());
  LPM.addPass(LoopFullUnrollPass());
  FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
}

void addCanonicalPostInline(FunctionPassManager &FPM) {
  FPM.addPass(GVNPass());
  FPM.addPass(SROAPass(SROAOptions::PreserveCFG));
}

std::string canonicalFormHash(const Function &F) {
  std::string form;
  raw_string_ostream os(form);
  size_t slot = 0;
  for (const Instruction &I : instructions(F)) {
    if (!isOptimizable(I))
      continue;
    os << slot++ << '|' << I.getOpcodeName() << '|';
    I.getType()->print(os);
    if (const auto *CI = dyn_cast<CallInst>(&I)) {
      for (const Value *arg : CI->args()) {
        os << '|';
        arg->getType()->print(os);
      }
      const Function *callee = CI->getCalledFunction();
      os << '|' << (callee ? callee->getName() : StringRef("<indirect>"));
    } else {
      for (const Use &op : I.operands()) {
        os << '|';
        op->getType()->print(os);
      }
    }
    os << '\n';
  }
  // FNV-1a, 64 bits.
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : form) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return utohexstr(h, /*LowerCase=*/true, /*Width=*/16);
}

namespace {

struct CanonicalizeAnalyses {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;

  explicit CanonicalizeAnalyses(LLVMContext &Ctx) {
    FAM.registerPass([] { return TypeBasedAA(); });
    FAM.registerPass([] { return BasicAA(); });
    MAM.registerPass([] { return GlobalsAA(); });
    MAM.registerPass([] { return CallGraphAnalysis(); });
    FAM.registerPass([] { return ScopedNoAliasAA(); });

    MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
    LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(FAM); });
    FAM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });

    FAM.registerPass([] {
      AAManager AM;
      AM.registerFunctionAnalysis<BasicAA>();
      AM.registerFunctionAnalysis<TypeBasedAA>();
      AM.registerModuleAnalysis<GlobalsAA>();
      AM.registerFunctionAnalysis<ScopedNoAliasAA>();
      return AM;
    });

    PassBuilder PB = makePassBuilder(Ctx);
    PB.registerModuleAnalyses(MAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
  }
};

} // namespace

/// A noreturn call marked willreturn is UB, which turns the guard in front of
/// an exit() into an assume in the clone that ships.
static void setFullWillReturn(Function *NewF) {
  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB || !isa<CallInst, InvokeInst>(CB) || CB->doesNotReturn())
        continue;
      CB->addFnAttr(Attribute::WillReturn);
      CB->addFnAttr(Attribute::MustProgress);
    }
  }
}

static std::pair<PHINode *, Instruction *>
insertNewCanonicalIV(Loop *L, Type *Ty, const Twine &Name) {
  BasicBlock *Header = L->getHeader();
  IRBuilder<> B = builderAt(Header, Header->begin());
  PHINode *CanonicalIV = B.CreatePHI(Ty, 1, Name);

  B.SetInsertPoint(getFirstNonPHIOrDbg(Header));
  auto *Inc = cast<Instruction>(
      B.CreateAdd(CanonicalIV, ConstantInt::get(Ty, 1), Name + ".next",
                  /*NUW*/ true, /*NSW*/ true));

  for (BasicBlock *Pred : predecessors(Header)) {
    if (L->contains(Pred))
      CanonicalIV->addIncoming(Inc, Pred);
    else
      CanonicalIV->addIncoming(ConstantInt::get(Ty, 0), Pred);
  }
  return {CanonicalIV, Inc};
}

static void removeRedundantIVs(BasicBlock *Header, PHINode *CanonicalIV,
                               Instruction *Increment, ScalarEvolution &SE) {
  const SCEV *CanonicalSCEV = SE.getSCEV(CanonicalIV);

  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II);) {
    PHINode *PN = cast<PHINode>(II);
    ++II;
    if (PN == CanonicalIV)
      continue;
    if (!SE.isSCEVable(PN->getType()))
      continue;
    const SCEV *S = SE.getSCEV(PN);
    if (SE.getCouldNotCompute() == S || isa<SCEVUnknown>(S))
      continue;
    if (!SE.dominates(S, Header))
      continue;

    if (S == CanonicalSCEV) {
      PN->replaceAllUsesWith(CanonicalIV);
      PN->eraseFromParent();
      continue;
    }

    IRBuilder<> B(PN);
    PHINode *Tmp = B.CreatePHI(PN->getType(), 0);
    for (auto *Pred : predecessors(Header))
      Tmp->addIncoming(PoisonValue::get(Tmp->getType()), Pred);
    PN->replaceAllUsesWith(Tmp);
    PN->eraseFromParent();

    SCEVExpander Exp = makeSCEVExpander(SE, "poseidon");
    // NOLINTNEXTLINE(enzyme-first-non-phi): ahead of the debug records
    BasicBlock::iterator at = Header->getFirstNonPHIIt();
    Value *NewIV = Exp.expandCodeFor(S, Tmp->getType(), at);

    if (auto *addrec = dyn_cast<SCEVAddRecExpr>(S)) {
      if (addrec->getLoop()->getHeader() == Header) {
        if (auto *add_or_mul = dyn_cast<BinaryOperator>(NewIV)) {
          if (addrec->hasNoUnsignedWrap())
            add_or_mul->setHasNoUnsignedWrap(true);
          if (addrec->hasNoSignedWrap())
            add_or_mul->setHasNoSignedWrap(true);
        }
      }
    }
    Tmp->replaceAllUsesWith(NewIV);
    Tmp->eraseFromParent();
  }

  Increment->moveAfter(getFirstNonPHI(CanonicalIV->getParent()));
  SmallVector<Instruction *, 1> toErase;
  for (auto *use : CanonicalIV->users()) {
    auto *BO = dyn_cast<BinaryOperator>(use);
    if (!BO || BO->getOpcode() != BinaryOperator::Add || use == Increment)
      continue;

    Value *toadd = BO->getOperand(0) == CanonicalIV ? BO->getOperand(1)
                                                    : BO->getOperand(0);
    auto *CI = dyn_cast<ConstantInt>(toadd);
    if (!CI || !CI->isOne())
      continue;
    BO->replaceAllUsesWith(Increment);
    toErase.push_back(BO);
  }
  for (auto *BO : toErase)
    BO->eraseFromParent();
}

static void canonicalizeLoops(Function *F, FunctionAnalysisManager &FAM) {
  LoopSimplifyPass().run(*F, FAM);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);
  AssumptionCache &AC = FAM.getResult<AssumptionAnalysis>(*F);
  TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(*F);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  for (Loop *L : LI.getLoopsInPreorder()) {
    auto [iv, increment] =
        insertNewCanonicalIV(L, Type::getInt64Ty(F->getContext()), "iv");
    removeRedundantIVs(L->getHeader(), iv, increment, SE);
  }
  PreservedAnalyses PA;
  PA.preserve<AssumptionAnalysis>();
  PA.preserve<TargetLibraryAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<TypeBasedAA>();
  PA.preserve<BasicAA>();
  PA.preserve<ScopedNoAliasAA>();
  FAM.invalidate(*F, PA);
}

static void removeRedundantPHI(Function *F, FunctionAnalysisManager &FAM) {
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
  for (BasicBlock &BB : *F) {
    for (BasicBlock::iterator II = BB.begin(); isa<PHINode>(II);) {
      PHINode *PN = cast<PHINode>(II);
      ++II;
      SmallPtrSet<Value *, 2> vals;
      SmallPtrSet<PHINode *, 2> done;
      SmallVector<PHINode *, 2> todo = {PN};
      while (todo.size() > 0) {
        PHINode *N = todo.back();
        todo.pop_back();
        if (done.count(N))
          continue;
        done.insert(N);
        if (vals.size() == 0 && todo.size() == 0 && PN != N &&
            DT.dominates(N, PN)) {
          vals.insert(N);
          break;
        }
        for (auto &v : N->incoming_values()) {
          if (isa<UndefValue>(v))
            continue;
          if (auto *NN = dyn_cast<PHINode>(v)) {
            todo.push_back(NN);
            continue;
          }
          vals.insert(v);
          if (vals.size() > 1)
            break;
        }
        if (vals.size() > 1)
          break;
      }
      if (vals.size() == 1) {
        auto *V = *vals.begin();
        if (!isa<Instruction>(V) || DT.dominates(cast<Instruction>(V), PN)) {
          PN->replaceAllUsesWith(V);
          PN->eraseFromParent();
        }
      }
    }
  }
}

namespace {

/// The operand of a fused multiply-add that is negated before the fusion.
enum class Negate { None, X, Z };

} // namespace

/// Replaces `I` by fmuladd(X, Y, Z), with the operand `N` names negated first;
/// the negation and the fmuladd carry the fast-math flags of `I`.
static void fuseMulAdd(BinaryOperator *I, Value *X, Value *Y, Value *Z,
                       Negate N) {
  IRBuilder<> B(I);
  B.setFastMathFlags(I->getFastMathFlags());
  if (N == Negate::X)
    X = B.CreateFNeg(X);
  if (N == Negate::Z)
    Z = B.CreateFNeg(Z);
  Value *FMulAdd =
      B.CreateIntrinsic(Intrinsic::fmuladd, I->getType(), {X, Y, Z});
  I->replaceAllUsesWith(FMulAdd);
  I->eraseFromParent();
}

/// fmul+fadd and fmul+fsub where contraction is allowed become fmuladd, and an
/// fcmp+select that picks the larger or the smaller operand becomes
/// maxnum/minnum.
static void preprocess(Function *F) {
  using namespace llvm::PatternMatch;

  for (auto &BB : *F) {
    for (auto &I : make_early_inc_range(BB)) {
      Value *X, *Y, *Z;
      if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
        // `contract` alone licenses fusing a*b+c into one rounded fmuladd;
        // `reassoc` is about regrouping across additions and is not needed.
        // CUDA-default -ffp-contract=fast sets contract but not reassoc.
        if (!isa<FPMathOperator>(BO) || !BO->hasAllowContract())
          continue;
        if (match(BO, m_FAdd(m_OneUse(m_FMul(m_Value(X), m_Value(Y))),
                             m_Value(Z))) ||
            match(BO,
                  m_FAdd(m_Value(Z), m_OneUse(m_FMul(m_Value(X), m_Value(Y))))))
          fuseMulAdd(BO, X, Y, Z, Negate::None);
        else if (match(BO, m_FSub(m_OneUse(m_FMul(m_Value(X), m_Value(Y))),
                                  m_Value(Z))))
          fuseMulAdd(BO, X, Y, Z, Negate::Z);
        else if (match(BO, m_FSub(m_Value(Z),
                                  m_OneUse(m_FMul(m_Value(X), m_Value(Y))))))
          fuseMulAdd(BO, X, Y, Z, Negate::X);
        continue;
      }

      auto *Select = dyn_cast<SelectInst>(&I);
      if (!Select || !Select->getType()->isFloatingPointTy())
        continue;
      Value *Cond = Select->getCondition();
      Value *TrueVal = Select->getTrueValue();
      Value *FalseVal = Select->getFalseValue();
      CmpPredicate Pred;
      Value *CmpLHS, *CmpRHS;
      if (!match(Cond, m_FCmp(Pred, m_Value(CmpLHS), m_Value(CmpRHS))))
        continue;
      IRBuilder<> B(Select);
      Value *Result = nullptr;
      if (Pred == FCmpInst::FCMP_OGT && match(CmpRHS, m_AnyZeroFP()) &&
          CmpLHS == TrueVal && match(FalseVal, m_AnyZeroFP())) {
        Result = B.CreateIntrinsic(
            Intrinsic::maxnum, CmpLHS->getType(),
            {CmpLHS, ConstantFP::get(CmpLHS->getType(), 0.0)});
      } else if (Pred == FCmpInst::FCMP_OLT && match(CmpRHS, m_AnyZeroFP()) &&
                 CmpLHS == FalseVal && match(TrueVal, m_AnyZeroFP())) {
        Result = B.CreateIntrinsic(
            Intrinsic::maxnum, CmpLHS->getType(),
            {CmpLHS, ConstantFP::get(CmpLHS->getType(), 0.0)});
      } else if (Pred == FCmpInst::FCMP_OGT && CmpLHS == TrueVal &&
                 CmpRHS == FalseVal) {
        Result = B.CreateIntrinsic(Intrinsic::maxnum, CmpLHS->getType(),
                                   {CmpLHS, CmpRHS});
      } else if (Pred == FCmpInst::FCMP_OLT && CmpLHS == TrueVal &&
                 CmpRHS == FalseVal) {
        Result = B.CreateIntrinsic(Intrinsic::minnum, CmpLHS->getType(),
                                   {CmpLHS, CmpRHS});
      }
      if (!Result)
        continue;
      Select->replaceAllUsesWith(Result);
      Select->eraseFromParent();
      if (auto *FCmp = dyn_cast<FCmpInst>(Cond); FCmp && FCmp->use_empty())
        FCmp->eraseFromParent();
    }
  }
}

Function *canonicalize(Function &F) {
  Function *NewF =
      Function::Create(F.getFunctionType(), F.getLinkage(),
                       kCanonicalPrefix + F.getName(), F.getParent());

  ValueToValueMapTy VMap;
  for (auto i = F.arg_begin(), j = NewF->arg_begin(); i != F.arg_end();) {
    VMap[i] = j;
    j->setName(i->getName());
    ++i;
    ++j;
  }

  SmallVector<ReturnInst *, 4> Returns;
  if (!F.empty())
    CloneFunctionInto(NewF, &F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", nullptr);

  // The clone ships, so it may promise no more than the original: a body that
  // can call exit() or spin forever is neither willreturn nor mustprogress.
  NewF->setAttributes(F.getAttributes());
  setFullWillReturn(NewF);
  resetClonedGUID(NewF);

  CanonicalizeAnalyses A(NewF->getContext());
  FunctionAnalysisManager &FAM = A.FAM;

  auto run = [&](auto &&Pass) { FAM.invalidate(*NewF, Pass.run(*NewF, FAM)); };
  run(PromotePass());
  run(LowerInvokePass());
  run(UnreachableBlockElimPass());
  run(PromotePass());
  run(SROAPass(SROAOptions::ModifyCFG));
  run(SROAPass(SROAOptions::PreserveCFG));
  run(SimplifyCFGPass(SimplifyCFGOptions()));

  preprocess(NewF);
  run(InstCombinePass());
  run(EarlyCSEPass());
  run(GVNPass());

  canonicalizeLoops(NewF, FAM);
  removeRedundantPHI(NewF, FAM);
  run(LoopSimplifyPass());

  for (auto &BB : *NewF) {
    for (auto &I : make_early_inc_range(BB)) {
      auto *MTI = dyn_cast<MemTransferInst>(&I);
      if (!MTI)
        continue;
      if (auto *CI = dyn_cast<ConstantInt>(MTI->getOperand(2)))
        if (CI->getValue() == 0)
          MTI->eraseFromParent();
    }
  }

  if (verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *NewF << '\n';
    reportFatalInternalError("Poseidon: a canonical clone does not verify");
  }
  return NewF;
}

} // namespace poseidon

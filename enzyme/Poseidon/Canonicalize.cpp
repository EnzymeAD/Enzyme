#include "Canonicalize.h"

#include "Optimize.h"

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
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

namespace poseidon {

std::string canonicalFormHash(const Function &F) {
  std::string form;
  raw_string_ostream os(form);
  size_t slot = 0;
  for (const Instruction &I : instructions(F)) {
    if (!isOptimizable(I))
      continue;
    os << slot++ << "|" << I.getOpcodeName() << "|";
    I.getType()->print(os);
    if (const auto *CI = dyn_cast<CallInst>(&I)) {
      for (const Value *arg : CI->args()) {
        os << "|";
        arg->getType()->print(os);
      }
      const Function *callee = CI->getCalledFunction();
      os << "|" << (callee ? callee->getName() : StringRef("<indirect>"));
    } else {
      for (const Use &op : I.operands()) {
        os << "|";
        op->getType()->print(os);
      }
    }
    os << "\n";
  }

  uint64_t h = 14695981039346656037ULL; // FNV-1a 64 offset basis
  for (char c : form) {
    h ^= (uint64_t)(unsigned char)c;
    h *= 1099511628211ULL; // FNV-1a 64 prime
  }
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[(size_t)i] = "0123456789abcdef"[h & 0xFULL];
    h >>= 4;
  }
  return out;
}

namespace {

struct CanonicalizeAnalyses {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;

  CanonicalizeAnalyses() {
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
      auto AM = AAManager();
      AM.registerFunctionAnalysis<BasicAA>();
      AM.registerFunctionAnalysis<TypeBasedAA>();
      AM.registerModuleAnalysis<GlobalsAA>();
      AM.registerFunctionAnalysis<ScopedNoAliasAA>();
      return AM;
    });

    PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
  }
};

// A noreturn call marked willreturn is UB, which turns the guard in front of an
// exit() into an assume in the clone that ships.
void setFullWillReturn(Function *NewF) {
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

std::pair<PHINode *, Instruction *> insertNewCanonicalIV(Loop *L, Type *Ty,
                                                         const Twine &Name) {
  BasicBlock *Header = L->getHeader();
  IRBuilder<> B(Header, Header->begin());
  PHINode *CanonicalIV = B.CreatePHI(Ty, 1, Name);

  B.SetInsertPoint(Header->getFirstNonPHIOrDbg());
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

void removeRedundantIVs(BasicBlock *Header, PHINode *CanonicalIV,
                        Instruction *Increment, ScalarEvolution &SE) {
  auto *CanonicalSCEV = SE.getSCEV(CanonicalIV);

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
    auto *Tmp = B.CreatePHI(PN->getType(), 0);
    for (auto *Pred : predecessors(Header))
      Tmp->addIncoming(UndefValue::get(Tmp->getType()), Pred);
    PN->replaceAllUsesWith(Tmp);
    PN->eraseFromParent();

#if LLVM_VERSION_MAJOR >= 22
    SCEVExpander Exp(SE, "poseidon");
#else
    SCEVExpander Exp(SE, SE.getDataLayout(), "poseidon");
#endif
    Value *NewIV =
        Exp.expandCodeFor(S, Tmp->getType(), Header->getFirstNonPHIIt());

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

  Increment->moveAfter(&*CanonicalIV->getParent()->getFirstNonPHIIt());
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

void canonicalizeLoops(Function *F, FunctionAnalysisManager &FAM) {
  LoopSimplifyPass().run(*F, FAM);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);
  AssumptionCache &AC = FAM.getResult<AssumptionAnalysis>(*F);
  TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(*F);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  for (Loop *L : LI.getLoopsInPreorder()) {
    auto pair =
        insertNewCanonicalIV(L, Type::getInt64Ty(F->getContext()), "iv");
    removeRedundantIVs(L->getHeader(), pair.first, pair.second, SE);
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

void removeRedundantPHI(Function *F, FunctionAnalysisManager &FAM) {
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

} // namespace

Function *canonicalize(Function &F) {
  Function *NewF = Function::Create(F.getFunctionType(), F.getLinkage(),
                                    "preprocess_" + F.getName(), F.getParent());

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

  NewF->setAttributes(F.getAttributes());
  NewF->addFnAttr(Attribute::WillReturn);
  NewF->addFnAttr(Attribute::MustProgress);
  setFullWillReturn(NewF);

  CanonicalizeAnalyses A;
  FunctionAnalysisManager &FAM = A.FAM;

  {
    auto PA = PromotePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }
  {
    auto PA = LowerInvokePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }
  {
    auto PA = UnreachableBlockElimPass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }
  {
    auto PA = PromotePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }
  {
    auto PA = SROAPass(SROAOptions::ModifyCFG).run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }
  {
    auto PA = SROAPass(SROAOptions::PreserveCFG).run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }
  {
    SimplifyCFGOptions scfgo;
    auto PA = SimplifyCFGPass(scfgo).run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }

  preprocess(NewF);
  {
    auto PA = InstCombinePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
    PA = EarlyCSEPass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
    PA = GVNPass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }

  canonicalizeLoops(NewF, FAM);
  removeRedundantPHI(NewF, FAM);

  {
    auto PA = LoopSimplifyPass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }

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
    llvm::errs() << *NewF << "\n";
    report_fatal_error("Poseidon canonicalized function failed verification");
  }
  return NewF;
}

} // namespace poseidon

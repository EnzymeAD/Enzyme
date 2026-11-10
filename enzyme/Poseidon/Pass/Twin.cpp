#include "Twin.h"

#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <memory>
#include <string>

using namespace llvm;

namespace poseidon {

// Runs the default pipeline of `Phase` on `twin` as a build without a profile,
// the diagnostics already given for the module's sites silenced. False if it
// reported an error.
bool compileAsWritten(Module &twin, const Session &parent, PassBuilder &PB,
                      OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
  struct Quiet : DiagnosticHandler {
    bool failed = false;
    bool handleDiagnostics(const DiagnosticInfo &DI) override {
      failed |= DI.getSeverity() == DS_Error;
      return true;
    }
  };
  LLVMContext &C = twin.getContext();
  std::unique_ptr<DiagnosticHandler> handler = C.getDiagnosticHandler();
  auto quiet = std::make_unique<Quiet>();
  Quiet *q = quiet.get();
  C.setDiagnosticHandler(std::move(quiet));
  Options asWritten = parent.Opts;
  asWritten.Profiling.ProfileUse.clear();
  asWritten.Diagnostics.Print = false;
  Session S(std::move(asWritten), /*IsTwin=*/true);
  {
    LocalAnalysisManagers AMs(PB, S);
    ModulePassManager MPM =
        Phase == ThinOrFullLTOPhase::ThinLTOPreLink
            ? PB.buildThinLTOPreLinkDefaultPipeline(Level)
            : PB.buildPerModuleDefaultPipeline(Level, Phase);
    MPM.run(twin, AMs.MAM);
  }
  const bool failed = q->failed;
  C.setDiagnosticHandler(std::move(handler));
  return !failed;
}

namespace {

// What a function copied from the rewritten module into the module as written
// refers to: a declaration, or a variable that is one object for the whole
// program, is the module's own of that name; anything else the rewritten
// module defines is copied along, under its name unless the module has it.
struct RewrittenReferences final : ValueMaterializer {
  Module &T;
  SmallVector<GlobalValue *, 8> pending;
  bool mismatch = false;

  explicit RewrittenReferences(Module &T) : T(T) {}

  Value *materialize(Value *V) override {
    auto *G = dyn_cast<GlobalValue>(V);
    if (!G || G->getParent() == &T)
      return nullptr;
    auto *F = dyn_cast<Function>(G);
    auto *var = dyn_cast<GlobalVariable>(G);
    if (!F && !var) {
      mismatch = true;
      return PoisonValue::get(G->getType());
    }
    const bool copied =
        !G->isDeclaration() &&
        (F ||
         (G->hasLocalLinkage() &&
          (var->isConstant() || Target::of(*var->getParent()).isShared(*var))));
    GlobalValue *own = T.getNamedValue(G->getName());
    if (own && !copied) {
      if (own->getValueID() != G->getValueID() ||
          own->getValueType() != G->getValueType() ||
          own->getAddressSpace() != G->getAddressSpace() ||
          (F && cast<Function>(own)->getCallingConv() != F->getCallingConv())) {
        mismatch = true;
        return PoisonValue::get(G->getType());
      }
      return own;
    }
    GlobalValue *NG;
    if (F) {
      Function *NF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                      F->getAddressSpace(), F->getName(), &T);
      NF->copyAttributesFrom(F);
      NG = NF;
    } else {
      auto *NV = new GlobalVariable(
          T, var->getValueType(), var->isConstant(), var->getLinkage(), nullptr,
          var->getName(), nullptr, var->getThreadLocalMode(),
          var->getAddressSpace(), var->isExternallyInitialized());
      NV->copyAttributesFrom(var);
      NG = NV;
    }
    if (own)
      NG->setLinkage(GlobalValue::InternalLinkage);
    if (!G->isDeclaration())
      pending.push_back(G);
    return NG;
  }
};

void collectReferences(Value *V, SmallVectorImpl<WeakVH> &out) {
  if (auto *G = dyn_cast<GlobalValue>(V))
    out.push_back(G);
  else if (auto *C = dyn_cast<Constant>(V))
    for (Value *op : C->operands())
      collectReferences(op, out);
}

void collectReferences(Function &F, SmallVectorImpl<WeakVH> &out) {
  for (Instruction &I : instructions(F))
    for (Value *op : I.operands())
      if (isa<Constant>(op))
        collectReferences(op, out);
}

} // namespace

// Into `T`, the module as written compiled, the functions of `M` a rewrite
// reached, each in place of T's of the same name, with what they refer to.
// False if a reference has no counterpart in T, which is then unusable.
bool takeRewritten(Module &T, ArrayRef<Function *> rewritten) {
  ValueToValueMapTy VMap;
  RewrittenReferences refs(T);
  SmallVector<std::pair<Function *, Function *>, 4> replaced;
  for (Function *F : rewritten) {
    Function *own = T.getFunction(F->getName());
    Function *NF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                    F->getAddressSpace(), F->getName());
    if (own)
      T.getFunctionList().insert(own->getIterator(), NF);
    else
      T.getFunctionList().push_back(NF);
    if (own && own->getFunctionType() == F->getFunctionType())
      replaced.push_back({own, NF});
    else if (own)
      NF->setLinkage(GlobalValue::InternalLinkage);
    VMap[F] = NF;
    refs.pending.push_back(F);
  }
  while (!refs.pending.empty()) {
    GlobalValue *G = refs.pending.pop_back_val();
    if (auto *F = dyn_cast<Function>(G)) {
      auto *NF = cast<Function>(VMap[F]);
      for (auto [from, to] : zip(F->args(), NF->args()))
        VMap[&from] = &to;
      // Cloning copies the visibility, which a local copy cannot keep.
      const GlobalValue::LinkageTypes linkage = NF->getLinkage();
      NF->setLinkage(F->getLinkage());
      SmallVector<ReturnInst *, 4> returns;
      CloneFunctionInto(NF, F, VMap, CloneFunctionChangeType::DifferentModule,
                        returns, "", nullptr, nullptr, &refs);
      NF->setLinkage(linkage);
    } else {
      auto *V = cast<GlobalVariable>(G);
      cast<GlobalVariable>(VMap[V])->setInitializer(
          MapValue(V->getInitializer(), VMap, RF_None, nullptr, &refs));
    }
  }
  if (refs.mismatch)
    return false;

  SmallVector<WeakVH, 16> maybeDead;
  for (auto [own, NF] : replaced) {
    collectReferences(*own, maybeDead);
    own->replaceAllUsesWith(NF);
    NF->setComdat(own->getComdat());
    std::string name = own->getName().str();
    own->eraseFromParent();
    NF->setName(name);
  }
  // What only T's own versions of them referred to.
  while (!maybeDead.empty()) {
    auto *G = dyn_cast_or_null<GlobalValue>(maybeDead.pop_back_val());
    if (!G)
      continue;
    G->removeDeadConstantUsers();
    if (!G->use_empty() || !G->isDiscardableIfUnused())
      continue;
    if (auto *F = dyn_cast<Function>(G))
      collectReferences(*F, maybeDead);
    else if (auto *V = dyn_cast<GlobalVariable>(G); V && V->hasInitializer())
      collectReferences(V->getInitializer(), maybeDead);
    else
      continue;
    G->eraseFromParent();
  }
  return true;
}

// The contents of `M` replaced by those of `from`, which is left empty.
void adoptModule(Module &M, Module &from) {
  M.dropAllReferences();
  for (GlobalValue &GV : make_early_inc_range(M.global_values()))
    GV.removeDeadConstantUsers();
  for (Function &F : make_early_inc_range(M))
    F.eraseFromParent();
  for (GlobalVariable &GV : make_early_inc_range(M.globals()))
    GV.eraseFromParent();
  for (GlobalAlias &GA : make_early_inc_range(M.aliases()))
    GA.eraseFromParent();
  for (GlobalIFunc &GI : make_early_inc_range(M.ifuncs()))
    GI.eraseFromParent();
  for (NamedMDNode &N : make_early_inc_range(M.named_metadata()))
    M.eraseNamedMetadata(&N);

  auto comdat = [&](GlobalObject &GO) {
    if (const Comdat *C = GO.getComdat()) {
      Comdat *own = M.getOrInsertComdat(C->getName());
      own->setSelectionKind(C->getSelectionKind());
      GO.setComdat(own);
    }
  };
  for (GlobalVariable &GV : make_early_inc_range(from.globals())) {
    GV.removeFromParent();
    M.insertGlobalVariable(&GV);
    comdat(GV);
  }
  for (Function &F : make_early_inc_range(from)) {
    F.removeFromParent();
    M.getFunctionList().push_back(&F);
    comdat(F);
  }
  for (GlobalAlias &GA : make_early_inc_range(from.aliases())) {
    GA.removeFromParent();
    M.insertAlias(&GA);
  }
  for (GlobalIFunc &GI : make_early_inc_range(from.ifuncs())) {
    GI.removeFromParent();
    M.insertIFunc(&GI);
  }
  for (NamedMDNode &N : from.named_metadata()) {
    NamedMDNode *own = M.getOrInsertNamedMetadata(N.getName());
    for (MDNode *op : N.operands())
      own->addOperand(op);
  }
  M.setModuleInlineAsm(from.getModuleInlineAsm());
}

} // namespace poseidon

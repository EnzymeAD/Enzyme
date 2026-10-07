//===- ThinLTOImport.cpp - Bring functions to differentiate into ThinLTO --===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See ThinLTOImport.h.
//
//===----------------------------------------------------------------------===//

#include "ThinLTOImport.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

// Internal, kept alive by llvm.compiler.used, and never called: it exists only
// for the import GUIDs on its !prof.
static constexpr const char *AnchorName = "enzyme.thinlto.imports";
static constexpr const char *CopySuffix = ".enzyme.thinlto";

static Function *calledFunction(const CallBase &CB) {
  return dyn_cast<Function>(CB.getCalledOperand()->stripPointerCasts());
}

// Calls to the __enzyme_* markers that Enzyme lowers.
template <typename T> static void forEachEnzymeCall(Module &M, T &&Fn) {
  for (Function &F : M) {
    if (!F.isDeclaration() || !F.getName().contains("__enzyme_"))
      continue;
    for (User *U : F.users())
      if (auto *CB = dyn_cast<CallBase>(U))
        if (calledFunction(*CB) == &F)
          Fn(*CB);
  }
}

static GlobalValue::GUID guidOf(const Function &F) {
#if LLVM_VERSION_MAJOR >= 24
  if (auto G = F.getGUIDIfAssigned())
    return *G;
  return GlobalValue::getGUIDAssumingExternalLinkage(
      GlobalValue::getGlobalIdentifier(F.getName(), F.getLinkage(),
                                       F.getParent()->getSourceFileName()));
#else
  return F.getGUID();
#endif
}

bool enzymeThinLTORequestImports(Module &M) {
  // Every function an __enzyme_* call takes, whether the one to differentiate
  // or a function pointer passed as an argument, which needs a shadow.
  SmallVector<Function *, 8> Worklist;
  forEachEnzymeCall(M, [&](CallBase &CB) {
    for (Value *A : CB.args())
      if (auto *G = dyn_cast<Function>(A->stripPointerCasts()))
        Worklist.push_back(G);
  });

  // Differentiating a function defined here needs the bodies of the functions
  // it calls as well, so ask for those it only declares. The importer follows
  // the call edges of imported functions on its own, with a threshold that
  // decays per level.
  DenseSet<GlobalValue::GUID> GUIDs;
  SmallPtrSet<Function *, 16> Seen;
  while (!Worklist.empty()) {
    Function *G = Worklist.pop_back_val();
    if (!Seen.insert(G).second)
      continue;
    if (G->isDeclaration()) {
      if (!G->isIntrinsic() && !G->hasLocalLinkage() &&
          !G->getName().starts_with("__enzyme_"))
        GUIDs.insert(guidOf(*G));
      continue;
    }
    for (Instruction &I : instructions(*G))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *Callee = calledFunction(*CB))
          Worklist.push_back(Callee);
  }
  if (GUIDs.empty())
    return false;

  LLVMContext &Ctx = M.getContext();
  Function *Anchor = M.getFunction(AnchorName);
  if (Anchor) {
    for (auto G : Anchor->getImportGUIDs())
      GUIDs.insert(G);
  } else {
    Anchor = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                              GlobalValue::InternalLinkage, AnchorName, M);
    Anchor->addFnAttr(Attribute::NoInline);
    Anchor->addFnAttr(Attribute::OptimizeNone);
    ReturnInst::Create(Ctx, BasicBlock::Create(Ctx, "", Anchor));
    appendToCompilerUsed(M, {Anchor});
  }
  // ModuleSummaryAnalysis turns these GUIDs into call edges of hotness
  // Critical, which raise the size threshold for importing the callee
  // (-import-critical-multiplier).
  Anchor->setMetadata(
      LLVMContext::MD_prof,
      MDBuilder(Ctx).createFunctionEntryCount(1, /*Synthetic=*/false, &GUIDs));
  return true;
}

bool enzymeThinLTOLocalizeImports(Module &M) {
  bool Changed = false;
  if (Function *Anchor = M.getFunction(AnchorName)) {
    removeFromUsedLists(M, [&](Constant *C) { return C == Anchor; });
    Anchor->eraseFromParent();
    Changed = true;
  }

  // The function to differentiate is the first argument. An imported one has
  // available_externally linkage, so EliminateAvailableExternally would drop
  // its body before Enzyme runs.
  auto IsImported = [](Function *G) {
    return G && G->hasAvailableExternallyLinkage() && !G->isDeclaration();
  };
  SmallVector<Use *, 4> RootUses;
  forEachEnzymeCall(M, [&](CallBase &CB) {
    if (CB.arg_size() == 0)
      return;
    Use &U = CB.getArgOperandUse(0);
    if (IsImported(dyn_cast<Function>(U.get()->stripPointerCasts())))
      RootUses.push_back(&U);
  });
  if (RootUses.empty())
    return Changed;

  // The imported functions they call, transitively, lose their bodies too.
  SetVector<Function *> Closure;
  for (Use *U : RootUses)
    Closure.insert(cast<Function>(U->get()->stripPointerCasts()));
  for (size_t i = 0; i < Closure.size(); ++i)
    for (Instruction &I : instructions(*Closure[i]))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *Callee = calledFunction(*CB); IsImported(Callee))
          Closure.insert(Callee);

  // Internal copies keep their bodies. Only calls and the __enzyme_* operand
  // are redirected to them: any other use of the function, e.g. its address
  // stored or compared, still means the original.
  MapVector<Function *, Function *> Copies;
  for (Function *G : Closure) {
    ValueToValueMapTy VMap;
    Function *C = CloneFunction(G, VMap);
    C->setName(G->getName() + CopySuffix);
    C->setLinkage(GlobalValue::InternalLinkage);
    C->setVisibility(GlobalValue::DefaultVisibility);
    C->setDLLStorageClass(GlobalValue::DefaultStorageClass);
    C->setComdat(nullptr);
    Copies[G] = C;
  }
  for (auto &[G, C] : Copies)
    for (Instruction &I : instructions(*C))
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        auto Found = Copies.find(calledFunction(*CB));
        if (Found != Copies.end() && CB->getCalledOperand() == Found->first)
          CB->setCalledOperand(Found->second);
      }
  for (Use *U : RootUses) {
    Function *C = Copies[cast<Function>(U->get()->stripPointerCasts())];
    U->set(ConstantExpr::getPointerCast(C, U->get()->getType()));
  }
  return true;
}

llvm::AnalysisKey EnzymeThinLTOImportPass::Key;

PreservedAnalyses EnzymeThinLTOImportPass::run(Module &M,
                                               ModuleAnalysisManager &) {
  bool Changed = PostLink ? enzymeThinLTOLocalizeImports(M)
                          : enzymeThinLTORequestImports(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

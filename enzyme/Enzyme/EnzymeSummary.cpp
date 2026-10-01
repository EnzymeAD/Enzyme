//===- EnzymeSummary.cpp - Per-module facts for a thin-link Enzyme step ---===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The enzyme-summary pass writes, for one module, the facts a whole-program
// ("thin link") step needs to plan separate compilation with
// -enzyme-separate-compilation, without seeing any function body:
//
//  * the call graph (direct callees, functions whose address is taken),
//  * per function: whether it touches floating-point data, copies memory,
//    allocates, returns a pointer, and the "enzyme_type" declared on its
//    signature,
//  * the __enzyme_* calls (what is differentiated, in which mode, width,
//    strong zero, runtime activity),
//  * the __enzyme_* registrations (inactive, nofree, non-escaping
//    allocations, custom derivatives),
//  * the global variables (for the table of shadows of COMMON blocks).
//
// The facts are written as JSON to -enzyme-summary-out. A tool run between
// the ThinLTO index phase and the backends combines them into per-module
// instructions (which derivatives to export, which callees are inactive).
//
//===----------------------------------------------------------------------===//

#include "EnzymeSummary.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <string>

using namespace llvm;

static cl::opt<std::string>
    EnzymeSummaryOut("enzyme-summary-out", cl::init(""), cl::Hidden,
                     cl::desc("File the enzyme-summary pass writes the "
                              "module's facts to (JSON); stdout if empty"));

namespace {

/// Function a constant (e.g. a registration's initializer) points to.
Function *pointee(Constant *C) {
  if (!C)
    return nullptr;
  return dyn_cast<Function>(C->stripPointerCasts());
}

/// The primal function of a registration global: the pointer itself, or the
/// first field of a {primal, ...} aggregate.
Function *registeredPrimal(GlobalVariable &G) {
  if (!G.hasInitializer())
    return nullptr;
  auto *Init = G.getInitializer();
  if (auto *F = pointee(Init))
    return F;
  if (isa<ConstantStruct>(Init) || isa<ConstantArray>(Init))
    if (Init->getNumOperands())
      return pointee(cast<Constant>(Init->getOperand(0)));
  return nullptr;
}

StringRef registrationKind(StringRef Name) {
  if (Name.contains("__enzyme_inactivefn"))
    return "inactive";
  if (Name.contains("__enzyme_nofree"))
    return "nofree";
  if (Name.contains("__enzyme_no_escaping_allocation"))
    return "no_escape";
  if (Name.contains("__enzyme_register_gradient") ||
      Name.contains("__enzyme_register_derivative") ||
      Name.contains("__enzyme_register_splitderivative"))
    return "custom_rule";
  if (Name.contains("__enzyme_function_like"))
    return "function_like";
  return "";
}

StringRef adMode(StringRef Callee) {
  if (Callee.contains("__enzyme_autodiff") ||
      Callee.contains("__enzyme_augmentfwd") ||
      Callee.contains("__enzyme_reverse"))
    return "reverse";
  if (Callee.contains("__enzyme_fwdsplit"))
    return "forwardsplit";
  if (Callee.contains("__enzyme_fwddiff"))
    return "forward";
  return "";
}

bool isAllocationName(StringRef N) {
  return N == "malloc" || N == "calloc" || N == "realloc" ||
         N == "aligned_alloc" || N == "posix_memalign" || N == "_Znwm" ||
         N == "_Znam" || N.starts_with("_FortranAAllocatableAllocate") ||
         N.starts_with("_FortranAPointerAllocate");
}

bool touchesFloat(Instruction &I) {
  if (I.getType()->isFPOrFPVectorTy())
    return true;
  for (auto &Op : I.operands())
    if (Op->getType()->isFPOrFPVectorTy())
      return true;
  return false;
}

GlobalVariable *underlyingGlobal(Value *V) {
  V = V->stripPointerCasts();
  while (auto *GEP = dyn_cast<GEPOperator>(V))
    V = GEP->getPointerOperand()->stripPointerCasts();
  return dyn_cast<GlobalVariable>(V);
}

std::string linkageName(GlobalValue::LinkageTypes L) {
  switch (L) {
  case GlobalValue::ExternalLinkage:
    return "external";
  case GlobalValue::AvailableExternallyLinkage:
    return "available_externally";
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
    return "linkonce";
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
    return "weak";
  case GlobalValue::AppendingLinkage:
    return "appending";
  case GlobalValue::InternalLinkage:
    return "internal";
  case GlobalValue::PrivateLinkage:
    return "private";
  case GlobalValue::ExternalWeakLinkage:
    return "extern_weak";
  case GlobalValue::CommonLinkage:
    return "common";
  }
  return "unknown";
}

json::Array toArray(const std::set<std::string> &S) {
  json::Array A;
  for (auto &E : S)
    A.push_back(E);
  return A;
}

json::Value signatureType(const AttributeList &AL, unsigned ArgNo) {
  if (AL.hasParamAttr(ArgNo, "enzyme_type"))
    return AL.getParamAttr(ArgNo, "enzyme_type").getValueAsString().str();
  return nullptr;
}

json::Object summarizeFunction(Function &F) {
  std::set<std::string> Calls, Refs, Globals;
  bool FP = false, MemTransfer = false, Allocates = false;
  unsigned Indirect = 0, Insts = 0;
  for (auto &I : instructions(F)) {
    ++Insts;
    FP |= touchesFloat(I);
    if (isa<MemTransferInst>(&I) || isa<MemSetInst>(&I))
      MemTransfer = true;
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      auto *Callee =
          dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
      if (!Callee)
        ++Indirect;
      else if (!Callee->isIntrinsic()) {
        Calls.insert(Callee->getName().str());
        Allocates |= isAllocationName(Callee->getName());
      }
      for (auto &A : CB->args()) {
        if (auto *G = dyn_cast<Function>(A->stripPointerCasts()))
          Refs.insert(G->getName().str());
        if (auto *GV = underlyingGlobal(A))
          Globals.insert(GV->getName().str());
      }
      continue;
    }
    for (unsigned i = 0, e = I.getNumOperands(); i < e; ++i) {
      Value *Op = I.getOperand(i);
      if (auto *G = dyn_cast<Function>(Op->stripPointerCasts()))
        Refs.insert(G->getName().str());
      else if (auto *GV = underlyingGlobal(Op))
        Globals.insert(GV->getName().str());
    }
  }
  json::Array ArgTypes;
  for (auto &A : F.args())
    ArgTypes.push_back(signatureType(F.getAttributes(), A.getArgNo()));
  json::Value RetType = nullptr;
  if (F.getAttributes().getRetAttrs().hasAttribute("enzyme_type"))
    RetType = F.getAttributes()
                  .getRetAttrs()
                  .getAttribute("enzyme_type")
                  .getValueAsString()
                  .str();
  return json::Object{
      {"linkage", linkageName(F.getLinkage())},
      {"insts", Insts},
      {"calls", toArray(Calls)},
      {"refs", toArray(Refs)},
      {"indirect_calls", Indirect},
      {"globals", toArray(Globals)},
      {"fp", FP},
      {"memtransfer", MemTransfer},
      {"allocates", Allocates},
      {"returns_pointer", F.getReturnType()->isPointerTy()},
      {"arg_types", std::move(ArgTypes)},
      {"ret_type", std::move(RetType)},
      {"inactive", F.hasFnAttribute("enzyme_inactive")},
      {"nofree", F.hasFnAttribute(Attribute::NoFree)},
      {"no_escape", F.hasFnAttribute("enzyme_no_escaping_allocation")},
  };
}

/// One __enzyme_* differentiation call: the differentiated function and the
/// conventions callees' derivatives must be built under.
json::Value summarizeADCall(CallBase &CB, StringRef Mode) {
  Function *Fn = nullptr;
  if (CB.arg_size())
    Fn = dyn_cast<Function>(CB.getArgOperand(0)->stripPointerCasts());
  bool StrongZero = false, RuntimeActivity = false;
  int64_t Width = 1;
  for (unsigned i = 1, e = CB.arg_size(); i < e; ++i) {
    auto *G = dyn_cast<GlobalVariable>(CB.getArgOperand(i)->stripPointerCasts());
    if (!G)
      continue;
    StringRef N = G->getName();
    if (N == "enzyme_strong_zero")
      StrongZero = true;
    else if (N == "enzyme_runtime_activity")
      RuntimeActivity = true;
    else if (N == "enzyme_width" && i + 1 < e)
      if (auto *C = dyn_cast<ConstantInt>(CB.getArgOperand(i + 1)))
        Width = C->getSExtValue();
  }
  return json::Object{
      {"caller", CB.getFunction()->getName().str()},
      {"fn", Fn ? json::Value(Fn->getName().str()) : json::Value(nullptr)},
      {"mode", Mode.str()},
      {"width", Width},
      {"strong_zero", StrongZero},
      {"runtime_activity", RuntimeActivity},
  };
}

} // namespace

llvm::AnalysisKey EnzymeSummaryNewPM::Key;

PreservedAnalyses EnzymeSummaryNewPM::run(Module &M,
                                          ModuleAnalysisManager &) {
  json::Object Functions, Globals;
  json::Array ADCalls;
  std::map<std::string, std::set<std::string>> Registrations;

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    Functions[F.getName()] = summarizeFunction(F);
    for (auto &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (auto *Callee = dyn_cast<Function>(
                CB->getCalledOperand()->stripPointerCasts())) {
          auto Mode = adMode(Callee->getName());
          if (!Mode.empty())
            ADCalls.push_back(summarizeADCall(*CB, Mode));
        }
  }

  for (auto &G : M.globals()) {
    auto Kind = registrationKind(G.getName());
    if (!Kind.empty()) {
      if (auto *P = registeredPrimal(G))
        Registrations[Kind.str()].insert(P->getName().str());
      continue;
    }
    if (G.getName().starts_with("llvm."))
      continue;
    Globals[G.getName()] = json::Object{
        {"linkage", linkageName(G.getLinkage())},
        {"defined", !G.isDeclaration()},
        {"constant", G.isConstant()},
        {"size", (int64_t)M.getDataLayout().getTypeAllocSize(
                     G.getValueType())},
    };
  }

  json::Object Regs;
  for (auto &[K, S] : Registrations)
    Regs[K] = toArray(S);

  json::Value Out = json::Object{
      {"module", M.getModuleIdentifier()},
      {"functions", std::move(Functions)},
      {"ad_calls", std::move(ADCalls)},
      {"registrations", std::move(Regs)},
      {"globals", std::move(Globals)},
  };

  if (EnzymeSummaryOut.empty()) {
    outs() << formatv("{0:2}", Out) << "\n";
  } else {
    std::error_code EC;
    raw_fd_ostream OS(EnzymeSummaryOut, EC, sys::fs::OF_Text);
    if (EC)
      report_fatal_error(Twine("could not open -enzyme-summary-out file ") +
                         EnzymeSummaryOut + ": " + EC.message());
    OS << formatv("{0:2}", Out) << "\n";
  }
  return PreservedAnalyses::all();
}

//===- EnzymeSummary.cpp - Per-function facts for separate differentiation ===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// summarizeFunction computes, from one function's body, what it does to
// floating-point data: per argument and per global, whether it may read or
// write it, which arguments' data may flow into which arguments' memory or
// the return value, and symbolic edges for what its callees are given. It
// does not look into callees, so it can run per module (as the
// enzyme-summary pass does, for a thin-link step), or per Julia CodeInstance
// (through the C API), and the results can be cached and combined later.
//
// The enzyme-summary pass writes, for one module, every function's summary
// plus the facts a whole-program step needs without seeing any body:
//
//  * the __enzyme_* calls (what is differentiated, in which mode, width,
//    strong zero, runtime activity),
//  * the __enzyme_* registrations (inactive, nofree, non-escaping
//    allocations, custom derivatives),
//  * the global variables (for the table of shadows of COMMON blocks).
//
// The facts are written as JSON to -enzyme-summary-out.
//
//===----------------------------------------------------------------------===//

#include "EnzymeSummary.h"
#include "Utils.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

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
         N == "_Znam" || startsWith(N, "_FortranAAllocatableAllocate") ||
         startsWith(N, "_FortranAPointerAllocate") ||
         N == "julia.gc_alloc_obj" || N.contains("gc_pool_alloc") ||
         N.contains("gc_small_alloc") || N.contains("gc_big_alloc") ||
         N.contains("alloc_genericmemory") || N.contains("alloc_array");
}

/// Julia's codegen markers, which neither move data nor call user code.
/// julia.call and julia.call2 (calls through the generic entry point) are not
/// among them.
bool isJuliaMarker(StringRef N) {
  return startsWith(N, "julia.") && !startsWith(N, "julia.call") &&
         N != "julia.gc_alloc_obj";
}

/// Julia markers returning a pointer derived from one of their arguments.
const Value *juliaPassThrough(const CallBase &CB, StringRef N) {
  if (N == "julia.pointer_from_objref" && CB.arg_size() >= 1)
    return CB.getArgOperand(0);
  if (N == "julia.gc_loaded" && CB.arg_size() >= 2)
    return CB.getArgOperand(1);
  return nullptr;
}

Function *calledFunction(const CallBase &CB) {
  return dyn_cast<Function>(CB.getCalledOperand()->stripPointerCasts());
}

bool touchesFloat(Instruction &I) {
  if (I.getType()->isFPOrFPVectorTy())
    return true;
  for (auto &Op : I.operands())
    if (Op->getType()->isFPOrFPVectorTy())
      return true;
  return false;
}

bool carriesFloat(Type *T) {
  if (T->isFPOrFPVectorTy())
    return true;
  if (auto *ST = dyn_cast<StructType>(T)) {
    for (auto *E : ST->elements())
      if (carriesFloat(E))
        return true;
    return false;
  }
  if (auto *AT = dyn_cast<ArrayType>(T))
    return carriesFloat(AT->getElementType());
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

json::Value orNull(const std::string &S) {
  if (S.empty())
    return nullptr;
  return S;
}

std::string signatureType(const AttributeList &AL, unsigned ArgNo) {
  if (AL.hasParamAttr(ArgNo, "enzyme_type"))
    return AL.getParamAttr(ArgNo, "enzyme_type").getValueAsString().str();
  return "";
}

/// Where a pointer may point: into the memory reachable from an argument or
/// a global (dereferences collapsed, as in ActivityAnalysis.jl's pseudo
/// classes), into local memory, or anywhere.
struct Roots {
  std::set<unsigned> Args;
  std::set<std::string> Globals;
  bool Unknown = false;
  void merge(const Roots &O) {
    Args.insert(O.Args.begin(), O.Args.end());
    Globals.insert(O.Globals.begin(), O.Globals.end());
    Unknown |= O.Unknown;
  }
  bool nonLocal() const { return Unknown || !Args.empty() || !Globals.empty(); }
};

/// Effects on floating-point data of one function, before composing with
/// its callees: per argument and per global, whether it may read or write
/// floating-point data there, where that data flows, and symbolic edges for
/// what callees do with it.
class ActivityFacts {
public:
  ActivityFacts(Function &F, EnzymeFunctionSummary &S) : F(F), S(S) {
    unsigned N = F.arg_size();
    S.Args.assign(N, EnzymeArgEffects());
    S.Flow.assign(N + 1, std::vector<bool>(N + 2, false));
    for (auto &A : F.args()) {
      // An argument whose declared type has no floating-point part cannot
      // carry derivatives through untyped copies.
      auto T = signatureType(F.getAttributes(), A.getArgNo());
      if (!T.empty() && !StringRef(T).contains("Float"))
        IntTyped.insert(A.getArgNo());
    }
    for (auto &I : instructions(F))
      visit(I);
    S.ReturnsFP = F.getReturnType()->isFPOrFPVectorTy();
  }

private:
  Function &F;
  EnzymeFunctionSummary &S;
  std::set<unsigned> IntTyped;
  std::map<const Value *, Roots> Memo;
  std::map<const Value *, Roots> SourceMemo;

  Roots roots(const Value *V) {
    auto found = Memo.find(V);
    if (found != Memo.end())
      return found->second;
    Memo[V] = Roots(); // cycles (phis) contribute nothing new
    Roots R;
    V = V->stripPointerCasts();
    if (auto *GEP = dyn_cast<GEPOperator>(V))
      R = roots(GEP->getPointerOperand());
    else if (auto *A = dyn_cast<Argument>(V))
      R.Args.insert(A->getArgNo());
    else if (auto *G = dyn_cast<GlobalVariable>(V)) {
      if (!G->isConstant())
        R.Globals.insert(G->getName().str());
    } else if (isa<AllocaInst>(V) || isa<Constant>(V))
      ;
    else if (auto *L = dyn_cast<LoadInst>(V))
      R = roots(L->getPointerOperand()); // dereferences collapsed
    else if (auto *P = dyn_cast<PHINode>(V)) {
      for (auto &In : P->incoming_values())
        R.merge(roots(In));
    } else if (auto *Sel = dyn_cast<SelectInst>(V)) {
      R = roots(Sel->getTrueValue());
      R.merge(roots(Sel->getFalseValue()));
    } else if (auto *CB = dyn_cast<CallBase>(V)) {
      auto *Callee = calledFunction(*CB);
      StringRef N = Callee ? Callee->getName() : "";
      if (auto *Through = juliaPassThrough(*CB, N))
        R = roots(Through);
      // fresh memory, and the flang runtime's own handles (I/O cookies)
      else if (!(isAllocationName(N) || startsWith(N, "_Fortran")))
        R.Unknown = true;
    } else
      R.Unknown = true;
    Memo[V] = R;
    return R;
  }

  /// Where the data in a (non-pointer) value may come from: arguments passed
  /// by value, memory reachable from arguments or globals that was loaded,
  /// or somewhere unknown. Addresses are not data: pointer operands are not
  /// followed except through the loads that read them.
  Roots sources(const Value *V) {
    auto found = SourceMemo.find(V);
    if (found != SourceMemo.end())
      return found->second;
    SourceMemo[V] = Roots();
    Roots R;
    if (auto *A = dyn_cast<Argument>(V)) {
      if (!A->getType()->isPointerTy())
        R.Args.insert(A->getArgNo());
    } else if (isa<Constant>(V))
      ;
    else if (auto *L = dyn_cast<LoadInst>(V))
      R = roots(L->getPointerOperand());
    else if (auto *CB = dyn_cast<CallBase>(V);
             CB && !isa<IntrinsicInst>(CB)) {
      // What the callee computes from what it is given; what it reads
      // beyond its arguments is composed from its own summary later.
      for (auto &A : CB->args())
        R.merge(A->getType()->isPointerTy() ? roots(A) : sources(A));
    } else if (auto *I = dyn_cast<Instruction>(V)) {
      for (auto &Op : I->operands())
        if (!Op->getType()->isPointerTy() && !isa<BasicBlock>(Op))
          R.merge(sources(Op));
    }
    SourceMemo[V] = R;
    return R;
  }

  void flow(const Roots &From, const Roots &To, bool ToReturn = false) {
    auto flowTo = [&](unsigned Src) {
      for (auto i : To.Args)
        S.Flow[Src][i] = true;
      if (!To.Globals.empty())
        S.Flow[Src][S.globalsSink()] = true;
      if (ToReturn)
        S.Flow[Src][S.returnSink()] = true;
    };
    for (auto i : From.Args)
      flowTo(i);
    if (!From.Globals.empty())
      flowTo(S.globalsSource());
    S.Unknown |= From.Unknown;
  }

  /// A global that holds no floating-point data: its "enzyme_type" (e.g.
  /// a COMMON block annotated by flang) has no floating-point part, or it is
  /// a module-local variable whose IR type has none (flang types those).
  bool intTypedGlobal(const std::string &Name) const {
    auto *G = F.getParent()->getGlobalVariable(Name, /*AllowInternal*/ true);
    if (!G)
      return false;
    if (auto *MD = G->getMetadata("enzyme_type")) {
      SmallVector<const MDNode *, 4> todo = {MD};
      while (!todo.empty()) {
        auto *N = todo.pop_back_val();
        for (auto &Op : N->operands()) {
          if (auto *Str = dyn_cast_or_null<MDString>(Op.get()))
            if (Str->getString().contains("Float"))
              return false;
          if (auto *Sub = dyn_cast_or_null<MDNode>(Op.get()))
            todo.push_back(Sub);
        }
      }
      return true;
    }
    return G->hasLocalLinkage() && !carriesFloat(G->getValueType());
  }

  /// R without the arguments and globals declared to hold no floating-point
  /// data, for untyped effects (memcpy, memset, runtime calls).
  Roots typed(Roots R) const {
    for (auto i : IntTyped)
      R.Args.erase(i);
    for (auto it = R.Globals.begin(); it != R.Globals.end();)
      it = intTypedGlobal(*it) ? R.Globals.erase(it) : std::next(it);
    return R;
  }

  void read(const Roots &R) {
    for (auto i : R.Args)
      S.Args[i].ReadFP = true;
    S.GlobalsReadFP.insert(R.Globals.begin(), R.Globals.end());
    S.Unknown |= R.Unknown;
  }
  void write(const Roots &R) {
    for (auto i : R.Args)
      S.Args[i].WriteFP = true;
    S.GlobalsWriteFP.insert(R.Globals.begin(), R.Globals.end());
    S.Unknown |= R.Unknown;
  }
  void escape(const Roots &R) {
    for (auto i : R.Args)
      S.Args[i].Escape = true;
    if (!R.Globals.empty())
      S.Unknown = true;
  }

  void visit(Instruction &I) {
    if (auto *L = dyn_cast<LoadInst>(&I)) {
      if (carriesFloat(L->getType()))
        read(roots(L->getPointerOperand()));
      return;
    }
    if (auto *St = dyn_cast<StoreInst>(&I)) {
      Value *V = St->getValueOperand();
      bool FP = carriesFloat(V->getType());
      if (auto *BC = dyn_cast<BitCastInst>(V))
        FP |= carriesFloat(BC->getSrcTy());
      auto Dest = roots(St->getPointerOperand());
      if (FP) {
        write(Dest);
        flow(sources(V), Dest);
      } else if (V->getType()->isPointerTy() && Dest.nonLocal())
        escape(roots(V));
      return;
    }
    if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
      auto Dest = typed(roots(MT->getDest()));
      auto Src = typed(roots(MT->getSource()));
      write(Dest);
      read(Src);
      flow(Src, Dest);
      return;
    }
    if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      write(typed(roots(MS->getDest())));
      return;
    }
    if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I)) {
      S.Unknown = true;
      return;
    }
    if (auto *R = dyn_cast<ReturnInst>(&I)) {
      if (auto *V = R->getReturnValue()) {
        if (V->getType()->isPointerTy())
          escape(roots(V));
        else
          flow(sources(V), Roots(), /*ToReturn*/ true);
      }
      return;
    }
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || isa<IntrinsicInst>(&I))
      return;
    auto *Callee = calledFunction(*CB);
    if (!Callee) {
      S.Unknown = true;
      return;
    }
    StringRef N = Callee->getName();
    if (N == "free" || startsWith(N, "_FortranAAllocatableDeallocate") ||
        startsWith(N, "_FortranAPointerDeallocate"))
      S.Frees = true;
    if (startsWith(N, "__enzyme") || isAllocationName(N) || N == "free" ||
        isJuliaMarker(N))
      return;
    if (startsWith(N, "julia.call")) {
      S.Unknown = true;
      return;
    }
    if (startsWith(N, "_Fortran")) {
      // The flang runtime: I/O of characters, integers and logicals moves no
      // floating-point data; other transfers may.
      bool NoFP = N.contains("Ascii") || N.contains("Integer") ||
                  N.contains("Logical") || N.contains("Character");
      bool IO = startsWith(N, "_FortranAio");
      bool In = IO && N.contains("Input");
      bool Out = IO && N.contains("Output");
      if (IO && !In && !Out)
        return; // Begin/End/Set...: the statement's control
      for (auto &A : CB->args()) {
        if (!A->getType()->isPointerTy() || NoFP)
          continue;
        // the runtime's C signatures are untyped: declared types decide
        auto R = typed(roots(A));
        if (!Out)
          write(R);
        if (!In)
          read(R);
      }
      return;
    }
    for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
      Value *A = CB->getArgOperand(k);
      if (!A->getType()->isPointerTy())
        continue;
      auto R = roots(A);
      S.Unknown |= R.Unknown;
      for (auto i : R.Args)
        S.Edges.insert({"a" + std::to_string(i), N.str(), k});
      for (auto &G : R.Globals)
        S.Edges.insert({"g" + G, N.str(), k});
    }
  }
};

/// Writes of any type (not only floating-point data), for what may be
/// overwritten after a call: per argument and global written anywhere in the
/// function, and per call site whether the memory each pointer argument
/// points into may be written again afterwards (loops included).
class WriteFacts {
public:
  WriteFacts(Function &F, EnzymeFunctionSummary &S) : F(F), S(S) {
    for (auto &BB : F)
      for (auto &I : BB)
        Writes[&I] = writes(I);
    for (auto &[I, Ks] : Writes)
      for (auto &K : Ks) {
        if (K[0] == 'a')
          S.Args[std::stoul(K.substr(1))].WriteAny = true;
        else if (K[0] == 'g')
          S.GlobalsWriteAny.insert(K.substr(1));
        else if (K == "*")
          S.UnknownWrite = true;
      }
    // blocks reachable from each block's successors
    for (auto &BB : F) {
      SmallVector<BasicBlock *, 8> todo(succ_begin(&BB), succ_end(&BB));
      auto &R = After[&BB];
      while (!todo.empty()) {
        auto *B = todo.pop_back_val();
        if (!R.insert(B).second)
          continue;
        for (auto *Succ : successors(B))
          todo.push_back(Succ);
      }
    }
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB || isa<IntrinsicInst>(&I))
          continue;
        auto *Callee = calledFunction(*CB);
        if (!Callee || startsWith(Callee->getName(), "_Fortran") ||
            startsWith(Callee->getName(), "__enzyme") ||
            startsWith(Callee->getName(), "llvm.") ||
            isJuliaMarker(Callee->getName()))
          continue;
        EnzymeCallSite Site;
        Site.Callee = Callee->getName().str();
        for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
          Value *A = CB->getArgOperand(k);
          EnzymeCallArg Arg;
          if (A->getType()->isPointerTy()) {
            auto Ks = keys(A);
            Arg.Root = Ks.size() == 1 ? *Ks.begin() : "u";
            if (Arg.Root[0] == 'l')
              Arg.Root = "l";
            Arg.WrittenAfter = writtenAfter(*CB, Ks);
          }
          Site.Args.push_back(std::move(Arg));
        }
        S.CallsAt.push_back(std::move(Site));
      }
  }

private:
  Function &F;
  EnzymeFunctionSummary &S;
  std::map<const Instruction *, std::set<std::string>> Writes;
  std::map<const BasicBlock *, SmallPtrSet<BasicBlock *, 8>> After;
  std::map<const Value *, std::set<std::string>> Memo;
  std::map<const Value *, unsigned> LocalIds;

  std::string local(const Value *V) {
    auto Id = LocalIds.emplace(V, LocalIds.size()).first->second;
    return "l" + std::to_string(Id);
  }

  /// Memory a pointer may point into: "a<i>" (reachable from argument i),
  /// "g<name>", "l<n>" (a local object), "u" (unknown).
  std::set<std::string> keys(const Value *V) {
    auto found = Memo.find(V);
    if (found != Memo.end())
      return found->second;
    Memo[V] = {};
    std::set<std::string> K;
    V = V->stripPointerCasts();
    if (auto *GEP = dyn_cast<GEPOperator>(V))
      K = keys(GEP->getPointerOperand());
    else if (auto *A = dyn_cast<Argument>(V))
      K.insert("a" + std::to_string(A->getArgNo()));
    else if (auto *G = dyn_cast<GlobalVariable>(V))
      K.insert("g" + G->getName().str());
    else if (isa<AllocaInst>(V))
      K.insert(local(V));
    else if (isa<Constant>(V))
      ;
    else if (auto *L = dyn_cast<LoadInst>(V))
      K = keys(L->getPointerOperand()); // dereferences collapsed
    else if (auto *P = dyn_cast<PHINode>(V)) {
      for (auto &In : P->incoming_values()) {
        auto Ki = keys(In);
        K.insert(Ki.begin(), Ki.end());
      }
    } else if (auto *Sel = dyn_cast<SelectInst>(V)) {
      K = keys(Sel->getTrueValue());
      auto Kf = keys(Sel->getFalseValue());
      K.insert(Kf.begin(), Kf.end());
    } else if (auto *CB = dyn_cast<CallBase>(V)) {
      auto *Callee = calledFunction(*CB);
      StringRef N = Callee ? Callee->getName() : "";
      if (auto *Through = juliaPassThrough(*CB, N))
        K = keys(Through);
      // fresh memory, and the flang runtime's own handles (I/O cookies)
      else if (Callee &&
               (isAllocationName(N) || startsWith(N, "_Fortran")))
        K.insert(local(V));
      else
        K.insert("u");
    } else
      K.insert("u");
    Memo[V] = K;
    return K;
  }

  std::set<std::string> writes(Instruction &I) {
    std::set<std::string> W;
    auto add = [&](const Value *P) {
      auto K = keys(P);
      if (K.count("u"))
        W.insert("*");
      W.insert(K.begin(), K.end());
    };
    if (auto *St = dyn_cast<StoreInst>(&I))
      add(St->getPointerOperand());
    else if (auto *MI = dyn_cast<MemIntrinsic>(&I))
      add(MI->getDest());
    else if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I))
      add(I.getOperand(0));
    else if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (isa<IntrinsicInst>(&I) || !I.mayWriteToMemory())
        return W;
      auto *Callee = calledFunction(*CB);
      StringRef N = Callee ? Callee->getName() : "";
      if (!Callee)
        W.insert("*");
      if (isJuliaMarker(N))
        return W;
      bool outputIO = startsWith(N, "_FortranAio") && N.contains("Output");
      for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
        Value *A = CB->getArgOperand(k);
        if (!A->getType()->isPointerTy() || outputIO || CB->onlyReadsMemory(k))
          continue;
        add(A);
      }
      // what the callee writes beyond its pointer arguments (globals) is
      // composed by the thin-link step, which knows whether it has IR
    }
    return W;
  }

  bool writtenAfter(CallBase &CB, const std::set<std::string> &Ks) {
    auto hits = [&](const Instruction &I) {
      auto found = Writes.find(&I);
      if (found == Writes.end())
        return false;
      auto &W = found->second;
      if (W.count("*"))
        return true;
      for (auto &K : Ks)
        if (K == "u" || W.count(K))
          return true;
      // a later call may write globals through its own callees; arguments
      // (noalias Fortran dummies) only if it is given the pointer (above)
      if (auto *C = dyn_cast<CallBase>(&I))
        if (!isa<IntrinsicInst>(C) && C->mayWriteToMemory())
          for (auto &K : Ks)
            if (K[0] == 'g')
              return true;
      return false;
    };
    bool seen = false;
    for (auto &I : *CB.getParent()) {
      if (seen && hits(I))
        return true;
      if (&I == &CB)
        seen = true;
    }
    for (auto *B : After[CB.getParent()])
      for (auto &I : *B)
        if (hits(I))
          return true;
    return false;
  }
};

/// One __enzyme_* differentiation call: the differentiated function and the
/// conventions callees' derivatives must be built under.
json::Value summarizeADCall(CallBase &CB, StringRef Mode) {
  Function *Fn = nullptr;
  if (CB.arg_size())
    Fn = dyn_cast<Function>(CB.getArgOperand(0)->stripPointerCasts());
  bool StrongZero = false, RuntimeActivity = false;
  int64_t Width = 1;
  for (unsigned i = 1, e = CB.arg_size(); i < e; ++i) {
    auto *G =
        dyn_cast<GlobalVariable>(CB.getArgOperand(i)->stripPointerCasts());
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

EnzymeFunctionSummary summarizeFunction(Function &F) {
  EnzymeFunctionSummary S;
  S.Linkage = linkageName(F.getLinkage());
  for (auto &I : instructions(F)) {
    ++S.Insts;
    S.TouchesFP |= touchesFloat(I);
    if (isa<MemTransferInst>(&I) || isa<MemSetInst>(&I))
      S.MemTransfer = true;
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      auto *Callee = calledFunction(*CB);
      if (!Callee)
        ++S.IndirectCalls;
      else if (!Callee->isIntrinsic()) {
        S.Calls.insert(Callee->getName().str());
        S.Allocates |= isAllocationName(Callee->getName());
      }
      for (auto &A : CB->args()) {
        if (auto *G = dyn_cast<Function>(A->stripPointerCasts()))
          S.Refs.insert(G->getName().str());
        if (auto *GV = underlyingGlobal(A))
          S.Globals.insert(GV->getName().str());
      }
      continue;
    }
    for (unsigned i = 0, e = I.getNumOperands(); i < e; ++i) {
      Value *Op = I.getOperand(i);
      if (auto *G = dyn_cast<Function>(Op->stripPointerCasts()))
        S.Refs.insert(G->getName().str());
      else if (auto *GV = underlyingGlobal(Op))
        S.Globals.insert(GV->getName().str());
    }
  }
  for (auto &A : F.args())
    S.ArgTypes.push_back(signatureType(F.getAttributes(), A.getArgNo()));
  if (F.getAttributes().getRetAttrs().hasAttribute("enzyme_type"))
    S.RetType = F.getAttributes()
                    .getRetAttrs()
                    .getAttribute("enzyme_type")
                    .getValueAsString()
                    .str();
  S.ReturnsPointer = F.getReturnType()->isPointerTy();
  S.Inactive = F.hasFnAttribute("enzyme_inactive");
  S.NoFree = F.hasFnAttribute(Attribute::NoFree);
  S.NoEscapingAllocation = F.hasFnAttribute("enzyme_no_escaping_allocation");
  ActivityFacts(F, S);
  WriteFacts(F, S);
  return S;
}

json::Object EnzymeFunctionSummary::toJSON() const {
  json::Array JArgs, JWriteAny;
  for (auto &A : Args) {
    JArgs.push_back(json::Object{{"read", A.ReadFP},
                                 {"write", A.WriteFP},
                                 {"escape", A.Escape}});
    JWriteAny.push_back(A.WriteAny);
  }
  // {source: [sinks...]} for the sources whose data reaches some sink;
  // "a<i>" is argument i (its memory, as a sink), "g" any global, "ret" the
  // return value.
  auto sourceName = [&](unsigned s) {
    return s == globalsSource() ? std::string("g") : "a" + std::to_string(s);
  };
  auto sinkName = [&](unsigned t) {
    if (t == returnSink())
      return std::string("ret");
    if (t == globalsSink())
      return std::string("g");
    return "a" + std::to_string(t);
  };
  json::Object JFlow;
  for (unsigned s = 0; s < Flow.size(); ++s) {
    json::Array Sinks;
    for (unsigned t = 0; t < Flow[s].size(); ++t)
      if (Flow[s][t])
        Sinks.push_back(sinkName(t));
    if (!Sinks.empty())
      JFlow[sourceName(s)] = std::move(Sinks);
  }
  json::Array JEdges;
  for (auto &[Root, Callee, Idx] : Edges)
    JEdges.push_back(json::Array{Root, Callee, Idx});
  json::Array JCalls;
  for (auto &Site : CallsAt) {
    json::Array CArgs;
    for (auto &A : Site.Args) {
      if (A.Root.empty())
        CArgs.push_back(nullptr);
      else
        CArgs.push_back(
            json::Object{{"root", A.Root}, {"after", A.WrittenAfter}});
    }
    JCalls.push_back(
        json::Object{{"callee", Site.Callee}, {"args", std::move(CArgs)}});
  }
  json::Array JArgTypes;
  for (auto &T : ArgTypes)
    JArgTypes.push_back(orNull(T));

  json::Object Activity{{"args", std::move(JArgs)},
                        {"globals_read", toArray(GlobalsReadFP)},
                        {"globals_write", toArray(GlobalsWriteFP)},
                        {"flow", std::move(JFlow)},
                        {"edges", std::move(JEdges)},
                        {"unknown", Unknown},
                        {"returns_fp", ReturnsFP},
                        {"frees", Frees},
                        {"args_write_any", std::move(JWriteAny)},
                        {"globals_write_any", toArray(GlobalsWriteAny)},
                        {"unknown_write", UnknownWrite},
                        {"calls_at", std::move(JCalls)}};
  return json::Object{
      {"linkage", Linkage},
      {"insts", Insts},
      {"calls", toArray(Calls)},
      {"refs", toArray(Refs)},
      {"indirect_calls", IndirectCalls},
      {"globals", toArray(Globals)},
      {"fp", TouchesFP},
      {"memtransfer", MemTransfer},
      {"allocates", Allocates},
      {"returns_pointer", ReturnsPointer},
      {"arg_types", std::move(JArgTypes)},
      {"ret_type", orNull(RetType)},
      {"inactive", Inactive},
      {"nofree", NoFree},
      {"no_escape", NoEscapingAllocation},
      {"activity", std::move(Activity)},
  };
}

llvm::AnalysisKey EnzymeFunctionSummaryAnalysis::Key;

EnzymeFunctionSummary
EnzymeFunctionSummaryAnalysis::run(Function &F, FunctionAnalysisManager &) {
  return summarizeFunction(F);
}

json::Object summarizeModule(Module &M) {
  json::Object Functions, Globals;
  json::Array ADCalls;
  std::map<std::string, std::set<std::string>> Registrations;

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    Functions[F.getName()] = summarizeFunction(F).toJSON();
    for (auto &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (auto *Callee = calledFunction(*CB)) {
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
    if (startsWith(G.getName(), "llvm."))
      continue;
    Globals[G.getName()] = json::Object{
        {"linkage", linkageName(G.getLinkage())},
        {"defined", !G.isDeclaration()},
        {"constant", G.isConstant()},
        {"size", (int64_t)M.getDataLayout().getTypeAllocSize(G.getValueType())},
    };
  }

  json::Object Regs;
  for (auto &[K, S] : Registrations)
    Regs[K] = toArray(S);

  return json::Object{
      {"module", M.getModuleIdentifier()}, {"functions", std::move(Functions)},
      {"ad_calls", std::move(ADCalls)},    {"registrations", std::move(Regs)},
      {"globals", std::move(Globals)},
  };
}

llvm::AnalysisKey EnzymeSummaryNewPM::Key;

PreservedAnalyses EnzymeSummaryNewPM::run(Module &M, ModuleAnalysisManager &) {
  json::Value Out = summarizeModule(M);
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

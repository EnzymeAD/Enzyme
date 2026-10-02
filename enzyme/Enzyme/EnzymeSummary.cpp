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

#include "ActivityAnalysis.h"
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
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;

static cl::opt<std::string>
    EnzymeSummaryOut("enzyme-summary-out", cl::init(""), cl::Hidden,
                     cl::desc("File the enzyme-summary pass writes the "
                              "module's facts to (JSON); stdout if empty"));

namespace {

/// Function a constant (e.g. a registration's initializer) points to: a
/// function, or a global variable declaration naming one (C code registering a
/// Fortran procedure it declares as "extern char sym").
GlobalValue *pointee(Constant *C) {
  if (!C)
    return nullptr;
  auto *V = C->stripPointerCasts();
  if (auto *F = dyn_cast<Function>(V))
    return F;
  if (auto *G = dyn_cast<GlobalVariable>(V))
    if (G->isDeclaration())
      return G;
  return nullptr;
}

/// The primal function of a registration global: the pointer itself, or the
/// first field of a {primal, ...} aggregate.
GlobalValue *registeredPrimal(GlobalVariable &G) {
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
         startsWith(N, "_FortranAPointerAllocate");
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
/// floating-point data there (a store of a constant counts: it overwrites a
/// value that may have a derivative), and symbolic edges for what callees
/// do with it.
class ActivityFacts {
public:
  ActivityFacts(Function &F) : F(F) {
    ArgRead.assign(F.arg_size(), false);
    ArgWrite.assign(F.arg_size(), false);
    ArgEscape.assign(F.arg_size(), false);
    for (auto &A : F.args()) {
      // An argument whose declared type has no floating-point part cannot
      // carry derivatives through untyped copies.
      if (F.getAttributes().hasParamAttr(A.getArgNo(), "enzyme_type")) {
        auto T = F.getAttributes()
                     .getParamAttr(A.getArgNo(), "enzyme_type")
                     .getValueAsString();
        if (!T.contains("Float"))
          IntTyped.insert(A.getArgNo());
      }
    }
    computeLocalContents();
    for (auto &I : instructions(F))
      visit(I);
    // Requests of the Fortran MPI ABI are active (their shadows carry the
    // bookkeeping of the nonblocking calls), though they are integers.
    for (auto &A : F.args())
      if (A.getType()->isPointerTy() && isFortranMPIRequest(&A)) {
        ArgRead[A.getArgNo()] = true;
        ArgWrite[A.getArgNo()] = true;
      }
  }

  json::Object toJSON() const {
    json::Array Args;
    for (unsigned i = 0; i < ArgRead.size(); ++i)
      Args.push_back(json::Object{{"read", (bool)ArgRead[i]},
                                  {"write", (bool)ArgWrite[i]},
                                  {"escape", (bool)ArgEscape[i]}});
    json::Array Edges;
    for (auto &[Root, Callee, Idx] : Edges_)
      Edges.push_back(json::Array{Root, Callee, Idx});
    return json::Object{{"args", std::move(Args)},
                        {"globals_read", toArray(GlobalRead)},
                        {"globals_write", toArray(GlobalWrite)},
                        {"edges", std::move(Edges)},
                        {"unknown", Unknown},
                        {"returns_fp", F.getReturnType()->isFPOrFPVectorTy()},
                        {"frees", Frees}};
  }

private:
  Function &F;
  std::vector<bool> ArgRead, ArgWrite, ArgEscape;
  std::set<unsigned> IntTyped;
  std::set<std::string> GlobalRead, GlobalWrite;
  // (root: "a<i>" or "g<name>", callee, parameter index)
  std::set<std::tuple<std::string, std::string, unsigned>> Edges_;
  bool Unknown = false, Frees = false;
  std::map<const Value *, Roots> Memo;
  // What the pointers stored into a local object (alloca) may point to, e.g.
  // the base address flang copies from an argument's descriptor into the
  // descriptor of a reboxed array: loads from the local object yield them.
  std::map<const AllocaInst *, Roots> LocalContents;

  static const AllocaInst *localBase(const Value *V) {
    V = V->stripPointerCasts();
    while (auto *GEP = dyn_cast<GEPOperator>(V))
      V = GEP->getPointerOperand()->stripPointerCasts();
    return dyn_cast<AllocaInst>(V);
  }

  /// LocalContents to a fixpoint: pointers stored or copied into a local
  /// object, and for calls that take a local object, the other pointer
  /// arguments (which the callee may store into it, e.g.
  /// _FortranAPointerAssociate on a local descriptor).
  void computeLocalContents() {
    bool Changed = true;
    auto add = [&](const AllocaInst *AI, const Roots &R) {
      auto &C = LocalContents[AI];
      auto Before = std::make_tuple(C.Args.size(), C.Globals.size(), C.Unknown);
      C.merge(R);
      if (Before != std::make_tuple(C.Args.size(), C.Globals.size(), C.Unknown))
        Changed = true;
    };
    while (Changed) {
      Changed = false;
      Memo.clear();
      for (auto &I : instructions(F)) {
        if (auto *S = dyn_cast<StoreInst>(&I)) {
          if (!S->getValueOperand()->getType()->isPointerTy())
            continue;
          if (auto *AI = localBase(S->getPointerOperand()))
            add(AI, roots(S->getValueOperand()));
        } else if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
          if (auto *AI = localBase(MT->getDest()))
            add(AI, roots(MT->getSource()));
        } else if (auto *CB = dyn_cast<CallBase>(&I)) {
          if (isa<IntrinsicInst>(CB))
            continue;
          Roots Others;
          SmallVector<const AllocaInst *, 2> Locals;
          for (auto &A : CB->args()) {
            if (!A->getType()->isPointerTy())
              continue;
            if (auto *AI = localBase(A))
              Locals.push_back(AI);
            Others.merge(roots(A));
          }
          for (auto *AI : Locals)
            add(AI, Others);
        }
      }
    }
    Memo.clear();
  }

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
    } else if (auto *AI = dyn_cast<AllocaInst>(V)) {
      auto found = LocalContents.find(AI);
      if (found != LocalContents.end())
        R = found->second;
    } else if (isa<Constant>(V))
      ;
    else if (auto *L = dyn_cast<LoadInst>(V))
      R = roots(L->getPointerOperand()); // dereferences collapsed
    else if (auto *P = dyn_cast<PHINode>(V)) {
      for (auto &In : P->incoming_values())
        R.merge(roots(In));
    } else if (auto *S = dyn_cast<SelectInst>(V)) {
      R = roots(S->getTrueValue());
      R.merge(roots(S->getFalseValue()));
    } else if (auto *CB = dyn_cast<CallBase>(V)) {
      auto *Callee =
          dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
      StringRef N = Callee ? Callee->getName() : "";
      // fresh memory, and the flang runtime's own handles (I/O cookies)
      if (!(isAllocationName(N) || startsWith(N, "_Fortran")))
        R.Unknown = true;
    } else
      R.Unknown = true;
    Memo[V] = R;
    return R;
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
          if (auto *S = dyn_cast_or_null<MDString>(Op.get()))
            if (S->getString().contains("Float"))
              return false;
          if (auto *Sub = dyn_cast_or_null<MDNode>(Op.get()))
            todo.push_back(Sub);
        }
      }
      return true;
    }
    return G->hasLocalLinkage() && !carriesFloat(G->getValueType());
  }
  void read(const Roots &R, bool Untyped = false) {
    for (auto i : R.Args)
      if (!(Untyped && IntTyped.count(i)))
        ArgRead[i] = true;
    for (auto &G : R.Globals)
      if (!(Untyped && intTypedGlobal(G)))
        GlobalRead.insert(G);
    Unknown |= R.Unknown;
  }
  void write(const Roots &R, bool Untyped = false) {
    for (auto i : R.Args)
      if (!(Untyped && IntTyped.count(i)))
        ArgWrite[i] = true;
    for (auto &G : R.Globals)
      if (!(Untyped && intTypedGlobal(G)))
        GlobalWrite.insert(G);
    Unknown |= R.Unknown;
  }
  void escape(const Roots &R) {
    for (auto i : R.Args)
      ArgEscape[i] = true;
    if (!R.Globals.empty())
      Unknown = true;
  }

  static bool carriesFloat(Type *T) {
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

  void visit(Instruction &I) {
    // Accesses to a local object itself (not through a pointer loaded from
    // it) touch local memory only.
    if (auto *L = dyn_cast<LoadInst>(&I)) {
      if (carriesFloat(L->getType()) && !localBase(L->getPointerOperand()))
        read(roots(L->getPointerOperand()));
      return;
    }
    if (auto *S = dyn_cast<StoreInst>(&I)) {
      Value *V = S->getValueOperand();
      bool FP = carriesFloat(V->getType());
      if (auto *BC = dyn_cast<BitCastInst>(V))
        FP |= carriesFloat(BC->getSrcTy());
      if (localBase(S->getPointerOperand()))
        return;
      if (FP)
        write(roots(S->getPointerOperand()));
      else if (V->getType()->isPointerTy() &&
               roots(S->getPointerOperand()).nonLocal())
        escape(roots(V));
      return;
    }
    if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
      if (!localBase(MT->getDest()))
        write(roots(MT->getDest()), /*Untyped*/ true);
      if (!localBase(MT->getSource()))
        read(roots(MT->getSource()), /*Untyped*/ true);
      return;
    }
    if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      if (!localBase(MS->getDest()))
        write(roots(MS->getDest()), /*Untyped*/ true);
      return;
    }
    if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I)) {
      Unknown = true;
      return;
    }
    if (auto *R = dyn_cast<ReturnInst>(&I)) {
      if (auto *V = R->getReturnValue())
        if (V->getType()->isPointerTy())
          escape(roots(V));
      return;
    }
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || isa<IntrinsicInst>(&I))
      return;
    auto *Callee =
        dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
    if (!Callee) {
      Unknown = true;
      return;
    }
    StringRef N = Callee->getName();
    if (N == "free" || startsWith(N, "_FortranAAllocatableDeallocate") ||
        startsWith(N, "_FortranAPointerDeallocate"))
      Frees = true;
    if (startsWith(N, "__enzyme") || isAllocationName(N) || N == "free")
      return;
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
        auto R = roots(A);
        if (!Out)
          write(R, /*Untyped*/ true);
        if (!In)
          read(R, /*Untyped*/ true);
      }
      return;
    }
    for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
      Value *A = CB->getArgOperand(k);
      if (!A->getType()->isPointerTy())
        continue;
      auto R = roots(A);
      Unknown |= R.Unknown;
      for (auto i : R.Args)
        Edges_.insert({"a" + std::to_string(i), N.str(), k});
      for (auto &G : R.Globals)
        Edges_.insert({"g" + G, N.str(), k});
    }
  }
};

/// Writes of any type (not only floating-point data), for what may be
/// overwritten after a call: per argument and global written anywhere in the
/// function, and per call site whether the memory each pointer argument
/// points into may be written again afterwards (loops included).
class WriteFacts {
public:
  WriteFacts(Function &F) : F(F) {
    ArgWriteAny.assign(F.arg_size(), false);
    for (auto &BB : F)
      for (auto &I : BB)
        Writes[&I] = writes(I);
    for (auto &[I, Ks] : Writes)
      for (auto &K : Ks) {
        if (K[0] == 'a')
          ArgWriteAny[std::stoul(K.substr(1))] = true;
        else if (K[0] == 'g')
          GlobalWriteAny.insert(K.substr(1));
        else if (K == "*")
          UnknownWrite = true;
      }
    // blocks reachable from each block's successors
    for (auto &BB : F) {
      SmallVector<BasicBlock *, 8> todo(succ_begin(&BB), succ_end(&BB));
      auto &R = After[&BB];
      while (!todo.empty()) {
        auto *B = todo.pop_back_val();
        if (!R.insert(B).second)
          continue;
        for (auto *S : successors(B))
          todo.push_back(S);
      }
    }
  }

  void addTo(json::Object &Act) {
    json::Array Calls;
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB || isa<IntrinsicInst>(&I))
          continue;
        auto *Callee =
            dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
        if (!Callee || startsWith(Callee->getName(), "_Fortran") ||
            startsWith(Callee->getName(), "__enzyme") ||
            startsWith(Callee->getName(), "llvm."))
          continue;
        json::Array Args;
        for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
          Value *A = CB->getArgOperand(k);
          if (!A->getType()->isPointerTy()) {
            Args.push_back(nullptr);
            continue;
          }
          auto Ks = keys(A);
          std::string Root = Ks.size() == 1 ? *Ks.begin() : "u";
          if (Root[0] == 'l')
            Root = "l";
          Args.push_back(
              json::Object{{"root", Root}, {"after", writtenAfter(*CB, Ks)}});
        }
        Calls.push_back(json::Object{{"callee", Callee->getName().str()},
                                     {"args", std::move(Args)}});
      }
    json::Array WA;
    for (bool b : ArgWriteAny)
      WA.push_back(b);
    Act["args_write_any"] = std::move(WA);
    Act["globals_write_any"] = toArray(GlobalWriteAny);
    Act["unknown_write"] = UnknownWrite;
    Act["calls_at"] = std::move(Calls);
  }

private:
  Function &F;
  std::vector<bool> ArgWriteAny;
  std::set<std::string> GlobalWriteAny;
  bool UnknownWrite = false;
  std::map<const Instruction *, std::set<std::string>> Writes;
  std::map<const BasicBlock *, SmallPtrSet<BasicBlock *, 8>> After;
  std::map<const Value *, std::set<std::string>> Memo;

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
      K.insert("l" + std::to_string((uintptr_t)V));
    else if (isa<Constant>(V))
      ;
    else if (auto *L = dyn_cast<LoadInst>(V))
      K = keys(L->getPointerOperand()); // dereferences collapsed
    else if (auto *P = dyn_cast<PHINode>(V)) {
      for (auto &In : P->incoming_values()) {
        auto Ki = keys(In);
        K.insert(Ki.begin(), Ki.end());
      }
    } else if (auto *S = dyn_cast<SelectInst>(V)) {
      K = keys(S->getTrueValue());
      auto Kf = keys(S->getFalseValue());
      K.insert(Kf.begin(), Kf.end());
    } else if (auto *CB = dyn_cast<CallBase>(V)) {
      auto *Callee =
          dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
      // fresh memory, and the flang runtime's own handles (I/O cookies)
      if (Callee && (isAllocationName(Callee->getName()) ||
                     startsWith(Callee->getName(), "_Fortran")))
        K.insert("l" + std::to_string((uintptr_t)V));
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
    if (auto *S = dyn_cast<StoreInst>(&I))
      add(S->getPointerOperand());
    else if (auto *MI = dyn_cast<MemIntrinsic>(&I))
      add(MI->getDest());
    else if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I))
      add(I.getOperand(0));
    else if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (isa<IntrinsicInst>(&I) || !I.mayWriteToMemory())
        return W;
      auto *Callee =
          dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
      StringRef N = Callee ? Callee->getName() : "";
      if (!Callee)
        W.insert("*");
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
      {"activity",
       [&] {
         auto A = ActivityFacts(F).toJSON();
         WriteFacts(F).addTo(A);
         return A;
       }()},
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

llvm::AnalysisKey EnzymeSummaryNewPM::Key;

PreservedAnalyses EnzymeSummaryNewPM::run(Module &M, ModuleAnalysisManager &) {
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

  json::Value Out = json::Object{
      {"module", M.getModuleIdentifier()}, {"functions", std::move(Functions)},
      {"ad_calls", std::move(ADCalls)},    {"registrations", std::move(Regs)},
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

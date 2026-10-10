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
         N == "julia.gc_alloc_obj" || N == "julia.gc_alloc_bytes" ||
         N.contains("gc_pool_alloc") || N.contains("gc_small_alloc") ||
         N.contains("gc_big_alloc") || N.contains("alloc_genericmemory") ||
         N.contains("alloc_array");
}

/// Julia's codegen markers, which neither move data nor call user code.
/// julia.call and julia.call2 (calls through the generic entry point) are not
/// among them.
bool isJuliaMarker(StringRef N) {
  return startsWith(N, "julia.") && !startsWith(N, "julia.call") &&
         !isAllocationName(N);
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

/// The name standing for a global that cannot be named (a constant
/// inttoptr address, as Julia emits for its objects) and for memory a callee
/// may have allocated and handed back.
const char *const AnyGlobal = "*";

/// The type may hold pointers (an aggregate as well as a pointer).
bool hasPointer(Type *T) {
  if (T->isPtrOrPtrVectorTy())
    return true;
  if (auto *ST = dyn_cast<StructType>(T)) {
    for (auto *E : ST->elements())
      if (hasPointer(E))
        return true;
    return false;
  }
  if (auto *AT = dyn_cast<ArrayType>(T))
    return hasPointer(AT->getElementType());
  return false;
}

/// Where a pointer may point: into the memory reachable from an argument or
/// a global (dereferences collapsed, as in ActivityAnalysis.jl's pseudo
/// classes), into a local object (an alloca or a fresh allocation, tracked
/// per object), or anywhere. As a set of data sources (see DataFlow), the
/// same sets name where the data was loaded from.
struct Roots {
  std::set<unsigned> Args;
  std::set<std::string> Globals;
  std::set<const Value *> Locals;
  bool Unknown = false;
  /// Returns whether anything was added.
  bool merge(const Roots &O) {
    size_t Before = Args.size() + Globals.size() + Locals.size();
    bool WasUnknown = Unknown;
    Args.insert(O.Args.begin(), O.Args.end());
    Globals.insert(O.Globals.begin(), O.Globals.end());
    Locals.insert(O.Locals.begin(), O.Locals.end());
    Unknown |= O.Unknown;
    return Unknown != WasUnknown ||
           Args.size() + Globals.size() + Locals.size() != Before;
  }
  bool nonLocal() const { return Unknown || !Args.empty() || !Globals.empty(); }
  Roots withoutLocals() const {
    Roots R = *this;
    R.Locals.clear();
    return R;
  }
};

/// Calls the summary treats as their own kind of effect, rather than as a
/// call whose effects are composed from the callee's summary later.
bool isFreshPointer(StringRef N) {
  // fresh memory, the flang runtime's own handles (I/O cookies), and Julia's
  // markers returning task-local state (julia.get_pgcstack)
  return isAllocationName(N) || startsWith(N, "_Fortran") || isJuliaMarker(N);
}

/// Where each pointer in a function may point. Pointers stored into a local
/// object are tracked per object, so a pointer that goes through a stack
/// slot or a buffer a callee fills (a Julia sret) keeps its roots.
/// Computed as a fixpoint over the whole function, so cycles through phis
/// and through memory are followed.
class Provenance {
public:
  explicit Provenance(Function &F) {
    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (auto &I : instructions(F)) {
        if (hasPointer(I.getType()))
          Changed |= State[&I].merge(eval(I));
        Changed |= update(I);
      }
    }
  }

  /// The roots of a pointer, or of the pointers in an aggregate.
  Roots roots(const Value *V) const {
    Roots R;
    V = V->stripPointerCasts();
    if (auto *A = dyn_cast<Argument>(V)) {
      // Julia passes its task state as the swiftself parameter
      if (!A->hasSwiftSelfAttr())
        R.Args.insert(A->getArgNo());
    } else if (auto *G = dyn_cast<GlobalVariable>(V)) {
      if (!G->isConstant())
        R.Globals.insert(G->getName().str());
    } else if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      if (CE->getOpcode() == Instruction::IntToPtr)
        R.Globals.insert(AnyGlobal);
      else
        for (auto &Op : CE->operands())
          R.merge(roots(Op));
    } else if (auto *C = dyn_cast<ConstantAggregate>(V)) {
      for (auto &Op : C->operands())
        R.merge(roots(Op));
    } else if (isa<Constant>(V) || isa<MetadataAsValue>(V)) {
    } else if (auto *I = dyn_cast<Instruction>(V)) {
      auto found = State.find(I);
      if (found != State.end())
        R = found->second;
    } else
      R.Unknown = true;
    return R;
  }

  /// R and what the local objects in it point to, transitively: all the
  /// memory reachable from a pointer with roots R.
  Roots deep(const Roots &R) const {
    Roots Out = R;
    SmallVector<const Value *, 8> todo(R.Locals.begin(), R.Locals.end());
    while (!todo.empty()) {
      auto found = Contents.find(todo.pop_back_val());
      if (found == Contents.end())
        continue;
      for (auto *L : found->second.Locals)
        if (!Out.Locals.count(L))
          todo.push_back(L);
      Out.merge(found->second);
    }
    return Out;
  }

  /// The roots of a pointer loaded through a pointer with roots R.
  Roots loaded(const Roots &R) const {
    Roots Out = R.withoutLocals(); // dereferences collapsed
    for (auto *L : R.Locals) {
      auto found = Contents.find(L);
      if (found != Contents.end())
        Out.merge(found->second);
    }
    return Out;
  }

  /// What a call to a function the summary does not look into may return,
  /// or store into the memory it is given: a pointer into memory it can
  /// reach from its arguments, a global, or memory it allocated.
  Roots calleeResult(const CallBase &CB) const {
    Roots R;
    for (auto &A : CB.args())
      if (hasPointer(A->getType()))
        R.merge(deep(roots(A)));
    R.Globals.insert(AnyGlobal);
    return R;
  }

private:
  std::map<const Instruction *, Roots> State;

  /// The roots of a pointer computed as an integer: the pointers converted
  /// with ptrtoint in it, other operands taken as offsets. Found is set if
  /// there are any.
  Roots intRoots(const Value *V, bool &Found, unsigned Depth = 0) const {
    Roots R;
    if (auto *PI = dyn_cast<PtrToIntOperator>(V)) {
      Found = true;
      return roots(PI->getPointerOperand());
    }
    if (isa<Constant>(V))
      return R;
    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
      if (Depth < 8) {
        for (auto &Op : BO->operands())
          R.merge(intRoots(Op, Found, Depth + 1));
        return R;
      }
    } else if (auto *C = dyn_cast<CastInst>(V)) {
      if (C->getSrcTy()->isIntegerTy() && Depth < 8)
        return intRoots(C->getOperand(0), Found, Depth + 1);
    }
    // an integer from memory or elsewhere: maybe an address
    Found = true;
    R.Unknown = true;
    return R;
  }

  /// Local object -> roots of the pointers stored into it.
  std::map<const Value *, Roots> Contents;

  Roots eval(Instruction &I) {
    Roots R;
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
      return roots(GEP->getPointerOperand());
    if (isa<BitCastInst>(&I) || isa<AddrSpaceCastInst>(&I) ||
        isa<FreezeInst>(&I))
      return roots(I.getOperand(0));
    if (isa<AllocaInst>(&I)) {
      R.Locals.insert(&I);
      return R;
    }
    if (auto *L = dyn_cast<LoadInst>(&I))
      return loaded(roots(L->getPointerOperand()));
    if (auto *IP = dyn_cast<IntToPtrInst>(&I)) {
      // Julia round-trips pointers through ptrtoint/inttoptr; a constant
      // address is a global nothing here names
      bool Found = false;
      R = intRoots(IP->getOperand(0), Found);
      if (!Found)
        R.Globals.insert(AnyGlobal);
      return R;
    }
    if (auto *Sel = dyn_cast<SelectInst>(&I)) {
      R = roots(Sel->getTrueValue());
      R.merge(roots(Sel->getFalseValue()));
      return R;
    }
    if (isa<PHINode>(&I) || isa<ExtractValueInst>(&I) ||
        isa<InsertValueInst>(&I) || isa<ExtractElementInst>(&I) ||
        isa<InsertElementInst>(&I) || isa<ShuffleVectorInst>(&I)) {
      for (auto &Op : I.operands())
        if (hasPointer(Op->getType()))
          R.merge(roots(Op));
      return R;
    }
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (isa<IntrinsicInst>(CB)) {
        // llvm.ptrmask, llvm.launder.invariant.group, ...
        for (auto &A : CB->args())
          if (hasPointer(A->getType()))
            R.merge(roots(A));
        return R;
      }
      auto *Callee = calledFunction(*CB);
      StringRef N = Callee ? Callee->getName() : "";
      if (auto *Through = juliaPassThrough(*CB, N))
        return roots(Through);
      // task state (julia.get_pgcstack) and type tags hold no user data
      if (Callee && isJuliaMarker(N))
        return R;
      if (Callee && isFreshPointer(N)) {
        R.Locals.insert(&I);
        return R;
      }
      return calleeResult(*CB);
    }
    // inttoptr, pointers of unknown origin
    R.Unknown = true;
    return R;
  }

  /// Pointers stored into local objects.
  bool update(Instruction &I) {
    bool Changed = false;
    auto into = [&](const Roots &Dest, const Roots &R) {
      for (auto *L : Dest.Locals)
        Changed |= Contents[L].merge(R);
    };
    if (auto *St = dyn_cast<StoreInst>(&I)) {
      if (hasPointer(St->getValueOperand()->getType()))
        into(roots(St->getPointerOperand()), roots(St->getValueOperand()));
    } else if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
      into(roots(MT->getDest()), loaded(roots(MT->getSource())));
    } else if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (isa<IntrinsicInst>(CB) || CB->onlyReadsMemory())
        return false;
      auto *Callee = calledFunction(*CB);
      StringRef N = Callee ? Callee->getName() : "";
      // allocations do not store into what they are given, except
      // posix_memalign
      if (Callee && (isJuliaMarker(N) || startsWith(N, "__enzyme") ||
                     (isAllocationName(N) && N != "posix_memalign")))
        return false;
      Roots Out;
      if (Callee && isFreshPointer(N))
        Out.Locals.insert(CB); // e.g. posix_memalign, flang descriptors
      else
        Out = calleeResult(*CB);
      for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
        Value *A = CB->getArgOperand(k);
        if (hasPointer(A->getType()) && !CB->onlyReadsMemory(k))
          into(deep(roots(A)), Out);
      }
    }
    return Changed;
  }
};

/// Effects on floating-point data of one function, before composing with
/// its callees: per argument and per global, whether it may read or write
/// floating-point data there, where that data flows, and symbolic edges for
/// what callees do with it.
class ActivityFacts {
public:
  ActivityFacts(Function &F, EnzymeFunctionSummary &S, const Provenance &P)
      : F(F), S(S), P(P) {
    unsigned N = F.arg_size();
    S.Args.assign(N, EnzymeArgEffects());
    S.Flow.assign(N + 1, std::vector<bool>(N + 2, false));
    S.PointsTo.assign(N + 1, std::vector<bool>(N + 2, false));
    for (auto &A : F.args()) {
      // An argument whose declared type has no floating-point part cannot
      // carry derivatives through untyped copies.
      auto T = signatureType(F.getAttributes(), A.getArgNo());
      if (!T.empty() && !StringRef(T).contains("Float"))
        IntTyped.insert(A.getArgNo());
    }
    // where the data in each value comes from, through local memory
    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (auto &I : instructions(F)) {
        if (!I.getType()->isPtrOrPtrVectorTy() && !I.getType()->isVoidTy())
          Changed |= Sources[&I].merge(evalSources(I));
        Changed |= update(I);
      }
    }
    for (auto &I : instructions(F))
      visit(I);
    for (unsigned i = 0; i < N; ++i)
      for (unsigned t = 0; t < N + 2; ++t)
        if (t != i && S.PointsTo[i][t])
          S.Args[i].Escape = true;
    S.ReturnsFP = F.getReturnType()->isFPOrFPVectorTy();
  }

private:
  Function &F;
  EnzymeFunctionSummary &S;
  const Provenance &P;
  std::set<unsigned> IntTyped;
  /// Value -> where its data may come from: "a<i>" an argument passed by
  /// value or memory reachable from it, a global's memory; never a local
  /// object (its contents are substituted).
  std::map<const Instruction *, Roots> Sources;
  /// Local object -> where the data stored into it may come from.
  std::map<const Value *, Roots> Contents;

  /// Where the data in a value may come from. Addresses are not data:
  /// pointers only contribute through the loads that read through them.
  Roots sources(const Value *V) const {
    Roots R;
    if (V->getType()->isPtrOrPtrVectorTy())
      return R;
    if (auto *A = dyn_cast<Argument>(V))
      R.Args.insert(A->getArgNo());
    else if (auto *I = dyn_cast<Instruction>(V)) {
      auto found = Sources.find(I);
      if (found != Sources.end())
        R = found->second;
    }
    return R;
  }

  /// The data in the memory a pointer with roots R points to directly.
  Roots stored(const Roots &R) const {
    Roots Out = R.withoutLocals();
    for (auto *L : R.Locals) {
      auto found = Contents.find(L);
      if (found != Contents.end())
        Out.merge(found->second);
    }
    return Out;
  }

  /// The data a call may compute its result, or what it stores, from: what
  /// it is passed, the memory reachable from that, and globals.
  Roots callInputs(const CallBase &CB) const {
    Roots R;
    for (auto &A : CB.args()) {
      R.merge(sources(A));
      if (hasPointer(A->getType()))
        R.merge(stored(P.deep(P.roots(A))));
    }
    auto *Callee = calledFunction(CB);
    StringRef N = Callee ? Callee->getName() : "";
    if (!CB.doesNotAccessMemory() &&
        !(Callee && (isFreshPointer(N) || startsWith(N, "__enzyme"))))
      R.Globals.insert(AnyGlobal);
    return R;
  }

  Roots evalSources(Instruction &I) const {
    if (auto *L = dyn_cast<LoadInst>(&I))
      return stored(P.roots(L->getPointerOperand()));
    auto *CB = dyn_cast<CallBase>(&I);
    if (CB && !isa<IntrinsicInst>(CB))
      return callInputs(*CB);
    Roots R;
    for (auto &Op : I.operands()) {
      if (isa<BasicBlock>(Op))
        continue;
      R.merge(sources(Op));
      // intrinsics reading memory (llvm.masked.load, ...)
      if (CB && CB->mayReadFromMemory() && Op->getType()->isPointerTy())
        R.merge(stored(P.roots(Op)));
    }
    return R;
  }

  /// Data stored into local objects.
  bool update(Instruction &I) {
    bool Changed = false;
    auto into = [&](const Roots &Dest, const Roots &R) {
      for (auto *L : Dest.Locals)
        Changed |= Contents[L].merge(R);
    };
    if (auto *St = dyn_cast<StoreInst>(&I)) {
      into(P.roots(St->getPointerOperand()), sources(St->getValueOperand()));
    } else if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
      into(P.roots(MT->getDest()), stored(P.roots(MT->getSource())));
    } else if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (isa<IntrinsicInst>(CB) || CB->onlyReadsMemory())
        return false;
      auto *Callee = calledFunction(*CB);
      StringRef N = Callee ? Callee->getName() : "";
      // fresh memory holds no data yet; what the flang runtime reads into
      // local variables (I/O) is not derived from anything
      if (Callee && (isFreshPointer(N) || startsWith(N, "__enzyme")))
        return false;
      auto In = callInputs(*CB);
      for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
        Value *A = CB->getArgOperand(k);
        if (hasPointer(A->getType()) && !CB->onlyReadsMemory(k))
          into(P.deep(P.roots(A)), In);
      }
    }
    return Changed;
  }

  /// Marks, for each source s in From and sink t in To, that s may reach t
  /// in M (Flow or PointsTo).
  void relate(std::vector<std::vector<bool>> &M, const Roots &From,
              const Roots &To, bool ToReturn) {
    auto row = [&](unsigned Src) {
      for (auto i : To.Args)
        M[Src][i] = true;
      if (!To.Globals.empty())
        M[Src][S.globalsSink()] = true;
      if (ToReturn)
        M[Src][S.returnSink()] = true;
    };
    for (auto i : From.Args)
      row(i);
    if (!From.Globals.empty())
      row(S.globalsSource());
    S.Unknown |= From.Unknown || To.Unknown;
  }
  void flow(const Roots &From, const Roots &To, bool ToReturn = false) {
    relate(S.Flow, From, To, ToReturn);
  }

  /// A pointer with roots R becomes reachable from To: the memory it
  /// reaches now aliases To's, and the data in its local objects flows
  /// there.
  void publish(const Roots &R, const Roots &To, bool ToReturn = false) {
    auto D = P.deep(R);
    relate(S.PointsTo, D.withoutLocals(), To, ToReturn);
    flow(stored(D), To, ToReturn);
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

  /// A call to a callee that is not known (an indirect call, or Julia's
  /// dynamic julia.call): it may read, write and free everything reachable
  /// from what it is given and any global, and move data and pointers
  /// among them.
  void opaqueCall(CallBase &CB) {
    Roots In, Sinks;
    In.Globals.insert(AnyGlobal);
    Sinks.Globals.insert(AnyGlobal);
    for (auto &A : CB.args()) {
      In.merge(sources(A));
      if (!hasPointer(A->getType()))
        continue;
      auto R = P.deep(P.roots(A));
      In.merge(stored(R));
      Sinks.merge(R.withoutLocals());
    }
    read(Sinks);
    write(Sinks);
    flow(In, Sinks);
    relate(S.PointsTo, Sinks, Sinks, /*ToReturn*/ false);
    S.Frees = true;
  }

  void visit(Instruction &I) {
    if (auto *L = dyn_cast<LoadInst>(&I)) {
      if (carriesFloat(L->getType()))
        read(P.roots(L->getPointerOperand()).withoutLocals());
      return;
    }
    if (auto *St = dyn_cast<StoreInst>(&I)) {
      Value *V = St->getValueOperand();
      bool FP = carriesFloat(V->getType());
      if (auto *BC = dyn_cast<BitCastInst>(V))
        FP |= carriesFloat(BC->getSrcTy());
      auto Dest = P.roots(St->getPointerOperand()).withoutLocals();
      if (FP) {
        write(Dest);
        flow(sources(V), Dest);
      }
      if (hasPointer(V->getType()) && Dest.nonLocal())
        publish(P.roots(V), Dest);
      return;
    }
    if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
      auto Src = P.roots(MT->getSource());
      auto Dest = typed(P.roots(MT->getDest()).withoutLocals());
      write(Dest);
      read(typed(Src.withoutLocals()));
      flow(typed(stored(Src)), Dest);
      if (Dest.nonLocal())
        publish(P.loaded(Src), Dest);
      return;
    }
    if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      write(typed(P.roots(MS->getDest()).withoutLocals()));
      return;
    }
    if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I)) {
      S.Unknown = true;
      return;
    }
    if (auto *R = dyn_cast<ReturnInst>(&I)) {
      if (auto *V = R->getReturnValue()) {
        flow(sources(V), Roots(), /*ToReturn*/ true);
        if (hasPointer(V->getType()))
          publish(P.roots(V), Roots(), /*ToReturn*/ true);
      }
      return;
    }
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || isa<IntrinsicInst>(&I))
      return;
    auto *Callee = calledFunction(*CB);
    if (!Callee) {
      opaqueCall(*CB);
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
      opaqueCall(*CB);
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
        auto R = typed(P.deep(P.roots(A)).withoutLocals());
        if (!Out)
          write(R);
        if (!In)
          read(R);
      }
      return;
    }
    // What the callee does with the memory it is given is composed from its
    // own summary through the edges. What it does with data that has no
    // root here (passed by value, or held in local objects) is not: that
    // may reach anything the callee can write.
    Roots Local, Sinks;
    for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
      Value *A = CB->getArgOperand(k);
      Local.merge(sources(A));
      if (!hasPointer(A->getType()))
        continue;
      auto R = P.deep(P.roots(A));
      S.Unknown |= R.Unknown;
      for (auto i : R.Args)
        S.Edges.insert({"a" + std::to_string(i), N.str(), k});
      for (auto &G : R.Globals)
        S.Edges.insert({"g" + G, N.str(), k});
      Roots InLocals;
      InLocals.Locals = R.Locals;
      Local.merge(stored(InLocals));
      if (!CB->onlyReadsMemory(k))
        Sinks.merge(R.withoutLocals());
    }
    if (!CB->onlyReadsMemory()) {
      Sinks.Globals.insert(AnyGlobal);
      flow(Local, Sinks);
    }
  }
};

/// Writes of any type (not only floating-point data), for what may be
/// overwritten after a call: per argument and global written anywhere in the
/// function, and per call site whether the memory each pointer argument
/// points into may be written again afterwards (loops included).
class WriteFacts {
public:
  WriteFacts(Function &F, EnzymeFunctionSummary &S, const Provenance &P)
      : S(S), P(P) {
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
            auto Ks = keys(P.roots(A));
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
  EnzymeFunctionSummary &S;
  const Provenance &P;
  std::map<const Instruction *, std::set<std::string>> Writes;
  std::map<const BasicBlock *, SmallPtrSet<BasicBlock *, 8>> After;
  std::map<const Value *, unsigned> LocalIds;

  /// Memory a pointer with roots R may point into: "a<i>" (reachable from
  /// argument i), "g<name>", "l<n>" (a local object), "u" (unknown).
  std::set<std::string> keys(const Roots &R) {
    std::set<std::string> K;
    for (auto i : R.Args)
      K.insert("a" + std::to_string(i));
    for (auto &G : R.Globals)
      K.insert("g" + G);
    for (auto *L : R.Locals)
      K.insert("l" + std::to_string(
                         LocalIds.emplace(L, LocalIds.size()).first->second));
    if (R.Unknown)
      K.insert("u");
    return K;
  }

  std::set<std::string> writes(Instruction &I) {
    std::set<std::string> W;
    auto add = [&](const Roots &R) {
      auto K = keys(R);
      if (K.count("u"))
        W.insert("*");
      W.insert(K.begin(), K.end());
    };
    if (auto *St = dyn_cast<StoreInst>(&I))
      add(P.roots(St->getPointerOperand()));
    else if (auto *MI = dyn_cast<MemIntrinsic>(&I))
      add(P.roots(MI->getDest()));
    else if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I))
      add(P.roots(I.getOperand(0)));
    else if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (isa<IntrinsicInst>(&I) || !I.mayWriteToMemory())
        return W;
      auto *Callee = calledFunction(*CB);
      StringRef N = Callee ? Callee->getName() : "";
      if (!Callee)
        W.insert("*");
      // a callee that is not known may write any global
      if (!Callee || startsWith(N, "julia.call"))
        W.insert("g" + std::string(AnyGlobal));
      if (isJuliaMarker(N) || (isAllocationName(N) && N != "posix_memalign"))
        return W;
      bool outputIO = startsWith(N, "_FortranAio") && N.contains("Output");
      for (unsigned k = 0, e = CB->arg_size(); k < e; ++k) {
        Value *A = CB->getArgOperand(k);
        if (!A->getType()->isPointerTy() || outputIO || CB->onlyReadsMemory(k))
          continue;
        add(P.deep(P.roots(A)));
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
  Provenance P(F);
  ActivityFacts(F, S, P);
  WriteFacts(F, S, P);
  return S;
}

json::Object EnzymeFunctionSummary::toJSON() const {
  json::Array JArgs, JWriteAny;
  for (auto &A : Args) {
    JArgs.push_back(json::Object{
        {"read", A.ReadFP}, {"write", A.WriteFP}, {"escape", A.Escape}});
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
  auto matrix = [&](const std::vector<std::vector<bool>> &M) {
    json::Object J;
    for (unsigned s = 0; s < M.size(); ++s) {
      json::Array Sinks;
      for (unsigned t = 0; t < M[s].size(); ++t)
        if (M[s][t])
          Sinks.push_back(sinkName(t));
      if (!Sinks.empty())
        J[sourceName(s)] = std::move(Sinks);
    }
    return J;
  };
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
                        {"flow", matrix(Flow)},
                        {"pts", matrix(PointsTo)},
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

PreservedAnalyses
EnzymeFunctionSummaryPrinterPass::run(Function &F,
                                      FunctionAnalysisManager &FAM) {
  if (!F.isDeclaration())
    OS << "enzyme-function-summary " << F.getName() << ": "
       << json::Value(FAM.getResult<EnzymeFunctionSummaryAnalysis>(F).toJSON())
       << "\n";
  return PreservedAnalyses::all();
}

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

llvm::AnalysisKey EnzymeFunctionSummaryPrinterPass::Key;

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

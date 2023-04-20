//===- Utils.h - Declaration of miscellaneous utilities -------------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @misc{enzymeGithub,
//  author = {William S. Moses and Valentin Churavy},
//  title = {Enzyme: High Performance Automatic Differentiation of LLVM},
//  year = {2020},
//  howpublished = {\url{https://github.com/wsmoses/Enzyme}},
//  note = {commit xxxxxxx}
// }
//
//===----------------------------------------------------------------------===//
//
// This file declares miscellaneous utilities that are used as part of the
// AD process.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_UTILS_H
#define ENZYME_UTILS_H

#include "llvm/ADT/SmallPtrSet.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"

#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/CommandLine.h"

#include "llvm/ADT/SetVector.h"

#include "llvm/IR/Dominators.h"

#if LLVM_VERSION_MAJOR >= 10
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#endif

#include <map>

#include "llvm/IR/DiagnosticInfo.h"

#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#include "TypeAnalysis/ConcreteType.h"

namespace llvm {
class ScalarEvolution;
}

extern "C" {
/// Print additional debug info relevant to performance
extern llvm::cl::opt<bool> EnzymePrintPerf;
}

llvm::SmallVector<llvm::Instruction *, 2> PostCacheStore(llvm::StoreInst *SI,
                                                         llvm::IRBuilder<> &B);

llvm::Value *CreateAllocation(llvm::IRBuilder<> &B, llvm::Type *T,
                              llvm::Value *Count, llvm::Twine Name = "",
                              llvm::CallInst **caller = nullptr,
                              llvm::Instruction **ZeroMem = nullptr,
                              bool isDefault = false);
llvm::CallInst *CreateDealloc(llvm::IRBuilder<> &B, llvm::Value *ToFree);
void ZeroMemory(llvm::IRBuilder<> &Builder, llvm::Type *T, llvm::Value *obj,
                bool isTape);

llvm::Value *CreateReAllocation(llvm::IRBuilder<> &B, llvm::Value *prev,
                                llvm::Type *T, llvm::Value *OuterCount,
                                llvm::Value *InnerCount, llvm::Twine Name = "",
                                llvm::CallInst **caller = nullptr,
                                bool ZeroMem = false);

llvm::PointerType *getDefaultAnonymousTapeType(llvm::LLVMContext &C);

class GradientUtils;
extern std::map<std::string,
                std::function<llvm::Value *(
                    llvm::IRBuilder<> &, llvm::CallInst *,
                    llvm::ArrayRef<llvm::Value *>, GradientUtils *)>>
    shadowHandlers;

template <typename... Args>
void EmitWarning(llvm::StringRef RemarkName,
                 const llvm::DiagnosticLocation &Loc,
                 const llvm::BasicBlock *BB, const Args &...args) {

  llvm::LLVMContext &Ctx = BB->getContext();
  if (Ctx.getDiagHandlerPtr()->isPassedOptRemarkEnabled("enzyme")) {
    std::string str;
    llvm::raw_string_ostream ss(str);
    (ss << ... << args);
    auto R = llvm::OptimizationRemark("enzyme", RemarkName, Loc, BB)
             << ss.str();
    Ctx.diagnose(R);
  }

  if (EnzymePrintPerf)
    (llvm::errs() << ... << args) << "\n";
}

template <typename... Args>
void EmitWarning(llvm::StringRef RemarkName, const llvm::Instruction &I,
                 const Args &...args) {
  EmitWarning(RemarkName, I.getDebugLoc(), I.getParent(), args...);
}

template <typename... Args>
void EmitWarning(llvm::StringRef RemarkName, const llvm::Function &F,
                 const Args &...args) {
  llvm::LLVMContext &Ctx = F.getContext();
  if (Ctx.getDiagHandlerPtr()->isPassedOptRemarkEnabled("enzyme")) {
    std::string str;
    llvm::raw_string_ostream ss(str);
    (ss << ... << args);
    auto R = llvm::OptimizationRemark("enzyme", RemarkName, &F) << ss.str();
    Ctx.diagnose(R);
  }
  if (EnzymePrintPerf)
    (llvm::errs() << ... << args) << "\n";
}

class EnzymeFailure final : public llvm::DiagnosticInfoUnsupported {
public:
  EnzymeFailure(llvm::Twine Msg, const llvm::DiagnosticLocation &Loc,
                const llvm::Instruction *CodeRegion);
};

template <typename... Args>
void EmitFailure(llvm::StringRef RemarkName,
                 const llvm::DiagnosticLocation &Loc,
                 const llvm::Instruction *CodeRegion, Args &...args) {
  std::string *str = new std::string();
  llvm::raw_string_ostream ss(*str);
  (ss << ... << args);
  CodeRegion->getContext().diagnose(
      (EnzymeFailure(llvm::Twine("Enzyme: ") + ss.str(), Loc, CodeRegion)));
}

static inline llvm::Function *isCalledFunction(llvm::Value *val) {
  if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(val)) {
    return CI->getCalledFunction();
  }
  return nullptr;
}

/// Get LLVM fast math flags
static inline llvm::FastMathFlags getFast() {
  llvm::FastMathFlags f;
  f.set();
  return f;
}

/// Pick the maximum value
template <typename T> static inline T max(T a, T b) {
  if (a > b)
    return a;
  return b;
}
/// Pick the maximum value
template <typename T> static inline T min(T a, T b) {
  if (a < b)
    return a;
  return b;
}

/// Get the next non-debug instruction, if one exists
static inline llvm::Instruction *
getNextNonDebugInstructionOrNull(llvm::Instruction *Z) {
  for (llvm::Instruction *I = Z->getNextNode(); I; I = I->getNextNode())
    if (!llvm::isa<llvm::DbgInfoIntrinsic>(I))
      return I;
  return nullptr;
}

/// Get the next non-debug instruction, erring if none exists
static inline llvm::Instruction *
getNextNonDebugInstruction(llvm::Instruction *Z) {
  auto z = getNextNonDebugInstructionOrNull(Z);
  if (z)
    return z;
  llvm::errs() << *Z->getParent() << "\n";
  llvm::errs() << *Z << "\n";
  llvm_unreachable("No valid subsequent non debug instruction");
  exit(1);
  return nullptr;
}

/// Check if a global has metadata
static inline llvm::MDNode *hasMetadata(const llvm::GlobalObject *O,
                                        llvm::StringRef kind) {
  return O->getMetadata(kind);
}

/// Check if an instruction has metadata
static inline llvm::MDNode *hasMetadata(const llvm::Instruction *O,
                                        llvm::StringRef kind) {
  return O->getMetadata(kind);
}

/// Check whether this instruction is returned
static inline bool isReturned(llvm::Instruction *inst) {
  for (const auto a : inst->users()) {
    if (llvm::isa<llvm::ReturnInst>(a))
      return true;
  }
  return false;
}

/// Convert a floating point type to an integer type
/// of the same size
static inline llvm::Type *FloatToIntTy(llvm::Type *T) {
  assert(T->isFPOrFPVectorTy());
  if (auto ty = llvm::dyn_cast<llvm::VectorType>(T)) {
    return llvm::VectorType::get(FloatToIntTy(ty->getElementType()),
#if LLVM_VERSION_MAJOR >= 11
                                 ty->getElementCount());
#else
                                 ty->getNumElements());
#endif
  }
  if (T->isHalfTy())
    return llvm::IntegerType::get(T->getContext(), 16);
  if (T->isFloatTy())
    return llvm::IntegerType::get(T->getContext(), 32);
  if (T->isDoubleTy())
    return llvm::IntegerType::get(T->getContext(), 64);
  assert(0 && "unknown floating point type");
  return nullptr;
}

/// Convert a integer type to a floating point type
/// of the same size
static inline llvm::Type *IntToFloatTy(llvm::Type *T) {
  assert(T->isIntOrIntVectorTy());
  if (auto ty = llvm::dyn_cast<llvm::VectorType>(T)) {
    return llvm::VectorType::get(IntToFloatTy(ty->getElementType()),
#if LLVM_VERSION_MAJOR >= 11
                                 ty->getElementCount());
#else
                                 ty->getNumElements());
#endif
  }
  if (auto ty = llvm::dyn_cast<llvm::IntegerType>(T)) {
    switch (ty->getBitWidth()) {
    case 16:
      return llvm::Type::getHalfTy(T->getContext());
    case 32:
      return llvm::Type::getFloatTy(T->getContext());
    case 64:
      return llvm::Type::getDoubleTy(T->getContext());
    }
  }
  assert(0 && "unknown int to floating point type");
  return nullptr;
}

static inline bool isDebugFunction(llvm::Function *called) {
  if (!called)
    return false;
  switch (called->getIntrinsicID()) {
  case llvm::Intrinsic::dbg_declare:
  case llvm::Intrinsic::dbg_value:
#if LLVM_VERSION_MAJOR > 6
  case llvm::Intrinsic::dbg_label:
#endif
  case llvm::Intrinsic::dbg_addr:
  case llvm::Intrinsic::lifetime_start:
  case llvm::Intrinsic::lifetime_end:
    return true;
  default:
    break;
  }
  return false;
}

static inline bool isCertainPrint(const llvm::StringRef name) {
  if (name == "printf" || name == "puts" || name == "fprintf" ||
      name == "putchar" ||
      name.startswith("_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_") ||
      name.startswith("_ZNSolsE") || name.startswith("_ZNSo9_M_insert") ||
      name.startswith("_ZSt16__ostream_insert") ||
      name.startswith("_ZNSo3put") || name.startswith("_ZSt4endl") ||
      name.startswith("_ZN3std2io5stdio6_print") ||
      name.startswith("_ZNSo5flushEv") || name.startswith("_ZN4core3fmt") ||
      name == "vprintf")
    return true;
  return false;
}

/// Create function for type that performs the derivative memcpy on floating
/// point memory
llvm::Function *
getOrInsertDifferentialFloatMemcpy(llvm::Module &M, llvm::Type *T,
                                   unsigned dstalign, unsigned srcalign,
                                   unsigned dstaddr, unsigned srcaddr);

/// Create function for type that performs memcpy with a stride
llvm::Function *getOrInsertMemcpyStrided(llvm::Module &M, llvm::PointerType *T,
                                         llvm::Type *IT, unsigned dstalign,
                                         unsigned srcalign);

/// Create function for type that performs the derivative memmove on floating
/// point memory
llvm::Function *
getOrInsertDifferentialFloatMemmove(llvm::Module &M, llvm::Type *T,
                                    unsigned dstalign, unsigned srcalign,
                                    unsigned dstaddr, unsigned srcaddr);

llvm::Function *getOrInsertCheckedFree(llvm::Module &M, llvm::CallInst *call,
                                       llvm::Type *Type, unsigned width);

/// Create function for type that performs the derivative MPI_Wait
llvm::Function *getOrInsertDifferentialMPI_Wait(llvm::Module &M,
                                                llvm::ArrayRef<llvm::Type *> T,
                                                llvm::Type *reqType);

/// Create function to computer nearest power of two
llvm::Value *nextPowerOfTwo(llvm::IRBuilder<> &B, llvm::Value *V);

#include "llvm/IR/CFG.h"
#include <deque>
#include <functional>
/// Call the function f for all instructions that happen after inst
/// If the function returns true, the iteration will early exit
static inline void allFollowersOf(llvm::Instruction *inst,
                                  std::function<bool(llvm::Instruction *)> f) {

  for (auto uinst = inst->getNextNode(); uinst != nullptr;
       uinst = uinst->getNextNode()) {
    if (f(uinst))
      return;
  }

  std::deque<llvm::BasicBlock *> todo;
  std::set<llvm::BasicBlock *> done;
  for (auto suc : llvm::successors(inst->getParent())) {
    todo.push_back(suc);
  }
  while (todo.size()) {
    auto BB = todo.front();
    todo.pop_front();
    if (done.count(BB))
      continue;
    done.insert(BB);
    for (auto &ni : *BB) {
      if (f(&ni))
        return;
      if (&ni == inst)
        break;
    }
    for (auto suc : llvm::successors(BB)) {
      todo.push_back(suc);
    }
  }
}

/// Call the function f for all instructions that happen before inst
/// If the function returns true, the iteration will early exit
static inline void
allPredecessorsOf(llvm::Instruction *inst,
                  std::function<bool(llvm::Instruction *)> f) {

  for (auto uinst = inst->getPrevNode(); uinst != nullptr;
       uinst = uinst->getPrevNode()) {
    if (f(uinst))
      return;
  }

  std::deque<llvm::BasicBlock *> todo;
  std::set<llvm::BasicBlock *> done;
  for (auto suc : llvm::predecessors(inst->getParent())) {
    todo.push_back(suc);
  }
  while (todo.size()) {
    auto BB = todo.front();
    todo.pop_front();
    if (done.count(BB))
      continue;
    done.insert(BB);

    llvm::BasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend();
    for (; I != E; ++I) {
      if (f(&*I))
        return;
      if (&*I == inst)
        break;
    }
    for (auto suc : llvm::predecessors(BB)) {
      todo.push_back(suc);
    }
  }
}

/// Call the function f for all instructions that happen before inst
/// If the function returns true, the iteration will early exit
static inline void
allDomPredecessorsOf(llvm::Instruction *inst, llvm::DominatorTree &DT,
                     std::function<bool(llvm::Instruction *)> f) {

  for (auto uinst = inst->getPrevNode(); uinst != nullptr;
       uinst = uinst->getPrevNode()) {
    if (f(uinst))
      return;
  }

  std::deque<llvm::BasicBlock *> todo;
  std::set<llvm::BasicBlock *> done;
  for (auto suc : llvm::predecessors(inst->getParent())) {
    todo.push_back(suc);
  }
  while (todo.size()) {
    auto BB = todo.front();
    todo.pop_front();
    if (done.count(BB))
      continue;
    done.insert(BB);

    if (DT.properlyDominates(BB, inst->getParent())) {
      llvm::BasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend();
      for (; I != E; ++I) {
        if (f(&*I))
          return;
        if (&*I == inst)
          break;
      }
      for (auto suc : llvm::predecessors(BB)) {
        todo.push_back(suc);
      }
    }
  }
}

/// Call the function f for all instructions that happen before inst
/// If the function returns true, the iteration will early exit
static inline void
allUnsyncdPredecessorsOf(llvm::Instruction *inst,
                         std::function<bool(llvm::Instruction *)> f,
                         std::function<void()> preEntry) {

  for (auto uinst = inst->getPrevNode(); uinst != nullptr;
       uinst = uinst->getPrevNode()) {
    if (auto II = llvm::dyn_cast<llvm::IntrinsicInst>(uinst)) {
      if (II->getIntrinsicID() == llvm::Intrinsic::nvvm_barrier0 ||
          II->getIntrinsicID() == llvm::Intrinsic::amdgcn_s_barrier) {
        return;
      }
    }
    if (f(uinst))
      return;
  }

  std::deque<llvm::BasicBlock *> todo;
  std::set<llvm::BasicBlock *> done;
  for (auto suc : llvm::predecessors(inst->getParent())) {
    todo.push_back(suc);
  }
  while (todo.size()) {
    auto BB = todo.front();
    todo.pop_front();
    if (done.count(BB))
      continue;
    done.insert(BB);

    bool syncd = false;
    llvm::BasicBlock::reverse_iterator I = BB->rbegin(), E = BB->rend();
    for (; I != E; ++I) {
      if (auto II = llvm::dyn_cast<llvm::IntrinsicInst>(&*I)) {
        if (II->getIntrinsicID() == llvm::Intrinsic::nvvm_barrier0 ||
            II->getIntrinsicID() == llvm::Intrinsic::amdgcn_s_barrier) {
          syncd = true;
          break;
        }
      }
      if (f(&*I))
        return;
      if (&*I == inst)
        break;
    }
    if (!syncd) {
      for (auto suc : llvm::predecessors(BB)) {
        todo.push_back(suc);
      }
      if (&BB->getParent()->getEntryBlock() == BB) {
        preEntry();
      }
    }
  }
}

#include "llvm/Analysis/LoopInfo.h"

static inline llvm::Loop *getAncestor(llvm::Loop *R1, llvm::Loop *R2) {
  if (!R1 || !R2)
    return nullptr;
  for (llvm::Loop *L1 = R1; L1; L1 = L1->getParentLoop())
    for (llvm::Loop *L2 = R2; L2; L2 = L2->getParentLoop()) {
      if (L1 == L2) {
        return L1;
      }
    }
  return nullptr;
}

// Add all of the stores which may execute after the instruction `inst`
// into the resutls vector.
void mayExecuteAfter(llvm::SmallVectorImpl<llvm::Instruction *> &results,
                     llvm::Instruction *inst,
                     const llvm::SmallPtrSetImpl<llvm::Instruction *> &stores,
                     const llvm::Loop *region);

/// Return whether maybeReader can read from memory written to by maybeWriter
bool writesToMemoryReadBy(llvm::AAResults &AA, llvm::TargetLibraryInfo &TLI,
                          llvm::Instruction *maybeReader,
                          llvm::Instruction *maybeWriter);

// A more advanced version of writesToMemoryReadBy, where the writing
// instruction comes after the reading function. Specifically, even if the two
// instructions may access the same location, this variant checks whether
// also checks whether ScalarEvolution ensures that a subsequent write will not
// overwrite the value read by the load.
//   A simple example: the load/store might write/read from the same
//   location. However, no store will overwrite a previous load.
//   for(int i=0; i<N; i++) {
//      load A[i-1]
//      store A[i] = ...
//   }
bool overwritesToMemoryReadBy(llvm::AAResults &AA, llvm::TargetLibraryInfo &TLI,
                              llvm::ScalarEvolution &SE, llvm::LoopInfo &LI,
                              llvm::DominatorTree &DT,
                              llvm::Instruction *maybeReader,
                              llvm::Instruction *maybeWriter,
                              llvm::Loop *scope = nullptr);
static inline void
/// Call the function f for all instructions that happen between inst1 and inst2
/// If the function returns true, the iteration will early exit
allInstructionsBetween(llvm::LoopInfo &LI, llvm::Instruction *inst1,
                       llvm::Instruction *inst2,
                       std::function<bool(llvm::Instruction *)> f) {
  assert(inst1->getParent()->getParent() == inst2->getParent()->getParent());
  for (auto uinst = inst1->getNextNode(); uinst != nullptr;
       uinst = uinst->getNextNode()) {
    if (f(uinst))
      return;
    if (uinst == inst2)
      return;
  }

  std::set<llvm::Instruction *> instructions;

  llvm::Loop *l1 = LI.getLoopFor(inst1->getParent());
  while (l1 && !l1->contains(inst2->getParent()))
    l1 = l1->getParentLoop();

  // Do all instructions from inst1 up to first instance of inst2's start block
  {
    std::deque<llvm::BasicBlock *> todo;
    std::set<llvm::BasicBlock *> done;
    for (auto suc : llvm::successors(inst1->getParent())) {
      todo.push_back(suc);
    }
    while (todo.size()) {
      auto BB = todo.front();
      todo.pop_front();
      if (done.count(BB))
        continue;
      done.insert(BB);

      for (auto &ni : *BB) {
        instructions.insert(&ni);
      }
      for (auto suc : llvm::successors(BB)) {
        if (!l1 || suc != l1->getHeader()) {
          todo.push_back(suc);
        }
      }
    }
  }

  allPredecessorsOf(inst2, [&](llvm::Instruction *I) -> bool {
    if (instructions.find(I) == instructions.end())
      return /*earlyReturn*/ false;
    return f(I);
  });
}

enum class MPI_CallType {
  ISEND = 1,
  IRECV = 2,
};

enum class MPI_Elem {
  Buf = 0,
  Count = 1,
  DataType = 2,
  Src = 3,
  Tag = 4,
  Comm = 5,
  Call = 6,
  Old = 7
};

static inline llvm::StructType *getMPIHelper(llvm::LLVMContext &Context) {
  using namespace llvm;
  auto i64 = Type::getInt64Ty(Context);
  Type *types[] = {
      /*buf      0 */ Type::getInt8PtrTy(Context),
      /*count    1 */ i64,
      /*datatype 2 */ Type::getInt8PtrTy(Context),
      /*src      3 */ i64,
      /*tag      4 */ i64,
      /*comm     5 */ Type::getInt8PtrTy(Context),
      /*fn       6 */ Type::getInt8Ty(Context),
      /*old      7 */ Type::getInt8PtrTy(Context),
  };
  return StructType::get(Context, types, false);
}

template <MPI_Elem E, bool Pointer = true>
static inline llvm::Value *getMPIMemberPtr(llvm::IRBuilder<> &B,
                                           llvm::Value *V) {
  using namespace llvm;
  auto i64 = Type::getInt64Ty(V->getContext());
  auto i32 = Type::getInt32Ty(V->getContext());
  auto c0_64 = ConstantInt::get(i64, 0);

  if (Pointer) {
#if LLVM_VERSION_MAJOR > 7
    return B.CreateInBoundsGEP(V->getType()->getPointerElementType(), V,
                               {c0_64, ConstantInt::get(i32, (uint64_t)E)});
#else
    return B.CreateInBoundsGEP(V, {c0_64, ConstantInt::get(i32, (uint64_t)E)});
#endif
  } else {
    return B.CreateExtractValue(V, {(unsigned)E});
  }
}

llvm::Value *getOrInsertOpFloatSum(llvm::Module &M, llvm::Type *OpPtr,
                                   ConcreteType CT, llvm::Type *intType,
                                   llvm::IRBuilder<> &B2);

template <typename T> static inline llvm::Function *getFunctionFromCall(T *op) {
  const llvm::Function *called = nullptr;
  using namespace llvm;
  const llvm::Value *callVal;
#if LLVM_VERSION_MAJOR >= 11
  callVal = op->getCalledOperand();
#else
  callVal = op->getCalledValue();
#endif

  while (!called) {
    if (auto castinst = dyn_cast<ConstantExpr>(callVal))
      if (castinst->isCast()) {
        callVal = castinst->getOperand(0);
        continue;
      }
    if (auto fn = dyn_cast<Function>(callVal)) {
      called = fn;
      break;
    }
#if LLVM_VERSION_MAJOR >= 11
    if (auto alias = dyn_cast<GlobalAlias>(callVal)) {
      callVal = dyn_cast<Function>(alias->getAliasee());
      continue;
    }
#endif
    break;
  }
  return called ? const_cast<llvm::Function *>(called) : nullptr;
}

template <typename T> static inline llvm::StringRef getFuncNameFromCall(T *op) {
  auto AttrList =
      op->getAttributes().getAttributes(llvm::AttributeList::FunctionIndex);
  if (AttrList.hasAttribute("enzyme_math"))
    return AttrList.getAttribute("enzyme_math").getValueAsString();
  if (AttrList.hasAttribute("enzyme_allocator"))
    return "enzyme_allocator";

  if (auto called = getFunctionFromCall(op)) {
    if (called->hasFnAttribute("enzyme_math"))
      return called->getFnAttribute("enzyme_math").getValueAsString();
    else if (called->hasFnAttribute("enzyme_allocator"))
      return "enzyme_allocator";
    else
      return called->getName();
  }
  return "";
}

template <typename T>
static inline llvm::Optional<size_t> getAllocationIndexFromCall(T *op) {
  auto AttrList =
      op->getAttributes().getAttributes(llvm::AttributeList::FunctionIndex);
  if (AttrList.hasAttribute("enzyme_allocator")) {
    size_t res;
    bool b = AttrList.getAttribute("enzyme_allocator")
                 .getValueAsString()
                 .getAsInteger(10, res);
    assert(!b);
    return llvm::Optional<size_t>(res);
  }

  if (auto called = getFunctionFromCall(op)) {
    if (called->hasFnAttribute("enzyme_allocator")) {
      size_t res;
      bool b = called->getFnAttribute("enzyme_allocator")
                   .getValueAsString()
                   .getAsInteger(10, res);
      assert(!b);
      return llvm::Optional<size_t>(res);
    }
  }
  return llvm::Optional<size_t>();
}

template <typename T>
static inline llvm::Function *getDeallocatorFnFromCall(T *op) {
  if (auto MD = hasMetadata(op, "enzyme_deallocator_fn")) {
    auto md2 = llvm::cast<llvm::MDTuple>(MD);
    assert(md2->getNumOperands() == 1);
    return llvm::cast<llvm::Function>(
        llvm::cast<llvm::ConstantAsMetadata>(md2->getOperand(0))->getValue());
  }
  if (auto called = getFunctionFromCall(op)) {
    if (auto MD = hasMetadata(called, "enzyme_deallocator_fn")) {
      auto md2 = llvm::cast<llvm::MDTuple>(MD);
      assert(md2->getNumOperands() == 1);
      return llvm::cast<llvm::Function>(
          llvm::cast<llvm::ConstantAsMetadata>(md2->getOperand(0))->getValue());
    }
  }
  llvm::errs() << "dealloc fn: " << *op->getParent()->getParent()->getParent()
               << "\n";
  llvm_unreachable("Illegal deallocatorfn");
}

template <typename T>
static inline std::vector<ssize_t> getDeallocationIndicesFromCall(T *op) {
  llvm::StringRef res = "";
  auto AttrList =
      op->getAttributes().getAttributes(llvm::AttributeList::FunctionIndex);
  if (AttrList.hasAttribute("enzyme_deallocator"))
    res = AttrList.getAttribute("enzyme_deaellocator").getValueAsString();

  if (auto called = getFunctionFromCall(op)) {
    if (called->hasFnAttribute("enzyme_deallocator"))
      res = called->getFnAttribute("enzyme_deallocator").getValueAsString();
  }
  if (res.size() == 0)
    llvm_unreachable("Illegal deallocator");
  llvm::SmallVector<llvm::StringRef, 1> inds;
  res.split(inds, ",");
  std::vector<ssize_t> vinds;
  for (auto ind : inds) {
    ssize_t Result;
    bool b = ind.getAsInteger(10, Result);
    assert(!b);
    vinds.push_back(Result);
  }
  return vinds;
}

llvm::Function *GetFunctionFromValue(llvm::Value *fn);

static inline bool shouldDisableNoWrite(const llvm::CallInst *CI) {
  auto F = getFunctionFromCall(CI);
  auto funcName = getFuncNameFromCall(CI);

  if (CI->hasFnAttr("enzyme_preserve_primal") ||
      hasMetadata(CI, "enzyme_augment") || hasMetadata(CI, "enzyme_gradient") ||
      hasMetadata(CI, "enzyme_derivative") ||
      hasMetadata(CI, "enzyme_splitderivative") ||
      (F &&
       (F->hasFnAttribute("enzyme_preserve_primal") ||
        hasMetadata(F, "enzyme_augment") || hasMetadata(F, "enzyme_gradient") ||
        hasMetadata(F, "enzyme_derivative") ||
        hasMetadata(F, "enzyme_splitderivative"))) ||
      !F) {
    return true;
  }
  if (funcName == "MPI_Wait" || funcName == "MPI_Waitall") {
    return true;
  }
  return false;
}

static inline llvm::Value *getBaseObject(llvm::Value *V) {
  while (true) {
    if (auto CI = llvm::dyn_cast<llvm::CastInst>(V)) {
      V = CI->getOperand(0);
      continue;
    } else if (auto CI = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
      V = CI->getOperand(0);
      continue;
    } else if (auto CI = llvm::dyn_cast<llvm::PHINode>(V)) {
      if (CI->getNumIncomingValues() == 1) {
        V = CI->getOperand(0);
        continue;
      }
    } else if (auto *GA = llvm::dyn_cast<llvm::GlobalAlias>(V)) {
      if (GA->isInterposable())
        break;
      V = GA->getAliasee();
      continue;
    } else if (auto CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
      if (CE->isCast() || CE->getOpcode() == llvm::Instruction::GetElementPtr) {
        V = CE->getOperand(0);
        continue;
      }
    } else if (auto *Call = llvm::dyn_cast<llvm::CallInst>(V)) {
      auto funcName = getFuncNameFromCall(Call);
      if (funcName == "julia.pointer_from_objref") {
        V = Call->getArgOperand(0);
        continue;
      }

      // CaptureTracking can know about special capturing properties of some
      // intrinsics like launder.invariant.group, that can't be expressed with
      // the attributes, but have properties like returning aliasing pointer.
      // Because some analysis may assume that nocaptured pointer is not
      // returned from some special intrinsic (because function would have to
      // be marked with returns attribute), it is crucial to use this function
      // because it should be in sync with CaptureTracking. Not using it may
      // cause weird miscompilations where 2 aliasing pointers are assumed to
      // noalias.
#if LLVM_VERSION_MAJOR >= 10
      if (auto *RP = llvm::getArgumentAliasingToReturnedPointer(Call, false)) {
        V = RP;
        continue;
      }
#endif
    }
#if LLVM_VERSION_MAJOR < 10
#if LLVM_VERSION_MAJOR <= 7
    if (auto CS = llvm::CallSite(V))
#else
    if (auto CS = llvm::dyn_cast<llvm::CallBase>(V))
#endif
    {
      if (auto *RP = llvm::getArgumentAliasingToReturnedPointer(CS)) {
        V = RP;
        continue;
      }
    }
#endif

    if (auto I = llvm::dyn_cast<llvm::Instruction>(V)) {
#if LLVM_VERSION_MAJOR >= 12
      auto V2 = llvm::getUnderlyingObject(I, 100);
#else
      auto V2 = llvm::GetUnderlyingObject(
          I, I->getParent()->getParent()->getParent()->getDataLayout(), 100);
#endif
      if (V2 != V) {
        V = V2;
        break;
      }
    }
    break;
  }
  return V;
}
static inline const llvm::Value *getBaseObject(const llvm::Value *V) {
  return getBaseObject(const_cast<llvm::Value *>(V));
}
#endif

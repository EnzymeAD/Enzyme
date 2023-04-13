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
#include <set>

#include "llvm/IR/DiagnosticInfo.h"

#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#include "TypeAnalysis/ConcreteType.h"

namespace llvm {
class ScalarEvolution;
}

enum class ErrorType {
  NoDerivative = 0,
  NoShadow = 1,
  IllegalTypeAnalysis = 2,
  NoType = 3,
  IllegalFirstPointer = 4,
  InternalError = 5,
  TypeDepthExceeded = 6,
  MixedActivityError = 7,
};

extern "C" {
/// Print additional debug info relevant to performance
extern llvm::cl::opt<bool> EnzymePrintPerf;
extern llvm::cl::opt<bool> EnzymeStrongZero;
extern void (*CustomErrorHandler)(const char *, LLVMValueRef, ErrorType,
                                  const void *, LLVMValueRef);
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
llvm::FastMathFlags getFast();

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

/// Output a set as a string
template <typename T>
static inline std::string to_string(const std::set<T> &us) {
  std::string s = "{";
  for (const auto &y : us)
    s += std::to_string(y) + ",";
  return s + "}";
}

/// Print a map, optionally with a shouldPrint function
/// to decide to print a given value
template <typename T, typename N>
static inline void dumpMap(
    const llvm::ValueMap<T, N> &o,
    std::function<bool(const llvm::Value *)> shouldPrint = [](T) {
      return true;
    }) {
  llvm::errs() << "<begin dump>\n";
  for (auto a : o) {
    if (shouldPrint(a.first))
      llvm::errs() << "key=" << *a.first << " val=" << *a.second << "\n";
  }
  llvm::errs() << "</end dump>\n";
}

/// Print a set
template <typename T>
static inline void dumpSet(const llvm::SmallPtrSetImpl<T *> &o) {
  llvm::errs() << "<begin dump>\n";
  for (auto a : o)
    llvm::errs() << *a << "\n";
  llvm::errs() << "</end dump>\n";
}

template <typename T>
static inline void dumpSet(const llvm::SetVector<T *> &o) {
  llvm::errs() << "<begin dump>\n";
  for (auto a : o)
    llvm::errs() << *a << "\n";
  llvm::errs() << "</end dump>\n";
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

/// Potential return type of generated functions
enum class ReturnType {
  /// Return is a struct of all args and the original return
  ArgsWithReturn,
  /// Return is a struct of all args and two of the original return
  ArgsWithTwoReturns,
  /// Return is a struct of all args
  Args,
  /// Return is a tape type and the original return
  TapeAndReturn,
  /// Return is a tape type and the two of the original return
  TapeAndTwoReturns,
  /// Return is a tape type
  Tape,
  TwoReturns,
  Return,
  Void,
};

/// Potential differentiable argument classifications
enum class DIFFE_TYPE {
  OUT_DIFF = 0,  // add differential to an output struct
  DUP_ARG = 1,   // duplicate the argument and store differential inside
  CONSTANT = 2,  // no differential
  DUP_NONEED = 3 // duplicate this argument and store differential inside, but
                 // don't need the forward
};

enum class BATCH_TYPE {
  SCALAR = 0,
  VECTOR = 1,
};

enum class DerivativeMode {
  ForwardMode = 0,
  ReverseModePrimal = 1,
  ReverseModeGradient = 2,
  ReverseModeCombined = 3,
  ForwardModeSplit = 4,
};

enum class ProbProgMode {
  Trace = 0,
  Condition = 1,
};

/// Classification of value as an original program
/// variable, a derivative variable, neither, or both.
/// This type is used both in differential use analysis
/// and to describe argument bundles.
enum class ValueType {
  // A value that is neither a value in the original
  // program, nor the derivative.
  None = 0,
  // The original program value
  Primal = 1,
  // The derivative value
  Shadow = 2,
  // Both the original program value and the shadow.
  Both = Primal | Shadow,
};

static inline std::string to_string(ValueType mode) {
  switch (mode) {
  case ValueType::None:
    return "None";
  case ValueType::Primal:
    return "Primal";
  case ValueType::Shadow:
    return "Shadow";
  case ValueType::Both:
    return "Both";
  }
  llvm_unreachable("illegal valuetype");
}

typedef std::pair<const llvm::Value *, ValueType> UsageKey;

static inline std::string to_string(DerivativeMode mode) {
  switch (mode) {
  case DerivativeMode::ForwardMode:
    return "ForwardMode";
  case DerivativeMode::ForwardModeSplit:
    return "ForwardModeSplit";
  case DerivativeMode::ReverseModePrimal:
    return "ReverseModePrimal";
  case DerivativeMode::ReverseModeGradient:
    return "ReverseModeGradient";
  case DerivativeMode::ReverseModeCombined:
    return "ReverseModeCombined";
  }
  llvm_unreachable("illegal derivative mode");
}

/// Convert DIFFE_TYPE to a string
static inline std::string to_string(DIFFE_TYPE t) {
  switch (t) {
  case DIFFE_TYPE::OUT_DIFF:
    return "OUT_DIFF";
  case DIFFE_TYPE::CONSTANT:
    return "CONSTANT";
  case DIFFE_TYPE::DUP_ARG:
    return "DUP_ARG";
  case DIFFE_TYPE::DUP_NONEED:
    return "DUP_NONEED";
  default:
    assert(0 && "illegal diffetype");
    return "";
  }
}

/// Convert ReturnType to a string
static inline std::string to_string(ReturnType t) {
  switch (t) {
  case ReturnType::ArgsWithReturn:
    return "ArgsWithReturn";
  case ReturnType::ArgsWithTwoReturns:
    return "ArgsWithTwoReturns";
  case ReturnType::Args:
    return "Args";
  case ReturnType::TapeAndReturn:
    return "TapeAndReturn";
  case ReturnType::TapeAndTwoReturns:
    return "TapeAndTwoReturns";
  case ReturnType::Tape:
    return "Tape";
  case ReturnType::TwoReturns:
    return "TwoReturns";
  case ReturnType::Return:
    return "Return";
  case ReturnType::Void:
    return "Void";
  }
  llvm_unreachable("illegal ReturnType");
}

#include <set>

/// Attempt to automatically detect the differentiable
/// classification based off of a given type
static inline DIFFE_TYPE whatType(llvm::Type *arg, DerivativeMode mode,
                                  bool integersAreConstant,
                                  std::set<llvm::Type *> &seen) {
  assert(arg);
  if (seen.find(arg) != seen.end())
    return DIFFE_TYPE::CONSTANT;
  seen.insert(arg);

  if (arg->isVoidTy() || arg->isEmptyTy()) {
    return DIFFE_TYPE::CONSTANT;
  }

  if (arg->isPointerTy()) {
#if LLVM_VERSION_MAJOR >= 15
    if (!arg->getContext().supportsTypedPointers()) {
      return DIFFE_TYPE::DUP_ARG;
    }
#endif
    switch (whatType(arg->getPointerElementType(), mode, integersAreConstant,
                     seen)) {
    case DIFFE_TYPE::OUT_DIFF:
      return DIFFE_TYPE::DUP_ARG;
    case DIFFE_TYPE::CONSTANT:
      return DIFFE_TYPE::CONSTANT;
    case DIFFE_TYPE::DUP_ARG:
      return DIFFE_TYPE::DUP_ARG;
    case DIFFE_TYPE::DUP_NONEED:
      llvm_unreachable("impossible case");
    }
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
    assert(0 && "Cannot handle type0");
    return DIFFE_TYPE::CONSTANT;
  } else if (arg->isArrayTy()) {
    return whatType(llvm::cast<llvm::ArrayType>(arg)->getElementType(), mode,
                    integersAreConstant, seen);
  } else if (arg->isStructTy()) {
    auto st = llvm::cast<llvm::StructType>(arg);
    if (st->getNumElements() == 0)
      return DIFFE_TYPE::CONSTANT;

    auto ty = DIFFE_TYPE::CONSTANT;
    for (unsigned i = 0; i < st->getNumElements(); ++i) {
      auto midTy =
          whatType(st->getElementType(i), mode, integersAreConstant, seen);
      switch (midTy) {
      case DIFFE_TYPE::OUT_DIFF:
        switch (ty) {
        case DIFFE_TYPE::OUT_DIFF:
        case DIFFE_TYPE::CONSTANT:
          ty = DIFFE_TYPE::OUT_DIFF;
          break;
        case DIFFE_TYPE::DUP_ARG:
          ty = DIFFE_TYPE::DUP_ARG;
          return ty;
        case DIFFE_TYPE::DUP_NONEED:
          llvm_unreachable("impossible case");
        }
        break;
      case DIFFE_TYPE::CONSTANT:
        switch (ty) {
        case DIFFE_TYPE::OUT_DIFF:
          ty = DIFFE_TYPE::OUT_DIFF;
          break;
        case DIFFE_TYPE::CONSTANT:
          break;
        case DIFFE_TYPE::DUP_ARG:
          ty = DIFFE_TYPE::DUP_ARG;
          return ty;
        case DIFFE_TYPE::DUP_NONEED:
          llvm_unreachable("impossible case");
        }
        break;
      case DIFFE_TYPE::DUP_ARG:
        return DIFFE_TYPE::DUP_ARG;
      case DIFFE_TYPE::DUP_NONEED:
        llvm_unreachable("impossible case");
      }
    }
    return ty;
  } else if (arg->isIntOrIntVectorTy() || arg->isFunctionTy()) {
    return integersAreConstant ? DIFFE_TYPE::CONSTANT : DIFFE_TYPE::DUP_ARG;
  } else if (arg->isFPOrFPVectorTy()) {
    return (mode == DerivativeMode::ForwardMode ||
            mode == DerivativeMode::ForwardModeSplit)
               ? DIFFE_TYPE::DUP_ARG
               : DIFFE_TYPE::OUT_DIFF;
  } else {
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
    assert(0 && "Cannot handle type");
    return DIFFE_TYPE::CONSTANT;
  }
}

static inline DIFFE_TYPE whatType(llvm::Type *arg, DerivativeMode mode) {
  std::set<llvm::Type *> seen;
  return whatType(arg, mode, /*intconst*/ true, seen);
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
llvm::Function *getOrInsertDifferentialFloatMemcpy(
    llvm::Module &M, llvm::Type *T, unsigned dstalign, unsigned srcalign,
    unsigned dstaddr, unsigned srcaddr, unsigned bitwidth);

/// Create function for type that performs memcpy with a stride
llvm::Function *getOrInsertMemcpyStrided(llvm::Module &M, llvm::PointerType *T,
                                         llvm::Type *IT, unsigned dstalign,
                                         unsigned srcalign);

/// Create function for type that performs the derivative memmove on floating
/// point memory
llvm::Function *getOrInsertDifferentialFloatMemmove(
    llvm::Module &M, llvm::Type *T, unsigned dstalign, unsigned srcalign,
    unsigned dstaddr, unsigned srcaddr, unsigned bitwidth);

llvm::Function *getOrInsertCheckedFree(llvm::Module &M, llvm::CallInst *call,
                                       llvm::Type *Type, unsigned width);

/// Create function for type that performs the derivative MPI_Wait
llvm::Function *getOrInsertDifferentialMPI_Wait(llvm::Module &M,
                                                llvm::ArrayRef<llvm::Type *> T,
                                                llvm::Type *reqType);

/// Create function to computer nearest power of two
llvm::Value *nextPowerOfTwo(llvm::IRBuilder<> &B, llvm::Value *V);

/// Insert into a map
template <typename K, typename V>
static inline typename std::map<K, V>::iterator
insert_or_assign(std::map<K, V> &map, K &key, V &&val) {
  auto found = map.find(key);
  if (found != map.end()) {
    map.erase(found);
  }
  return map.emplace(key, val).first;
}

/// Insert into a map
template <typename K, typename V>
static inline typename std::map<K, V>::iterator
insert_or_assign2(std::map<K, V> &map, K key, V val) {
  auto found = map.find(key);
  if (found != map.end()) {
    map.erase(found);
  }
  return map.emplace(key, val).first;
}

template <typename K, typename V>
static inline V *findInMap(std::map<K, V> &map, K key) {
  auto found = map.find(key);
  if (found == map.end())
    return nullptr;
  V *val = &found->second;
  return val;
}

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

class AssertingReplacingVH final : public llvm::CallbackVH {
public:
  AssertingReplacingVH() = default;

  AssertingReplacingVH(llvm::Value *new_value) { setValPtr(new_value); }

  void deleted() override final {
    assert(0 && "attempted to delete value with remaining handle use");
    llvm_unreachable("attempted to delete value with remaining handle use");
  }

  void allUsesReplacedWith(llvm::Value *new_value) override final {
    setValPtr(new_value);
  }
  virtual ~AssertingReplacingVH() {}
};

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
#if LLVM_VERSION_MAJOR >= 16
static inline std::optional<size_t> getAllocationIndexFromCall(T *op)
#else
static inline llvm::Optional<size_t> getAllocationIndexFromCall(T *op)
#endif
{
  auto AttrList =
      op->getAttributes().getAttributes(llvm::AttributeList::FunctionIndex);
  if (AttrList.hasAttribute("enzyme_allocator")) {
    size_t res;
    bool b = AttrList.getAttribute("enzyme_allocator")
                 .getValueAsString()
                 .getAsInteger(10, res);
    assert(!b);
#if LLVM_VERSION_MAJOR >= 16
    return std::optional<size_t>(res);
#else
    return llvm::Optional<size_t>(res);
#endif
  }

  if (auto called = getFunctionFromCall(op)) {
    if (called->hasFnAttribute("enzyme_allocator")) {
      size_t res;
      bool b = called->getFnAttribute("enzyme_allocator")
                   .getValueAsString()
                   .getAsInteger(10, res);
      assert(!b);
#if LLVM_VERSION_MAJOR >= 16
      return std::optional<size_t>(res);
#else
      return llvm::Optional<size_t>(res);
#endif
    }
  }
#if LLVM_VERSION_MAJOR >= 16
  return std::optional<size_t>();
#else
  return llvm::Optional<size_t>();
#endif
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

llvm::Function *
getOrInsertDifferentialWaitallSave(llvm::Module &M,
                                   llvm::ArrayRef<llvm::Type *> T,
                                   llvm::PointerType *reqType);

void ErrorIfRuntimeInactive(llvm::IRBuilder<> &B, llvm::Value *primal,
                            llvm::Value *shadow, const char *Message,
                            llvm::DebugLoc &&loc, llvm::Instruction *orig);

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

static inline bool isIntelSubscriptIntrinsic(const llvm::IntrinsicInst &II) {
  return getFuncNameFromCall(&II).startswith("llvm.intel.subscript");
}

static inline bool isIntelSubscriptIntrinsic(const llvm::Value *val) {
  if (auto II = llvm::dyn_cast<llvm::IntrinsicInst>(val)) {
    return isIntelSubscriptIntrinsic(*II);
  }
  return false;
}

static inline bool isPointerArithmeticInst(const llvm::Value *V,
                                           bool includephi = true,
                                           bool includebin = true) {
  if (llvm::isa<llvm::CastInst>(V) || llvm::isa<llvm::GetElementPtrInst>(V) ||
      (includephi && llvm::isa<llvm::PHINode>(V)))
    return true;

  if (includebin)
    if (auto BI = llvm::dyn_cast<llvm::BinaryOperator>(V)) {
      switch (BI->getOpcode()) {
      case llvm::BinaryOperator::Add:
      case llvm::BinaryOperator::Sub:
      case llvm::BinaryOperator::Mul:
      case llvm::BinaryOperator::SDiv:
      case llvm::BinaryOperator::UDiv:
      case llvm::BinaryOperator::SRem:
      case llvm::BinaryOperator::URem:
      case llvm::BinaryOperator::Or:
      case llvm::BinaryOperator::And:
      case llvm::BinaryOperator::Shl:
      case llvm::BinaryOperator::LShr:
      case llvm::BinaryOperator::AShr:
        return true;
      default:
        break;
      }
    }

  if (isIntelSubscriptIntrinsic(V)) {
    return true;
  }

  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(V)) {
    auto funcName = getFuncNameFromCall(Call);
    if (funcName == "julia.pointer_from_objref") {
      return true;
    }
    if (funcName.contains("__enzyme_todense")) {
      return true;
    }
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
    } else if (auto II = llvm::dyn_cast<llvm::IntrinsicInst>(V);
               II && isIntelSubscriptIntrinsic(*II)) {
      V = II->getOperand(3);
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
      auto AttrList = Call->getAttributes().getAttributes(
          llvm::AttributeList::FunctionIndex);
      if (AttrList.hasAttribute("enzyme_pointermath")) {
        size_t res;
        bool failed = AttrList.getAttribute("enzyme_pointermath")
                          .getValueAsString()
                          .getAsInteger(10, res);
        assert(!failed);
        V = Call->getArgOperand(res);
        continue;
      }
      if (funcName == "julia.pointer_from_objref") {
        V = Call->getArgOperand(0);
        continue;
      }
      if (funcName == "jl_reshape_array" || funcName == "ijl_reshape_array") {
        V = Call->getArgOperand(1);
        continue;
      }
      if (funcName.contains("__enzyme_todense")) {
#if LLVM_VERSION_MAJOR >= 14
        size_t numargs = Call->arg_size();
#else
        size_t numargs = Call->getNumArgOperands();
#endif
        if (numargs == 3) {
          V = Call->getArgOperand(2);
          continue;
        }
      }
      if (auto fn = getFunctionFromCall(Call)) {
        auto AttrList = fn->getAttributes().getAttributes(
            llvm::AttributeList::FunctionIndex);
        if (AttrList.hasAttribute("enzyme_pointermath")) {
          size_t res;
          bool failed = AttrList.getAttribute("enzyme_pointermath")
                            .getValueAsString()
                            .getAsInteger(10, res);
          assert(!failed);
          V = Call->getArgOperand(res);
          continue;
        }
        bool found = false;
        for (auto &arg : fn->args()) {
          if (arg.hasAttribute(llvm::Attribute::Returned)) {
            found = true;
            V = Call->getArgOperand(arg.getArgNo());
          }
        }
        if (found)
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

static inline bool isReadOnly(const llvm::CallInst *call, ssize_t arg = -1) {
#if LLVM_VERSION_MAJOR >= 8
  if (call->onlyReadsMemory())
    return true;
  if (arg != -1 && call->onlyReadsMemory(arg))
    return true;
#else
  if (call->hasFnAttr(llvm::Attribute::ReadOnly) ||
      call->hasFnAttr(llvm::Attribute::ReadNone))
    return true;
  if (arg != -1) {
    if (call->dataOperandHasImpliedAttr(arg + 1, llvm::Attribute::ReadOnly) ||
        call->dataOperandHasImpliedAttr(arg + 1, llvm::Attribute::ReadNone))
      return true;
  }
#endif

  auto F = getFunctionFromCall(call);
  if (F) {
    if (F->hasFnAttribute(llvm::Attribute::ReadOnly) ||
        F->hasFnAttribute(llvm::Attribute::ReadNone))
      return true;
    if (arg != -1) {
      if (F->hasParamAttribute(arg, llvm::Attribute::ReadOnly) ||
          F->hasParamAttribute(arg, llvm::Attribute::ReadNone))
        return true;
    }
  }
  return false;
}

static inline bool isWriteOnly(const llvm::CallInst *call, ssize_t arg = -1) {
#if LLVM_VERSION_MAJOR >= 14
  if (call->onlyWritesMemory())
    return true;
  if (arg != -1 && call->onlyWritesMemory(arg))
    return true;
#else
  if (call->hasFnAttr(llvm::Attribute::WriteOnly) ||
      call->hasFnAttr(llvm::Attribute::ReadNone))
    return true;
  if (arg != -1) {
    if (call->dataOperandHasImpliedAttr(arg + 1, llvm::Attribute::WriteOnly) ||
        call->dataOperandHasImpliedAttr(arg + 1, llvm::Attribute::ReadNone))
      return true;
  }
#endif

  auto F = getFunctionFromCall(call);
  if (F) {
    if (F->hasFnAttribute(llvm::Attribute::WriteOnly) ||
        F->hasFnAttribute(llvm::Attribute::ReadNone))
      return true;
    if (arg != -1) {
      if (F->hasParamAttribute(arg, llvm::Attribute::WriteOnly) ||
          F->hasParamAttribute(arg, llvm::Attribute::ReadNone))
        return true;
    }
  }
  return false;
}

static inline bool isReadNone(const llvm::CallInst *call, ssize_t arg = -1) {
  return !isReadOnly(call, arg) && !isWriteOnly(call, arg);
}

static inline bool isNoCapture(const llvm::CallInst *call, size_t idx) {

#if LLVM_VERSION_MAJOR >= 8
  if (call->doesNotCapture(idx))
    return true;
#else
  if (call->dataOperandHasImpliedAttr(idx + 1, llvm::Attribute::NoCapture))
    return true;

#endif

  auto F = getFunctionFromCall(call);
  if (F) {
    if (F->hasParamAttribute(idx, llvm::Attribute::NoCapture))
      return true;
  }
  return false;
}

void attributeKnownFunctions(llvm::Function &F);

struct BlasInfo {
  llvm::StringRef floatType;
  llvm::StringRef prefix;
  llvm::StringRef suffix;
  llvm::StringRef function;
};

llvm::Optional<BlasInfo> extractBLAS(llvm::StringRef in);

llvm::Constant *getUndefinedValueForType(llvm::Type *T, bool forceZero = false);

llvm::Value *SanitizeDerivatives(llvm::Value *val, llvm::Value *toset,
                                 llvm::IRBuilder<> &BuilderM,
                                 llvm::Value *mask = nullptr);

static inline llvm::Value *checkedMul(llvm::IRBuilder<> &Builder2,
                                      llvm::Value *idiff, llvm::Value *pres,
                                      llvm::Twine Name = "") {
  llvm::Value *res = Builder2.CreateFMul(idiff, pres, Name);
  if (EnzymeStrongZero) {
    llvm::Value *zero = llvm::Constant::getNullValue(idiff->getType());
    if (auto C = llvm::dyn_cast<llvm::ConstantFP>(pres))
      if (!C->isInfinity() && !C->isNaN())
        return res;
    res = Builder2.CreateSelect(Builder2.CreateFCmpOEQ(idiff, zero), zero, res);
  }
  return res;
}
static inline llvm::Value *checkedDiv(llvm::IRBuilder<> &Builder2,
                                      llvm::Value *idiff, llvm::Value *pres,
                                      llvm::Twine Name = "") {
  llvm::Value *res = Builder2.CreateFDiv(idiff, pres, Name);
  if (EnzymeStrongZero) {
    llvm::Value *zero = llvm::Constant::getNullValue(idiff->getType());
    if (auto C = llvm::dyn_cast<llvm::ConstantFP>(pres))
      if (!C->isZero() && !C->isNaN())
        return res;
    res = Builder2.CreateSelect(Builder2.CreateFCmpOEQ(idiff, zero), zero, res);
  }
  return res;
}

#endif

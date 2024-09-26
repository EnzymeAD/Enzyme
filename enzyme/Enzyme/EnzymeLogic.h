//===- EnzymeLogic.h - Implementation of forward and reverse pass generation==//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file declares two functions CreatePrimalAndGradient and
// CreateAugmentedPrimal. CreatePrimalAndGradient takes a function, known
// TypeResults of the calling context, known activity analysis of the
// arguments and a bool `topLevel`. It creates a corresponding gradient
// function, computing the forward pass as well if at `topLevel`.
// CreateAugmentedPrimal takes similar arguments and creates an augmented
// forward pass.
//
//===----------------------------------------------------------------------===//
#ifndef ENZYME_LOGIC_H
#define ENZYME_LOGIC_H

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"

#include "llvm/Analysis/AliasAnalysis.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include "ActivityAnalysis.h"
#include "FunctionUtils.h"
#include "TraceUtils.h"
#include "TypeAnalysis/TypeAnalysis.h"
#include "Utils.h"

extern "C" {
extern llvm::cl::opt<bool> EnzymePrint;
extern llvm::cl::opt<bool> EnzymeJuliaAddrLoad;
}

constexpr char EnzymeFPRTPrefix[] = "__enzyme_fprt_";
constexpr char EnzymeFPRTOriginalPrefix[] = "__enzyme_fprt_original_";

enum class AugmentedStruct { Tape, Return, DifferentialReturn };

static inline std::string str(AugmentedStruct c) {
  switch (c) {
  case AugmentedStruct::Tape:
    return "tape";
  case AugmentedStruct::Return:
    return "return";
  case AugmentedStruct::DifferentialReturn:
    return "DifferentialReturn";
  default:
    llvm_unreachable("unknown cache type");
  }
}

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &o,
                                            AugmentedStruct c) {
  return o << str(c);
}

enum class CacheType { Self, Shadow, Tape };

static inline std::string str(CacheType c) {
  switch (c) {
  case CacheType::Self:
    return "self";
  case CacheType::Shadow:
    return "shadow";
  case CacheType::Tape:
    return "tape";
  default:
    llvm_unreachable("unknown cache type");
  }
}

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &o, CacheType c) {
  return o << str(c);
}

//! return structtype if recursive function
class AugmentedReturn {
public:
  llvm::Function *fn;
  //! return structtype if recursive function
  llvm::Type *tapeType;

  std::map<std::pair<llvm::Instruction *, CacheType>, int> tapeIndices;

  //! Map from original call to sub augmentation data
  std::map<const llvm::CallInst *, const AugmentedReturn *> subaugmentations;

  //! Map from information desired from a augmented return to its index in the
  //! returned struct
  std::map<AugmentedStruct, int> returns;

  std::map<llvm::CallInst *, const std::vector<bool>> overwritten_args_map;

  std::map<llvm::Instruction *, bool> can_modref_map;

  std::set<ssize_t> tapeIndiciesToFree;

  const std::vector<DIFFE_TYPE> constant_args;

  bool shadowReturnUsed;

  bool isComplete;

  AugmentedReturn(
      llvm::Function *fn, llvm::Type *tapeType,
      std::map<std::pair<llvm::Instruction *, CacheType>, int> tapeIndices,
      std::map<AugmentedStruct, int> returns,
      std::map<llvm::CallInst *, const std::vector<bool>> overwritten_args_map,
      std::map<llvm::Instruction *, bool> can_modref_map,
      const std::vector<DIFFE_TYPE> &constant_args, bool shadowReturnUsed)
      : fn(fn), tapeType(tapeType), tapeIndices(tapeIndices), returns(returns),
        overwritten_args_map(overwritten_args_map),
        can_modref_map(can_modref_map), constant_args(constant_args),
        shadowReturnUsed(shadowReturnUsed), isComplete(false) {}
};

///  \p todiff is the function to differentiate
///  \p retType is the activity info of the return.
///  Only allowed to be DUP_ARG or CONSTANT. DUP_NONEED is not allowed,
///  set returnValue to false instead.
///  \p constant_args is the activity info of the arguments
///  \p returnValue is whether the primal's return should also be returned.
///  \p dretUsed is whether the shadow return value should also be returned.
///  Only allowed to be true if retType is CDIFFE_TYPE::DUP_ARG.
///  \p additionalArg is the type (or null) of an additional type in the
///  signature to hold the tape.
///  \p typeInfo is the type info information about the calling context
///  \p _overwritten_args marks whether an argument may be overwritten
///  before loads in the generated function (and thus cannot be cached).
///  \p AtomicAdd is whether to perform all adjoint
///  updates to memory in an atomic way
struct ReverseCacheKey {
  llvm::Function *todiff;
  DIFFE_TYPE retType;
  const std::vector<DIFFE_TYPE> constant_args;
  std::vector<bool> overwritten_args;
  bool returnUsed;
  bool shadowReturnUsed;
  DerivativeMode mode;
  unsigned width;
  bool freeMemory;
  bool AtomicAdd;
  llvm::Type *additionalType;
  bool forceAnonymousTape;
  const FnTypeInfo typeInfo;
  bool runtimeActivity;

  /*
  inline bool operator==(const ReverseCacheKey& rhs) const {
      return todiff == rhs.todiff &&
             retType == rhs.retType &&
             constant_args == rhs.constant_args &&
             overwritten_args == rhs.overwritten_args &&
             returnUsed == rhs.returnUsed &&
             shadowReturnUsed == rhs.shadowReturnUsed &&
             mode == rhs.mode &&
             freeMemory == rhs.freeMemory &&
             AtomicAdd == rhs.AtomicAdd &&
             additionalType == rhs.additionalType &&
             typeInfo == rhs.typeInfo;
  }
  */

  inline bool operator<(const ReverseCacheKey &rhs) const {
    if (todiff < rhs.todiff)
      return true;
    if (rhs.todiff < todiff)
      return false;

    if (retType < rhs.retType)
      return true;
    if (rhs.retType < retType)
      return false;

    if (std::lexicographical_compare(constant_args.begin(), constant_args.end(),
                                     rhs.constant_args.begin(),
                                     rhs.constant_args.end()))
      return true;
    if (std::lexicographical_compare(
            rhs.constant_args.begin(), rhs.constant_args.end(),
            constant_args.begin(), constant_args.end()))
      return false;

    if (std::lexicographical_compare(
            overwritten_args.begin(), overwritten_args.end(),
            rhs.overwritten_args.begin(), rhs.overwritten_args.end()))
      return true;
    if (std::lexicographical_compare(
            rhs.overwritten_args.begin(), rhs.overwritten_args.end(),
            overwritten_args.begin(), overwritten_args.end()))
      return false;

    if (returnUsed < rhs.returnUsed)
      return true;
    if (rhs.returnUsed < returnUsed)
      return false;

    if (shadowReturnUsed < rhs.shadowReturnUsed)
      return true;
    if (rhs.shadowReturnUsed < shadowReturnUsed)
      return false;

    if (mode < rhs.mode)
      return true;
    if (rhs.mode < mode)
      return false;

    if (width < rhs.width)
      return true;
    if (rhs.width < width)
      return false;

    if (freeMemory < rhs.freeMemory)
      return true;
    if (rhs.freeMemory < freeMemory)
      return false;

    if (AtomicAdd < rhs.AtomicAdd)
      return true;
    if (rhs.AtomicAdd < AtomicAdd)
      return false;

    if (additionalType < rhs.additionalType)
      return true;
    if (rhs.additionalType < additionalType)
      return false;

    if (forceAnonymousTape < rhs.forceAnonymousTape)
      return true;
    if (rhs.forceAnonymousTape < forceAnonymousTape)
      return false;

    if (typeInfo < rhs.typeInfo)
      return true;
    if (rhs.typeInfo < typeInfo)
      return false;

    if (runtimeActivity < rhs.runtimeActivity)
      return true;
    if (rhs.runtimeActivity < runtimeActivity)
      return false;

    // equal
    return false;
  }
};

// Holder class to represent a context in which a derivative
// or batch is being requested. This contains the instruction
// (or null) that led to the request, and a builder (or null)
// of the insertion point for code.
struct RequestContext {
  llvm::Instruction *req;
  llvm::IRBuilder<> *ip;
  RequestContext(llvm::Instruction *req = nullptr,
                 llvm::IRBuilder<> *ip = nullptr)
      : req(req), ip(ip) {}
};

[[maybe_unused]] static llvm::Type *
getTypeForWidth(llvm::LLVMContext &ctx, unsigned width, bool builtinFloat) {
  switch (width) {
  default:
    if (builtinFloat)
      llvm::report_fatal_error("Invalid float width requested");
    else
      llvm::report_fatal_error(
          "Truncation to non builtin float width unsupported");
  case 64:
    return llvm::Type::getDoubleTy(ctx);
  case 32:
    return llvm::Type::getFloatTy(ctx);
  case 16:
    return llvm::Type::getHalfTy(ctx);
  }
}

enum TruncateMode {
  TruncMemMode = 0b0001,
  TruncOpMode = 0b0010,
  TruncOpFullModuleMode = 0b0110,
};
[[maybe_unused]] static const char *truncateModeStr(TruncateMode mode) {
  switch (mode) {
  case TruncMemMode:
    return "mem";
  case TruncOpMode:
    return "op";
  case TruncOpFullModuleMode:
    return "op_full_module";
  }
  llvm_unreachable("Invalid truncation mode");
}

struct FloatRepresentation {
  // |_|__________|_________________|
  //  ^         ^         ^
  //  sign bit  exponent  significand
  //
  //  value = (sign) * significand * 2 ^ exponent
  unsigned exponentWidth;
  unsigned significandWidth;

  FloatRepresentation(unsigned e, unsigned s)
      : exponentWidth(e), significandWidth(s) {}

  unsigned getTypeWidth() const { return 1 + exponentWidth + significandWidth; }

  bool canBeBuiltin() const {
    unsigned w = getTypeWidth();
    return (w == 16 && significandWidth == 10) ||
           (w == 32 && significandWidth == 23) ||
           (w == 64 && significandWidth == 52);
  }

  llvm::Type *getBuiltinType(llvm::LLVMContext &ctx) const {
    if (!canBeBuiltin())
      return nullptr;
    return getTypeForWidth(ctx, getTypeWidth(), /*builtinFloat=*/true);
  }

  llvm::Type *getType(llvm::LLVMContext &ctx) const {
    llvm::Type *builtinType = getBuiltinType(ctx);
    if (builtinType)
      return builtinType;
    llvm_unreachable("TODO MPFR");
  }

  bool operator==(const FloatRepresentation &other) const {
    return other.exponentWidth == exponentWidth &&
           other.significandWidth == significandWidth;
  }
  bool operator<(const FloatRepresentation &other) const {
    return std::tuple(exponentWidth, significandWidth) <
           std::tuple(other.exponentWidth, other.significandWidth);
  }
  std::string to_string() const {
    return std::to_string(getTypeWidth()) + "_" +
           std::to_string(significandWidth);
  }
};

struct FloatTruncation {
private:
  FloatRepresentation from, to;
  TruncateMode mode;

public:
  FloatTruncation(FloatRepresentation From, FloatRepresentation To,
                  TruncateMode mode)
      : from(From), to(To), mode(mode) {
    if (!From.canBeBuiltin())
      llvm::report_fatal_error("Float truncation `from` type is not builtin.");
    if (From.exponentWidth < To.exponentWidth &&
        (mode == TruncOpMode || mode == TruncOpFullModuleMode))
      llvm::report_fatal_error("Float truncation `from` type must have "
                               "a wider exponent than `to`.");
    if (From.significandWidth < To.significandWidth &&
        (mode == TruncOpMode || mode == TruncOpFullModuleMode))
      llvm::report_fatal_error("Float truncation `from` type must have "
                               "a wider significand than `to`.");
    if (From == To)
      llvm::report_fatal_error(
          "Float truncation `from` and `to` type must not be the same.");
  }
  TruncateMode getMode() { return mode; }
  FloatRepresentation getTo() { return to; }
  unsigned getFromTypeWidth() { return from.getTypeWidth(); }
  unsigned getToTypeWidth() { return to.getTypeWidth(); }
  llvm::Type *getFromType(llvm::LLVMContext &ctx) {
    return from.getBuiltinType(ctx);
  }
  bool isToFPRT() {
    // TODO maybe add new mode in which we directly truncate to native fp ops,
    // for now everything goes through the runtime
    return true;
  }
  llvm::Type *getToType(llvm::LLVMContext &ctx) { return getFromType(ctx); }
  auto getTuple() const { return std::tuple(from, to, mode); }
  bool operator==(const FloatTruncation &other) const {
    return getTuple() == other.getTuple();
  }
  bool operator<(const FloatTruncation &other) const {
    return getTuple() < other.getTuple();
  }
  std::string mangleTruncation() const {
    return from.to_string() + "to" + to.to_string();
  }
  std::string mangleFrom() const { return from.to_string(); }
};

typedef std::map<std::tuple<std::string, unsigned, unsigned>,
                 llvm::GlobalValue *>
    UniqDebugLocStrsTy;

class EnzymeLogic {
public:
  PreProcessCache PPC;
  UniqDebugLocStrsTy UniqDebugLocStrs;

  /// \p PostOpt is whether to perform basic
  ///  optimization of the function after synthesis
  bool PostOpt;

  EnzymeLogic(bool PostOpt) : PostOpt(PostOpt) {}

  struct AugmentedCacheKey {
    llvm::Function *fn;
    DIFFE_TYPE retType;
    const std::vector<DIFFE_TYPE> constant_args;
    std::vector<bool> overwritten_args;
    bool returnUsed;
    bool shadowReturnUsed;
    const FnTypeInfo typeInfo;
    bool freeMemory;
    bool AtomicAdd;
    bool omp;
    unsigned width;
    bool runtimeActivity;

    inline bool operator<(const AugmentedCacheKey &rhs) const {
      if (fn < rhs.fn)
        return true;
      if (rhs.fn < fn)
        return false;

      if (retType < rhs.retType)
        return true;
      if (rhs.retType < retType)
        return false;

      if (std::lexicographical_compare(
              constant_args.begin(), constant_args.end(),
              rhs.constant_args.begin(), rhs.constant_args.end()))
        return true;
      if (std::lexicographical_compare(
              rhs.constant_args.begin(), rhs.constant_args.end(),
              constant_args.begin(), constant_args.end()))
        return false;

      if (std::lexicographical_compare(
              overwritten_args.begin(), overwritten_args.end(),
              rhs.overwritten_args.begin(), rhs.overwritten_args.end()))
        return true;
      if (std::lexicographical_compare(
              rhs.overwritten_args.begin(), rhs.overwritten_args.end(),
              overwritten_args.begin(), overwritten_args.end()))
        return false;

      if (returnUsed < rhs.returnUsed)
        return true;
      if (rhs.returnUsed < returnUsed)
        return false;

      if (shadowReturnUsed < rhs.shadowReturnUsed)
        return true;
      if (rhs.shadowReturnUsed < shadowReturnUsed)
        return false;

      if (freeMemory < rhs.freeMemory)
        return true;
      if (rhs.freeMemory < freeMemory)
        return false;

      if (AtomicAdd < rhs.AtomicAdd)
        return true;
      if (rhs.AtomicAdd < AtomicAdd)
        return false;

      if (omp < rhs.omp)
        return true;
      if (rhs.omp < omp)
        return false;

      if (typeInfo < rhs.typeInfo)
        return true;
      if (rhs.typeInfo < typeInfo)
        return false;

      if (width < rhs.width)
        return true;
      if (rhs.width < width)
        return false;

      if (runtimeActivity < rhs.runtimeActivity)
        return true;
      if (rhs.runtimeActivity < runtimeActivity)
        return false;

      // equal
      return false;
    }
  };

  std::map<llvm::Function *, llvm::Function *> NoFreeCachedFunctions;
  llvm::Function *CreateNoFree(RequestContext context, llvm::Function *todiff);
  llvm::Value *CreateNoFree(RequestContext context, llvm::Value *todiff);

  std::map<AugmentedCacheKey, AugmentedReturn> AugmentedCachedFunctions;

  /// Create an augmented forward pass.
  ///  \p context the instruction which requested this derivative (or null).
  ///  \p todiff is the function to differentiate
  ///  \p retType is the activity info of the return
  ///  \p constant_args is the activity info of the arguments
  ///  \p returnUsed is whether the primal's return should also be returned
  ///  \p typeInfo is the type info information about the calling context
  ///  \p _overwritten_args marks whether an argument may be rewritten before
  ///  loads in the generated function (and thus cannot be cached).
  ///  \p forceAnonymousTape forces the tape to be an i8* rather than the true
  ///  tape structure
  ///  \p AtomicAdd is whether to perform all adjoint updates to
  ///  memory in an atomic way
  const AugmentedReturn &CreateAugmentedPrimal(
      RequestContext context, llvm::Function *todiff, DIFFE_TYPE retType,
      llvm::ArrayRef<DIFFE_TYPE> constant_args, TypeAnalysis &TA,
      bool returnUsed, bool shadowReturnUsed, const FnTypeInfo &typeInfo,
      const std::vector<bool> _overwritten_args, bool forceAnonymousTape,
      bool runtimeActivity, unsigned width, bool AtomicAdd, bool omp = false);

  std::map<ReverseCacheKey, llvm::Function *> ReverseCachedFunctions;

  struct ForwardCacheKey {
    llvm::Function *todiff;
    DIFFE_TYPE retType;
    const std::vector<DIFFE_TYPE> constant_args;
    std::vector<bool> overwritten_args;
    bool returnUsed;
    DerivativeMode mode;
    unsigned width;
    llvm::Type *additionalType;
    const FnTypeInfo typeInfo;
    bool runtimeActivity;

    inline bool operator<(const ForwardCacheKey &rhs) const {
      if (todiff < rhs.todiff)
        return true;
      if (rhs.todiff < todiff)
        return false;

      if (retType < rhs.retType)
        return true;
      if (rhs.retType < retType)
        return false;

      if (std::lexicographical_compare(
              constant_args.begin(), constant_args.end(),
              rhs.constant_args.begin(), rhs.constant_args.end()))
        return true;
      if (std::lexicographical_compare(
              rhs.constant_args.begin(), rhs.constant_args.end(),
              constant_args.begin(), constant_args.end()))
        return false;

      if (std::lexicographical_compare(
              overwritten_args.begin(), overwritten_args.end(),
              rhs.overwritten_args.begin(), rhs.overwritten_args.end()))
        return true;
      if (std::lexicographical_compare(
              rhs.overwritten_args.begin(), rhs.overwritten_args.end(),
              overwritten_args.begin(), overwritten_args.end()))
        return false;

      if (returnUsed < rhs.returnUsed)
        return true;
      if (rhs.returnUsed < returnUsed)
        return false;

      if (mode < rhs.mode)
        return true;
      if (rhs.mode < mode)
        return false;

      if (width < rhs.width)
        return true;
      if (rhs.width < width)
        return false;

      if (additionalType < rhs.additionalType)
        return true;
      if (rhs.additionalType < additionalType)
        return false;

      if (typeInfo < rhs.typeInfo)
        return true;
      if (rhs.typeInfo < typeInfo)
        return false;

      if (runtimeActivity < rhs.runtimeActivity)
        return true;
      if (rhs.runtimeActivity < runtimeActivity)
        return false;

      // equal
      return false;
    }
  };

  std::map<ForwardCacheKey, llvm::Function *> ForwardCachedFunctions;

  using BatchCacheKey = std::tuple<llvm::Function *, unsigned,
                                   std::vector<BATCH_TYPE>, BATCH_TYPE>;
  std::map<BatchCacheKey, llvm::Function *> BatchCachedFunctions;

  using TraceCacheKey =
      std::tuple<llvm::Function *, ProbProgMode, bool, TraceInterface *>;
  std::map<TraceCacheKey, llvm::Function *> TraceCachedFunctions;

  /// Create the reverse pass, or combined forward+reverse derivative function.
  ///  \p context the instruction which requested this derivative (or null).
  ///  \p augmented is the data structure created by prior call to an
  ///   augmented forward pass
  llvm::Function *CreatePrimalAndGradient(RequestContext context,
                                          const ReverseCacheKey &&key,
                                          TypeAnalysis &TA,
                                          const AugmentedReturn *augmented,
                                          bool omp = false);

  /// Create the forward (or forward split) mode derivative function.
  ///  \p context the instruction which requested this derivative (or null).
  ///  \p todiff is the function to differentiate
  ///  \p retType is the activity info of the return
  ///  \p constant_args is the activity info of the arguments
  ///  \p TA is the type analysis results
  ///  \p returnValue is whether the primal's return should also be returned
  ///  \p mode is the requested derivative mode
  ///  \p is whether we should free memory allocated here (and could be
  ///  accessed externally).
  ///  \p width is the vector width requested.
  ///  \p additionalArg is the type (or null) of an additional type in the
  ///  signature to hold the tape.
  ///  \p FnTypeInfo is the known types of the argument and returns
  ///  \p _overwritten_args marks whether an argument may be rewritten
  ///  before loads in the generated function (and thus cannot be cached).
  ///  \p augmented is the data structure created by prior call to an
  ///   augmented forward pass
  ///  \p omp is whether this function is an OpenMP closure body.
  llvm::Function *CreateForwardDiff(
      RequestContext context, llvm::Function *todiff, DIFFE_TYPE retType,
      llvm::ArrayRef<DIFFE_TYPE> constant_args, TypeAnalysis &TA,
      bool returnValue, DerivativeMode mode, bool freeMemory,
      bool runtimeActivity, unsigned width, llvm::Type *additionalArg,
      const FnTypeInfo &typeInfo, const std::vector<bool> _overwritten_args,
      const AugmentedReturn *augmented, bool omp = false);

  /// Create a function batched in its inputs.
  ///  \p context the instruction which requested this batch (or null).
  ///  \p tobatch is the function to batch
  ///  \p width is the vector width requested.
  ///  \p arg_types denotes which arguments are batched.
  ///  \p ret_type denotes whether to batch the return.
  llvm::Function *CreateBatch(RequestContext context, llvm::Function *tobatch,
                              unsigned width,
                              llvm::ArrayRef<BATCH_TYPE> arg_types,
                              BATCH_TYPE ret_type);

  using TruncateCacheKey =
      std::tuple<llvm::Function *, FloatTruncation, unsigned>;
  std::map<TruncateCacheKey, llvm::Function *> TruncateCachedFunctions;
  llvm::Function *CreateTruncateFunc(RequestContext context,
                                     llvm::Function *tobatch,
                                     FloatTruncation truncation,
                                     TruncateMode mode);
  bool CreateTruncateValue(RequestContext context, llvm::Value *addr,
                           FloatRepresentation from, FloatRepresentation to,
                           bool isTruncate);

  /// Create a traced version of a function
  ///  \p context the instruction which requested this trace (or null).
  ///  \p totrace is the function to trace
  ///  \p sampleFunctions is a set of the functions to sample
  ///  \p observeFunctions is a set of the functions to observe
  ///  \p ActiveRandomVariables is a set of which variables are active
  ///  \p mode is the mode to use
  ///  \p autodiff is whether to also differentiate
  ///  \p interface specifies the ABI to use.
  llvm::Function *
  CreateTrace(RequestContext context, llvm::Function *totrace,
              const llvm::SmallPtrSetImpl<llvm::Function *> &sampleFunctions,
              const llvm::SmallPtrSetImpl<llvm::Function *> &observeFunctions,
              const llvm::StringSet<> &ActiveRandomVariables, ProbProgMode mode,
              bool autodiff, TraceInterface *interface);

  void clear();
};

extern "C" {
extern llvm::cl::opt<bool> looseTypeAnalysis;
extern llvm::cl::opt<bool> nonmarkedglobals_inactiveloads;
};

class GradientUtils;
bool shouldAugmentCall(llvm::CallInst *op, const GradientUtils *gutils);

bool legalCombinedForwardReverse(
    llvm::CallInst *origop,
    const std::map<llvm::ReturnInst *, llvm::StoreInst *> &replacedReturns,
    llvm::SmallVectorImpl<llvm::Instruction *> &postCreate,
    llvm::SmallVectorImpl<llvm::Instruction *> &userReplace,
    const GradientUtils *gutils,
    const llvm::SmallPtrSetImpl<const llvm::Instruction *>
        &unnecessaryInstructions,
    const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable,
    const bool subretused);

std::pair<llvm::SmallVector<llvm::Type *, 4>,
          llvm::SmallVector<llvm::Type *, 4>>
getDefaultFunctionTypeForAugmentation(llvm::FunctionType *called,
                                      bool returnUsed, DIFFE_TYPE retType);

std::pair<llvm::SmallVector<llvm::Type *, 4>,
          llvm::SmallVector<llvm::Type *, 4>>
getDefaultFunctionTypeForGradient(llvm::FunctionType *called,
                                  DIFFE_TYPE retType);
#endif

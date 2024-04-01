//===- blas-tblgen.cpp - Top-Level TableGen implementation for Enzyme
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for Enzyme's TableGen.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "caching.h"
#include "datastructures.h"

using namespace llvm;

// TODO: add this to .td file and generate it based on that
std::string get_blas_ret_ty(StringRef dfnc_name) {
  if (has_active_return(dfnc_name))
    return "fpType";
  else
    return "Builder2.getVoidTy()";
}

bool hasDiffeRet(Init *resultTree) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "DiffeRetIndex" || Def->isSubClassOf("DiffeRetIndex")) {
      return true;
    }
    for (auto arg : resultRoot->getArgs()) {
      if (hasDiffeRet(arg))
        return true;
    }
  }
  if (DefInit *DefArg = dyn_cast<DefInit>(resultTree)) {
    auto Def = DefArg->getDef();
    if (Def->isSubClassOf("DiffeRetIndex")) {
      return true;
    }
  }
  return false;
}

bool hasAdjoint(Init *resultTree, StringRef argName) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (Def->isSubClassOf("adj")) {
      auto name = Def->getValueAsString("name");
      return name == argName;
    }
    for (auto arg : resultRoot->getArgs()) {
      if (hasAdjoint(arg, argName))
        return true;
    }
  }
  if (DefInit *DefArg = dyn_cast<DefInit>(resultTree)) {
    auto Def = DefArg->getDef();
    if (Def->isSubClassOf("adj")) {
      auto name = Def->getValueAsString("name");
      return name == argName;
    }
  }
  return false;
}

static void checkBlasCallsInDag(const RecordKeeper &RK,
                                ArrayRef<Record *> blasPatterns,
                                StringRef blasName, const DagInit *toSearch) {

  // For nested FAdd, ... rules which don't directly call a blass fnc
  for (size_t i = 0; i < toSearch->getNumArgs(); i++) {
    if (DagInit *arg = dyn_cast<DagInit>(toSearch->getArg(i))) {
      checkBlasCallsInDag(RK, blasPatterns, blasName, arg);
    }
  }
}

/// Here we check that all the Blas derivatives who call another
/// blas function will use the correct amount of args
/// Later we might check for "types" too.
static void checkBlasCalls(const RecordKeeper &RK,
                           ArrayRef<Record *> blasPatterns) {
  for (auto &&pattern : blasPatterns) {
    ListInit *argOps = pattern->getValueAsListInit("ArgDerivatives");
    // for each possibly active parameter
    for (auto argOp : *argOps) {
      DagInit *resultRoot = cast<DagInit>(argOp);
      checkBlasCallsInDag(RK, blasPatterns, pattern->getName(), resultRoot);
    }
  }
}

// handleBLAS is called in the AdjointGenerator.h
void emit_handleBLAS(ArrayRef<TGPattern> blasPatterns, raw_ostream &os) {
  os << "bool handleBLAS(llvm::CallInst &call, llvm::Function *called,"
        "BlasInfo blas,const std::vector<bool> &overwritten_args) {         \n"
     << "  using llvm::Type;                                                \n"
     << "  bool result = true;                                              \n"
     << "  if (!gutils->isConstantInstruction(&call)) {                     \n"
     << "    Type *fpType;                                                  \n"
     << "    if (blas.floatType == \"d\" || blas.floatType == \"D\") {      \n"
     << "      fpType = Type::getDoubleTy(call.getContext());               \n"
     << "    } else if (blas.floatType == \"s\" || blas.floatType == \"S\"){\n"
     << "      fpType = Type::getFloatTy(call.getContext());                \n"
     << "    } else {                                                       \n"
     << "      assert(false && \"Unreachable\");                            \n"
     << "    }                                                              \n";
  bool first = true;
  for (auto &&pattern : blasPatterns) {
    bool hasNonInactive = false;
    for (Rule rule : pattern.getRules()) {
      const auto ruleDag = rule.getRuleDag();
      const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
      if (Def->isSubClassOf("MagicInst") && Def->getName() == "inactive")
        continue;
      hasNonInactive = true;
      break;
    }
    if (!hasNonInactive)
      continue;
    auto name = pattern.getName();
    os << "    " << ((first) ? "" : "} else ") << " if (blas.function == \""
       << name << "\") {                           \n"
       << "      result = handle_" << name
       << "(blas, call, called, overwritten_args, fpType);                 \n";
    first = false;
  }
  os << "    } else {                                                       \n"
     << "      llvm::errs() << \" fallback?\\n\";                           \n"
     << "      return false;                                                \n"
     << "    }                                                              \n"
     << "  } else {                                                         \n"
     << "    if (gutils->knownRecomputeHeuristic.find(&call) !=\n"
     << "      gutils->knownRecomputeHeuristic.end()) {\n"
     << "      if (!gutils->knownRecomputeHeuristic[&call]) {\n"
     << "       auto newCall = gutils->getNewFromOriginal(&call);\n"
     << "       llvm::IRBuilder<> BuilderZ(newCall);\n"
     << "       gutils->cacheForReverse(BuilderZ, newCall,\n"
     << "       getIndex(&call, CacheType::Self, BuilderZ));\n"
     << "      }\n"
     << "    }\n"
     << "    if (Mode == DerivativeMode::ReverseModeGradient) {             \n"
     << "      eraseIfUnused(call, /*erase*/ true, /*check*/ false);        \n"
     << "    } else {                                                       \n"
     << "      eraseIfUnused(call);                                         \n"
     << "    }                                                              \n"
     << "  }\n"
     << "  return result;                                                   \n"
     << "}                                                                  \n";
}

void emit_beginning(const TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  os << "\nbool handle_" << name
     << "(BlasInfo blas, llvm::CallInst &call, llvm::Function *called,\n"
     << "    const std::vector<bool> &overwritten_args, "
        "llvm::Type *fpType) "
        "{\n"
     << "  \n"
     << "#ifdef __clang__\n"
     << "#pragma clang diagnostic push\n"
     << "#pragma clang diagnostic ignored \"-Wunused-variable\"\n"
     << "#pragma clang diagnostic ignored \"-Wunused-but-set-variable\"\n"
     << "#else\n"
     << "#pragma GCC diagnostic push\n"
     << "#pragma GCC diagnostic ignored \"-Wunused-variable\"\n"
     << "#pragma GCC diagnostic ignored \"-Wunused-but-set-variable\"\n"
     << "#endif\n"
     << "  using namespace llvm;\n"
     << "  CallInst *const newCall = "
        "cast<CallInst>(gutils->getNewFromOriginal(&call));\n"
     << "  IRBuilder<> BuilderZ(newCall);\n"
     << "  BuilderZ.setFastMathFlags(getFast());\n"
     << "  IRBuilder<> allocationBuilder(gutils->inversionAllocs);\n"
     << "  allocationBuilder.setFastMathFlags(getFast());\n"
     << "  // never cache in Fwd Mode\n"
     << "  const bool cacheMode = (Mode != DerivativeMode::ForwardMode);\n";
}

void emit_free_and_ending(const TGPattern &pattern, raw_ostream &os) {
  os << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
     << "      Mode == DerivativeMode::ReverseModeGradient ||\n"
     << "      Mode == DerivativeMode::ForwardModeSplit) {\n"
     << "    if (shouldFree()) {\n";

  auto nameVec = pattern.getArgNames();
  auto typeMap = pattern.getArgTypeMap();
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (isVecLikeArg(ty)) {
      auto name = nameVec[i];
      os << "      if (cache_" << name << ") {\n"
         << "        CreateDealloc(Builder2, free_" << name << ");\n"
         << "      }\n";
    }
  }

  // next one is to handle input<name> usages.
  // Disabled for now, since input<X> and arg_x
  // overlap in the case that x will be cached.
  // This is ok for now since we don't have any rule which
  // would overwrite x internally. Figure out what to do
  // once we hit more complex rules that make good tests.
  // auto activeArgs = pattern.getActiveArgs();
  // auto rules = pattern.getRules();
  // std::string toCache = "";
  // for (size_t a = 0; a < activeArgs.size(); a++) {
  //  auto rule = rules[a];
  //  auto i = activeArgs[a];
  //  auto name = nameVec[i];
  //  const DagInit *ruleDag = rule.getRuleDag();
  //  std::string toCache = get_input_mat(ruleDag);
  //  if (toCache != "") {
  //    os << "      if (active_" << name << ") {\n"
  //       << "        CreateDealloc(Builder2, free_input_" << toCache << ");\n"
  //       << "      }\n";
  //    break;
  //  }
  //}

  os << "    }\n"
     << "  }\n"
     << "                                                                   \n"
     << "  if (gutils->knownRecomputeHeuristic.find(&call) !=\n"
     << "    gutils->knownRecomputeHeuristic.end()) {\n"
     << "    if (!gutils->knownRecomputeHeuristic[&call]) {\n"
     << "     auto cv = gutils->cacheForReverse(BuilderZ, newCall,\n"
     << "     getIndex(&call, CacheType::Self, BuilderZ));\n"
     << "    }\n"
     << "  } else if (Mode == DerivativeMode::ReverseModeGradient) {        \n"
     << "    eraseIfUnused(call, /*erase*/ true, /*check*/ false);          \n"
     << "  } else {                                                         \n"
     << "    eraseIfUnused(call);                                           \n"
     << "  }                                                                \n"
     << "  return true;\n"
     << "#ifdef __clang__\n"
     << "#pragma clang diagnostic pop\n"
     << "#else\n"
     << "#pragma GCC diagnostic pop\n"
     << "#endif\n"
     << "}\n\n";
}

void emit_helper(const TGPattern &pattern, raw_ostream &os) {
  auto nameVec = pattern.getArgNames();
  assert(nameVec.size() > 0);
  auto argTypeMap = pattern.getArgTypeMap();
  bool lv23 = pattern.isBLASLevel2or3();
  const auto mutArgSet = pattern.getMutableArgs();

  os << "  const bool byRef = blas.prefix == \"\" || blas.prefix == "
        "\"cublas_\";\n";
  os << "  const bool cblas = blas.prefix == \"cblas_\";\n";
  os << "  const bool cublas = blas.prefix == \"cublas_\" || blas.prefix == "
        "\"cublas\";\n";
  os << "  Value *cacheval = nullptr;\n\n";
  // lv 2 or 3 functions have an extra arg under the cblas_ abi
  os << "  const int offset = (";
  if (lv23) {
    os << "(cblas || cublas)";
  } else {
    os << "cublas";
  }
  os << " ? 1 : 0);\n";

  os << "// Next ones shall only be called in the cublas case,\n"
     << "// they have incorrect meaning otherwise\n"
     << "  const int pos_handle = 0;\n"
     << "  Value *orig_handle = nullptr;\n"
     << "  Value *arg_handle = nullptr;\n"
     << "  Type *type_handle = nullptr;\n"
     << "  bool overwritten_handle = true;\n"
     << "  if (cublas) {\n"
     << "    orig_handle = call.getArgOperand(pos_handle);\n"
     << "    arg_handle = gutils->getNewFromOriginal(orig_handle);\n"
     << "    type_handle = arg_handle->getType();\n"
     << "    overwritten_handle"
     << " = (cacheMode ? overwritten_args[pos_handle] : false);\n\n"
     << "  }\n\n";
  if (lv23) {
    auto name = nameVec[0];
    os << "// Next ones shall only be called in the cblas case,\n"
       << "// they have incorrect meaning otherwise\n"
       << "  const int pos_" << name << " = 0;\n"
       << "  const auto orig_" << name << " = call.getArgOperand(pos_" << name
       << ");\n"
       << "  auto arg_" << name << " = gutils->getNewFromOriginal(orig_" << name
       << ");\n"
       << "  const auto type_" << name << " = arg_" << name << "->getType();\n"
       << "  const bool overwritten_" << name
       << " = (cacheMode ? overwritten_args[pos_" << name << "] : false);\n\n";
  }

  auto actArgs = pattern.getActiveArgs();
  for (size_t i = (lv23 ? 1 : 0); i < nameVec.size(); i++) {
    auto name = nameVec[i];
    size_t j = (lv23 ? i - 1 : i);
    os << "  const int pos_" << name << " = " << j << " + offset;\n"
       << "  const auto orig_" << name << " = call.getArgOperand(pos_" << name
       << ");\n"
       << "  auto arg_" << name << " = gutils->getNewFromOriginal(orig_" << name
       << ");\n"
       << "  const auto type_" << name << " = arg_" << name << "->getType();\n"
       << "  const bool overwritten_" << name
       << " = (cacheMode ? overwritten_args[pos_" << name << "] : false);\n";
    if (std::count(actArgs.begin(), actArgs.end(), i)) {
      os << "  bool active_" << name << " = !gutils->isConstantValue(orig_"
         << name << ");\n"
         << "  Value *rt_inactive_" << name << " = nullptr;\n";
    }
    os << "\n";
  }
  if (get_blas_ret_ty(pattern.getName()) == "fpType") {
    os << "  if (cublas) {\n"
       << "    const int pos_ret = " << nameVec.size() << ";\n"
       << "    const auto orig_ret = call.getArgOperand(pos_ret);\n"
       << "    auto arg_ret = gutils->getNewFromOriginal(orig_ret);\n"
       << "    const auto type_ret = arg_ret->getType();\n"
       // TODO: check next line
       << "    const bool overwritten_ret = (cacheMode ? "
          "overwritten_args[pos_ret] : false);\n"
       << "    bool active_ret = !gutils->isConstantValue(orig_ret);\n"
       << "    Value *rt_inactive_ret = nullptr;\n"
       << "  }\n\n";
  }

  os << "\n  // <X> is inactive either if gutils->isConstantValue(<X>)\n"
     << "  // returns true, or if runtimeActivity is on and the\n"
     << "  // shadow points to the primal arg.\n";

  os << "  if(EnzymeRuntimeActivityCheck && cacheMode) {\n";
  for (size_t i = 0; i < actArgs.size(); i++) {
    auto name = nameVec[actArgs[i]];

    // floats are passed by calue, except of the Fortran Abi (byRef)
    auto ty = argTypeMap.lookup(actArgs[i]);
    os << "    if (";
    if (ty == ArgType::fp)
      os << "byRef && ";
    os << "active_" << name << ") {\n"
       << "      auto shadow_" << name << " = gutils->invertPointerM(orig_"
       << name << ", BuilderZ);\n"
       << "      rt_inactive_" << name << " = BuilderZ.CreateICmpEQ(shadow_"
       << name << ", arg_" << name << ", \"rt.tmp.inactive.\" \"" << name
       << "\");\n"
       << "    }\n";
  }
  // Blas functions return one float XOR modify one output arg.
  // If we have runtimeActivity and the output arg is inactive,
  // we don't need to do anything here and can return early.
  if (mutArgSet.size() == 1) {
    for (auto pos : mutArgSet) {
      auto name = nameVec[pos];
      os << "    Value *rt_inactive_out = nullptr;\n";
      os << "    if (active_" << name << ") {\n"
         << "      rt_inactive_out = rt_inactive_" << name << ";\n"
         << "    } else {\n"
         << "      rt_inactive_out = "
            "ConstantInt::getTrue(BuilderZ.getContext());\n"
         << "    }\n";
      break;
    }
    for (size_t i = 0; i < actArgs.size(); i++) {
      auto name = nameVec[actArgs[i]];
      // floats are passed by calue, except of the Fortran Abi (byRef)
      auto ty = argTypeMap.lookup(actArgs[i]);
      os << "    if (";
      if (ty == ArgType::fp)
        os << "byRef && ";
      os << "active_" << name << ") {\n"
         << "      rt_inactive_" << name << " = BuilderZ.CreateOr(rt_inactive_"
         << name << ", rt_inactive_out, \"rt.inactive.\" \"" << name << "\");\n"
         << "    }\n";
    }
  }

  os << "  }\n";

  bool hasFP = false;
  for (auto name : enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto ty = argTypeMap.lookup(name.index());
    if (ty == ArgType::fp) {
      os << "  Type* blasFPType = type_" << name.value() << ";\n";
      hasFP = true;
      break;
    }
  }
  if (!hasFP)
    os << "  Type* blasFPType = byRef ? (Type*)PointerType::getUnqual(fpType) "
          ": (Type*)fpType;\n";

  bool hasChar = false;
  for (auto name : enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto ty = argTypeMap.lookup(name.index());
    if (ty == ArgType::diag || ty == ArgType::side || ty == ArgType::uplo ||
        ty == ArgType::trans) {
      os << "  Type* blasCharType = type_" << name.value() << ";\n";
      hasChar = true;
      break;
    }
  }
  if (!hasChar)
    os << "  Type* blasCharType = byRef ? "
          "(Type*) getInt8PtrTy(call.getContext()) : "
          "(Type*) Type::getInt8Ty(call.getContext());\n";

  for (auto name : enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto ty = argTypeMap.lookup(name.index());
    if (ty == ArgType::trans) {
      os << "  Type *cublasEnumType = nullptr;\n";
      os << "  if (cublas) cublasEnumType = type_" << name.value() << ";\n";
      break;
    }
  }

  bool hasInt = false;
  for (auto name : enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto ty = argTypeMap.lookup(name.index());
    if (ty == ArgType::len || ty == ArgType::vincInc || ty == ArgType::mldLD) {
      os << "  Type* blasIntType = type_" << name.value() << ";\n";
      hasInt = true;
      break;
    }
  }
  (void)hasInt;
  assert(hasInt);

  os << "  Type* cublas_retty = nullptr;\n"
     << "  Value* cublas_handle = nullptr;\n"
     << "  if (cublas) {\n"
     << "    cublas_retty = call.getFunctionType()->getReturnType();\n"
     << "    cublas_handle = call.getArgOperand(0);\n"
     << "  }\n";

  for (auto name : enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto ty = argTypeMap.lookup(name.index());
    if (isVecLikeArg(ty)) {
      os << "  const bool julia_decl = !type_" << name.value()
         << "->isPointerTy();\n";
      os << "  Type* type_vec_like = type_" << name.value() << ";\n";
      return;
    }
  }
  PrintFatalError("Blas function without vector or matrix?");
}

void emit_scalar_types(const TGPattern &pattern, raw_ostream &os) {
  // We only look at the type of the first integer showing up.
  // This allows to learn if we use Fortran abi (byRef) or cabi
  std::string name = "";
  bool foundInt = false;

  auto inputTypes = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();

  for (auto val : inputTypes) {
    if (val.second == ArgType::len) {
      foundInt = true;
      name = nameVec[val.first];
      break;
    }
  }
  (void)foundInt;
  assert(foundInt && "no int type found in blas call");

  os << "  // fpType already given by blas type (s, d, c, z) \n"
     << "  IntegerType *intType = dyn_cast<IntegerType>(type_" << name << ");\n"
     << "  // TODO: add Fortran testcases for Fortran ABI\n"
     << "  if (!intType) {\n"
     << "    const auto PT = cast<PointerType>(type_" << name << ");\n"
     << "    if (blas.is64)\n"
     << "      intType = IntegerType::get(PT->getContext(), 64);\n"
     << "    else\n"
     << "      intType = IntegerType::get(PT->getContext(), 32);\n"
     << "  }\n\n"
     << "  IntegerType *charType = IntegerType::get(intType->getContext(), "
        "8);\n\n";
  os << "  IntegerType *julia_decl_type = nullptr;\n"
     << "  if (julia_decl)\n"
     << "    julia_decl_type = intType;\n";

  auto argTypeMap = pattern.getArgTypeMap();
  bool hasTrans = false;
  for (auto name : enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto ty = argTypeMap.lookup(name.index());
    if (ty == ArgType::trans) {
      hasTrans = true;
      break;
    }
  }
  if (hasTrans) {
    os << "  Value *valueN = nullptr;\n"
       << "  Value *valueT = nullptr;\n"
       << "  Value *valueG = nullptr;\n"
       << "  if (cublas) {\n"
       << "    valueN = ConstantInt::get(cublasEnumType, "
          "cublasOperation_t::CUBLAS_OP_N);\n"
       << "    valueT = ConstantInt::get(cublasEnumType, "
          "cublasOperation_t::CUBLAS_OP_T);\n"
       << "    // TODO lascl not available in cublas, nor op G\n"
       << "    valueG = ConstantInt::get(cublasEnumType, "
          "'G');\n"
       << "  } else {\n"
       << "    valueN = ConstantInt::get(charType, 'N');\n"
       << "    valueT = ConstantInt::get(charType, 'T');\n"
       << "    valueG = ConstantInt::get(charType, 'G');\n"
       << "  }\n\n";
  }
}

void extract_scalar(StringRef name, StringRef elemTy, raw_ostream &os) {
  os << "      if (cache_" << name << ") {\n"
     << "        arg_" << name << " = (cacheTypes.size() == 1)\n"
     << "                    ? cacheval\n"
     << "                    : Builder2.CreateExtractValue(cacheval, "
     << "{cacheidx}, \"tape.ext." << name << "\");\n"
     << "        auto alloc = allocationBuilder.CreateAlloca(" << elemTy
     << ", nullptr, \"byref." << name << "\");\n"
     << "        Builder2.CreateStore(arg_" << name << ", alloc);\n"
     << "        arg_" << name << " = Builder2.CreatePointerCast(\n"
     << "            alloc, type_" << name << ", \"cast." << name << "\");\n"
     << "        cacheidx++;\n"
     << "      }\n"
     << "\n";
}

void extract_input_mat(StringRef name, StringRef actName, raw_ostream &os) {
  os << "    if (active_" << actName << ") {\n"
     << "      input_" << name << " = (cacheTypes.size() == 1)\n"
     << "                  ? cacheval\n"
     << "                  : Builder2.CreateExtractValue(cacheval, "
        "{cacheidx}, \"tape.ext."
     << name << "\");\n"
     << "      free_input_" << name << " = input_" << name << ";\n"
     << "      if (type_" << name << "->isIntegerTy()) {\n"
     << "        input_" << name << " = Builder2.CreatePtrToInt(input_" << name
     << ", type_" << name << ");\n"
     << "      } else if (input_" << name << "->getType() != type_" << name
     << "){\n"
     << "        input_" << name << " = Builder2.CreatePointerCast(input_"
     << name << ", type_" << name << ");\n"
     << "      }\n"
     << "    }\n";
}

void extract_mat_or_vec(StringRef name, raw_ostream &os) {
  os << "    if (cache_" << name << ") {\n"
     << "      arg_" << name << " = (cacheTypes.size() == 1)\n"
     << "                  ? cacheval\n"
     << "                  : Builder2.CreateExtractValue(cacheval, "
        "{cacheidx}, \"tape.ext."
     << name << "\");\n"
     << "      free_" << name << " = arg_" << name << ";\n"
     << "      if (type_" << name << "->isIntegerTy()) {\n"
     << "        arg_" << name << " = Builder2.CreatePtrToInt(arg_" << name
     << ", type_" << name << ");\n"
     << "      } else if (arg_" << name << "->getType() != type_" << name
     << "){\n"
     << "        arg_" << name << " = Builder2.CreatePointerCast(arg_" << name
     << ", type_" << name << ");\n"
     << "      }\n"
     << "      cacheidx++;\n"
     << "    }\n";
}

void emit_extract_calls(const TGPattern &pattern, raw_ostream &os) {
  const auto typeMap = pattern.getArgTypeMap();
  const auto nameVec = pattern.getArgNames();
  const auto argUsers = pattern.getArgUsers();
  const auto activeArgs = pattern.getActiveArgs();
  auto rules = pattern.getRules();

  os << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
     << "      Mode == DerivativeMode::ReverseModeGradient ||\n"
     << "      Mode == DerivativeMode::ForwardModeSplit) {\n"
     << "\n"
     << "    if (cachetype) {\n"
     << "      if (Mode != DerivativeMode::ReverseModeCombined) {\n"
     << "        cacheval = BuilderZ.CreatePHI(cachetype, 0);\n"
     << "      }\n"
     << "      cacheval = gutils->cacheForReverse(\n"
     << "          BuilderZ, cacheval, getIndex(&call, CacheType::Tape, "
        "BuilderZ));\n"
     << "      if (Mode != DerivativeMode::ForwardModeSplit)\n"
     << "        cacheval = lookup(cacheval, Builder2);\n"
     << "    }\n"
     << "\n"
     << "    if (byRef) {\n";

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    auto name = nameVec[i];
    // this branch used "true_" << name everywhere instead of "arg_" << name
    // before. probably randomly, but check to make sure
    if (ty == ArgType::len || ty == ArgType::vincInc || ty == ArgType::mldLD) {
      extract_scalar(name, "intType", os);
    } else if (ty == ArgType::fp) {
      extract_scalar(name, "fpType", os);
    } else if (ty == ArgType::trans) {
      // we are in the byRef branch and trans only exist in lv23.
      // So just unconditionally asume that no layout exist and use i-1
      extract_scalar(name, "charType", os);
    }
  }
  os << "    }\n";

  std::string input_var = "";
  size_t actVar = 0;
  for (size_t a = 0; a < activeArgs.size(); a++) {
    auto rule = rules[a];
    const DagInit *ruleDag = rule.getRuleDag();
    std::string tmp = get_input_mat(ruleDag);
    if (tmp != "") {
      input_var = tmp;
      actVar = activeArgs[a];
      break;
    }
  }

  for (size_t j = 0; j < activeArgs.size(); j++) {
    size_t i = activeArgs[j];
    auto ty = typeMap.lookup(i);
    if (ty != ArgType::mldData && ty != ArgType::ap)
      continue;
    auto name = nameVec[i];
    auto rule = rules[j];
    auto input_mat_name = nameVec[actVar];
    if (name == input_var) {
      // we not only use arg_<X>, but also input_<X>
      extract_input_mat(name, input_mat_name, os);
    }
    extract_mat_or_vec(name, os);
    // TODO: corresponding LD should become matrix width?
  }

  for (size_t j = 0; j < activeArgs.size(); j++) {
    size_t i = activeArgs[j];
    if (typeMap.lookup(i) != ArgType::vincData)
      continue;
    auto name = nameVec[i];
    auto rule = rules[j];
    auto input_vec_name = nameVec[actVar];
    if (name == input_var) {
      os << "    //handling input\n";
      // we not only use arg_<X>, but also input_<X>
      extract_input_mat(name, input_vec_name, os);
    }
    extract_mat_or_vec(name, os);

    // caching a vector implies that the corresponding inc will now be 1.
    // We still don't overwritte it here, since it's shadow, or another var
    // might use it. So instead we insert a constantint 1 on the call site.
  }

  os << "  } else {\n"
     << "\n";

  for (size_t i = 0; i < nameVec.size(); i++) {
    if (typeMap.lookup(i) != ArgType::vincData)
      continue;
    auto vecName = nameVec[i];
    os << "    if (type_" << vecName << "->isIntegerTy())\n"
       << "      arg_" << vecName << " = Builder2.CreatePtrToInt(arg_"
       << vecName << ", type_" << vecName << ");\n";
  }

  os << "  }\n";
}

// Will be used by Julia
SmallString<80> ValueType_helper(const TGPattern &pattern, size_t actPos) {
  const auto nameVec = pattern.getArgNames();
  const auto typeMap = pattern.getArgTypeMap();
  SmallString<80> valueTypes{};

  // start with 1 since layout is only used for cblas (!byRef)
  for (size_t pos = 1; pos < nameVec.size();) {
    auto name = nameVec[pos];
    auto ty = typeMap.lookup(pos);

    if (pos > 1) {
      valueTypes.append(", ");
    }

    if (ty == ArgType::len) {
      valueTypes.append("ValueType::Both");
    } else if (ty == ArgType::fp) {
      auto floatName = nameVec[pos];
      if (pos == actPos) {
        valueTypes.append("ValueType::Both");
      } else {
        valueTypes.append((Twine("cache_") + floatName +
                           " ? ValueType::Both : ValueType::Both")
                              .str());
      }
    } else if (ty == ArgType::vincData) {
      const auto nextName = nameVec[pos + 1];
      // Check that the next should be an increment
      assert(typeMap.lookup(pos + 1) == ArgType::vincInc);
      const auto vecName = nameVec[pos];
      if (pos == actPos) {
        valueTypes.append("ValueType::Both, ValueType::Both");
      } else {
        valueTypes.append(
            (Twine("cache_") + vecName +
             " ? ValueType::Both : ValueType::Both, ValueType::Both")
                .str());
      }
      pos++; // extra inc, since vector cover two args (vincInc+vincData)
    } else {
      // TODO
      valueTypes.append("ValueType::Both");
      // llvm::errs() << "type: " << ty << "\n";
      // PrintFatalError("Unhandled type!");
    }
    pos++;
  }
  return valueTypes;
}

// TODO: think about how to handle nested rules which aren't simple calling
// another BLAS fnc.

size_t fwd_call_args(const TGPattern &pattern, size_t actArg,
                     SmallString<40> &result) {
  const auto nameVec = pattern.getArgNames();
  const auto nameMap = pattern.getArgNameMap();
  const auto typeMap = pattern.getArgTypeMap();
  const size_t startArg = pattern.isBLASLevel2or3() ? 1 : 0;

  // just replace argOps with rule
  // We start with 1 and conditionally add the cblas only first arg
  // only in the !byRef case
  for (size_t pos = startArg; pos < nameVec.size();) {
    if (pos > startArg) {
      result.append(", ");
    }

    const auto name = nameVec[pos];
    // get the position of this argument in the primary blas call
    assert(typeMap.count(pos) == 1);
    // and based on that get the fp/int + scalar/vector type
    auto ty = typeMap.lookup(pos);
    if (ty == ArgType::len) {
      result.append((Twine("arg_") + name).str());
    } else if (ty == ArgType::fp || ty == ArgType::ap ||
               ty == ArgType::vincData) {
      if (pos == actArg) {
        result.append((Twine("d_") + name).str());
      } else {
        result.append((Twine("arg_") + name).str());
      }
    } else if (ty == ArgType::vincInc) {
      if (pos - 1 == actArg) {
        // all ok, single inc after shadow of vec
        // use original inc, since shadow is never cached
        result.append((Twine("arg_") + name).str());
      } else {
        auto prevName = nameVec[pos - 1];
        result.append(
            (Twine("(cache_") + prevName + " ? const_one : arg_" + name + ")")
                .str());
      }
    } else if (ty == ArgType::mldData) {
      auto nextName = nameVec[pos + 1];
      // get the position of the argument in the primary blas call
      auto nextArgPosition = nameMap.lookup(nextName);
      // and based on that get the fp/int + scalar/vector type
      auto nextTy = typeMap.lookup(nextArgPosition);
      (void)nextTy;
      assert(nextTy == ArgType::mldLD);
      if (pos == actArg) {
        result.append((Twine("d_") + name + ", true_" + nextName).str());
      } else {
        result.append((Twine("arg_") + name + ", arg_" + nextName).str());
      }
      pos++; // extra ++ due to also handling mldLD
    } else if (ty == ArgType::mldLD) {
      // might come without mldData, e.g. after DiffeRet
      // coppied from vincInc, but should verify if actually needed
      result.append((Twine("arg_") + name).str());
    } else if (ty == ArgType::cblas_layout) {
      // layout is only allowed as first type (which we skipped)
      errs() << "name: " << name << " typename: " << ty
             << " only allowed as first arg!\n";
      llvm_unreachable("layout only allowed as first type!\n");
    } else if (ty == ArgType::trans || ty == ArgType::diag ||
               ty == ArgType::uplo || ty == ArgType::side) {
      result.append((Twine("arg_") + name).str());
    } else {
      errs() << "name: " << name << " typename: " << ty << "\n";
      llvm_unreachable("unimplemented input type in fwd mode!\n");
    }
    pos++;
  }

  // return the size - 1 due to only using the cblas_layout in the !byRef case
  return nameVec.size() - startArg;
}

void emit_fwd_rewrite_rules(const TGPattern &pattern, raw_ostream &os) {
  auto rules = pattern.getRules();
  bool lv23 = pattern.isBLASLevel2or3();
  os << "  /* fwd-rewrite */                                 \n"
     << "  if (Mode == DerivativeMode::ForwardMode ||        \n"
     << "      Mode == DerivativeMode::ForwardModeSplit) {   \n"
     << "                                                    \n"
     << "    auto callval = call.getCalledOperand();       \n\n";

  os << "  if (EnzymeRuntimeActivityCheck) {\n"
     << "    std::string s;\n"
     << "    llvm::raw_string_ostream ss(s);\n"
     << "    ss << \"" << pattern.getName() << "\" << \"\\n\";\n"
     << "    ss << call.getDebugLoc() << \"\\n\";\n"
     << "    ss << \"Runtime Activity not supported for BLAS calls\" << "
        "\"\\n\";\n"
     << "    if (CustomErrorHandler) {\n"
     << "      IRBuilder<> Builder2(&call);\n"
     << "      getForwardBuilder(BuilderZ);\n"
     << "      CustomErrorHandler(ss.str().c_str(), wrap(&call), "
        "ErrorType::NoDerivative,\n"
     << "                         gutils, nullptr, wrap(&BuilderZ));\n"
     << "      return false;\n"
     << "    } else {\n"
     << "      EmitFailure(\"Unsupported Mode\", call.getDebugLoc(), &call, "
        "ss.str());\n"
     << "      return false;\n"
     << "    }\n"
     << "  }\n";

  // just make this const one available now to have less variable name repition
  os << "Value * const_one = to_blas_callconv(Builder2, "
        "ConstantInt::get(intType, 1), "
     << "byRef, cublas, intType, allocationBuilder, \"int.one\");\n";

  const auto nameVec = pattern.getArgNames();
  const auto inputTypes = pattern.getArgTypeMap();
  const auto activeArgs = pattern.getActiveArgs();
  for (auto inputType : inputTypes) {
    auto ty = inputType.second;
    if (isVecLikeArg(ty)) {
      const auto name = nameVec[inputType.first];
      os << "    Value *d_" << name << " = active_" << name << "\n"
         << "     ? gutils->invertPointerM(orig_" << name << ", Builder2)\n"
         << "     : nullptr;\n";
    }
    if (ty == ArgType::fp) {
      const auto name = nameVec[inputType.first];
      os << "    Value *d_" << name
         << " = Constant::getNullValue(gutils->getShadowType(fpType));\n";
    }
  }

  os << "    Value *dres = applyChainRule(\n"
     << "        call.getType(), Builder2,\n"
     << "        [&](";
  bool first = true;
  for (auto activeArg : activeArgs) {
    auto name = nameVec[activeArg];
    os << ((first) ? "" : ", ") << "Value *d_" << name;
    first = false;
  }
  os << "  ) {\n"
     << "      Value *dres = nullptr;\n";

  for (size_t i = 0; i < activeArgs.size(); i++) {
    const auto activeArg = activeArgs[i];
    const auto rule = rules[i];
    const auto actName = nameVec[activeArg];
    auto dcallArgs = SmallString<40>();
    const size_t numArgs = fwd_call_args(pattern, activeArg, dcallArgs);
    const auto valueTypes = ValueType_helper(pattern, activeArg);
    os << "      if(active_" << actName << ") {\n";

    if (lv23) {
      // add extra cblas_arg for the !byRef case
      os << "        Value *args1_cblas[" << numArgs + 1 << "] = "
         << " {arg_layout, " << dcallArgs << "};\n";
      os << "        auto Defs_cblas = gutils->getInvertedBundles(\n"
         << "          &call, {ValueType::Both, " << valueTypes
         << "}, Builder2, /* lookup */ false);\n";
    }
    os << "        Value *args1[" << numArgs << "] = {" << dcallArgs << "};\n";
    os << "        auto Defs = gutils->getInvertedBundles(\n"
       << "          &call, {" << valueTypes
       << "}, Builder2, /* lookup */ false);\n";
    if (i == 0) {
      if (lv23) {
        os << "          if (byRef) {\n"
           << "          dres = Builder2.CreateCall(call.getFunctionType(), "
              "callval, args1, Defs);\n"
           << "          } else /*cblas*/ {\n"
           << "          dres = Builder2.CreateCall(call.getFunctionType(), "
              "callval, args1_cblas, Defs_cblas);\n"
           << "          };\n";
      } else {
        os << "          dres = Builder2.CreateCall(call.getFunctionType(), "
              "callval, args1, Defs);\n";
      }
    } else {
      os << "          Value *nextCall;\n";
      if (lv23) {
        os << "          if (byRef) {\n"
           << "        nextCall = Builder2.CreateCall(\n"
           << "          call.getFunctionType(), callval, args1, Defs);\n"
           << "          } else {\n"
           << "        nextCall = Builder2.CreateCall(\n"
           << "          call.getFunctionType(), callval, args1_cblas, "
              "Defs_cblas);\n"
           << "          }\n";
      } else {
        os << "        nextCall = Builder2.CreateCall(\n"
           << "          call.getFunctionType(), callval, args1, Defs);\n";
      }
      os << "        if (dres)\n"
         << "          dres = Builder2.CreateFAdd(dres, nextCall);\n"
         << "        else\n"
         << "          dres = nextCall;\n";
    }
    os << "      }\n";
  }
  os << "      return dres;\n"
     << "    },\n"
     << "    ";

  first = true;
  for (auto activeArg : activeArgs) {
    os << ((first) ? "" : ", ") << "d_" + nameVec[activeArg];
    first = false;
  }
  os << ");\n";
  os << "    setDiffe(&call, dres, Builder2);\n";
  os << "  }\n";
}

void emit_tmp_free(Record *Def, raw_ostream &os, StringRef builder) {
  const auto args = Def->getValueAsListOfStrings("args");
  // allocating tmp variables is optional, return if not required
  if (args.size() == 0)
    return;
  const auto matName = args[0];
  const auto allocName = "mat_" + matName;
  os << "    CreateDealloc(" << builder << ", true_" << allocName << ");\n";
}

void emit_tmp_creation(Record *Def, raw_ostream &os, StringRef builder) {
  const auto args = Def->getValueAsListOfStrings("args");
  // allocating tmp variables is optional, return if not required
  if (args.size() == 0)
    return;

  // First, let's prepare some cache for the vec or mat
  assert(args.size() >= 2);
  auto action = args[1];
  assert(action == "product" || action == "is_normal" ||
         action == "triangular");
  if (action == "product") {
    const auto matName = args[0];
    const auto dim1 = "arg_" + args[2];
    const auto dim2 = "arg_" + args[3];
    os << "    Value *len1 = load_if_ref(" << builder << ", intType," << dim1
       << ", byRef);\n"
       << "    Value *len2 = load_if_ref(" << builder << ", intType," << dim2
       << ", byRef);\n"
       << "    Value *size_" << matName << " = " << builder
       << ".CreateNUWMul(len1, len2, \"size_" << matName << "\");\n";
  } else if (action == "is_normal") {
    assert(args.size() == 5);
    const auto vecName = args[0];
    const auto trans = "arg_" + args[2];
    const auto dim1 = "arg_" + args[3];
    const auto dim2 = "arg_" + args[4];
    os << "    Value *len1 = load_if_ref(" << builder << ", intType," << dim1
       << ", byRef);\n"
       << "    Value *len2 = load_if_ref(" << builder << ", intType," << dim2
       << ", byRef);\n";
    os << "    Value *size_" << vecName << " = " << builder
       << ".CreateSelect(is_normal(" << builder << ", " << trans
       << ", byRef, cublas), len1, len2);\n";
  } else if (action == "triangular") {
    assert(args.size() == 3);
    const auto vecName = args[0];
    const auto dim1 = "arg_" + args[2];
    os << "    Value *len = load_if_ref(" << builder << ", intType," << dim1
       << ", byRef);\n";
    //  Size has to be (at least)
    //  ( ( n*( n + 1 ) )/2 )
    os << "    Value *size_" << vecName << " = " << builder
       << ".CreateMul(len, " << builder
       << ".CreateAdd(len, "
          "ConstantInt::get(intType, 1)), \"square_mat_size_"
       << vecName << "\");\n"
       << "    size_" << vecName << " = " << builder << ".CreateUDiv(size_"
       << vecName << ", ConstantInt::get(intType, 2), \"size_" << vecName
       << "\");\n";
  }
  const auto matName = args[0];
  const auto allocName = "mat_" + matName;
  os << "    Value * true_" << allocName << " = CreateAllocation(" << builder
     << ", fpType, size_" << matName << ", \"" << allocName << "\");\n"
     << "    Value * " << allocName << " = true_" << allocName << ";\n"
     << "    if (type_vec_like->isIntegerTy()) {\n"
     << "      " << allocName << " = " << builder << ".CreatePtrToInt("
     << allocName << ", type_vec_like);\n"
     << "    } else if (" << allocName << "->getType() != type_vec_like){\n"
     << "      " << allocName << " = " << builder << ".CreatePointerCast("
     << allocName << ", type_vec_like);\n"
     << "    }\n";
}

void emit_deriv_rule(const StringMap<TGPattern> &patternMap, Rule &rule,
                     StringSet<> &handled, raw_ostream &os) {
  const auto ruleDag = rule.getRuleDag();
  const auto opName = ruleDag->getOperator()->getAsString();
  const auto nameMap = rule.getArgNameMap();
  const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
  if (Def->isSubClassOf("b")) {
    // emit_deriv_blas_call(ruleDag, patternMap, handled, os);
  } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "noop") {
    // nothing to prepare
  } else if (Def->isSubClassOf("DiffeRetIndex")) {
    // nothing to prepare
  } else if (Def->isSubClassOf("Inst")) {
    PrintFatalError("Unhandled Inst Rule!");
    // TODO:
    return;
  } else if (Def->isSubClassOf("Seq")) {
    // handle seq rules
    for (size_t i = 0; i < ruleDag->getNumArgs(); i++) {
      Init *subArg = ruleDag->getArg(i);
      DagInit *sub_Dag = cast<DagInit>(subArg);
      if (auto sub_def = dyn_cast<DefInit>(sub_Dag->getOperator())) {
        const auto sub_Def = sub_def->getDef();
        if (sub_Def->isSubClassOf("b")) {
          os << "    //handling nested blas: " << std::to_string(i) << "\n";
          // emit_deriv_blas_call(sub_Dag, patternMap, handled, os);
          os << "    //handled nested blas: " << std::to_string(i) << "\n";
        } else if (sub_Def->isSubClassOf("FrobInnerProd")) {
          // nothing to prepare
          assert(sub_Dag->getNumArgs() == 4);
        } else if (sub_Def->isSubClassOf("DiagUpdateSPMV")) {
          // nothing to prepare
          assert(sub_Dag->getNumArgs() == 6);
        }
      }
    }
  } else if (Def->isSubClassOf("FrobInnerProd")) {
    // nothing to prepare
    assert(ruleDag->getNumArgs() == 4);
  } else if (Def->isSubClassOf("DiagUpdateSPMV")) {
    // nothing to prepare
    assert(ruleDag->getNumArgs() == 6);
  } else {
    PrintFatalError("Unhandled deriv Rule!");
  }
}

// Emit the corresponding code rom (ruleDag arg # pos), given
// that the arg being differentiated is argAct.
// The map offsetToBaseNames takes vinc, ld, and maps them to
// the arg name of the original vector/matrix
void rev_call_arg(DagInit *ruleDag, Rule &rule, size_t actArg, size_t pos,
                  raw_ostream &os) {
  const auto nameMap = rule.getArgNameMap();
  const auto typeMap = rule.getArgTypeMap();
  auto arg = ruleDag->getArg(pos);
  if (auto Dag = dyn_cast<DagInit>(arg)) {
    auto Def = cast<DefInit>(Dag->getOperator())->getDef();

    if (Def->isSubClassOf("MagicInst")) {
      if (Def->getName() == "Rows") {
        os << "({";
        for (size_t i = Dag->getNumArgs() - 1;; i--) {
          os << "auto brow_" << i << " = ";
          rev_call_arg(Dag, rule, actArg, i, os);
          os << "; ";
          if (i == 0)
            break;
        }
        os << "get_blas_row(Builder2, ";
        for (size_t i = 0; i < Dag->getNumArgs(); i++) {
          os << "brow_" << i;
          os << ", ";
        }
        os << "byRef, cublas);})";
        return;
      }
      if (Def->getName() == "Concat") {
        os << "({";
        for (size_t i = 0; i < Dag->getNumArgs(); i++) {
          os << "auto concat_" << i << " = ";
          rev_call_arg(Dag, rule, actArg, i, os);
          os << "; ";
        }
        os << "concat_values<";
        for (size_t i = 0; i < Dag->getNumArgs(); i++) {
          if (i != 0)
            os << ", ";
          os << "ArrayRef<Value*>";
        }
        os << ">(";
        for (size_t i = 0; i < Dag->getNumArgs(); i++) {
          if (i != 0)
            os << ", ";
          os << "concat_" << i;
        }
        os << "); })";
        return;
      }
      if (Def->getName() == "ld") {
        assert(Dag->getNumArgs() == 5);
        //(ld $A, $transa, $lda, $m, $k)
        const auto ldName = Dag->getArgNameStr(2);
        const auto dim1Name = Dag->getArgNameStr(3);
        const auto dim2Name = Dag->getArgNameStr(4);
        const auto matName = Dag->getArgNameStr(0);
        os << "{get_cached_mat_width(Builder2, ";
        rev_call_arg(Dag, rule, actArg, 1, os);
        os << ", arg_" << ldName << ", arg_" << dim1Name << ", arg_" << dim2Name
           << ", cache_" << matName << ", byRef, cublas)}";
        return;
      }
    }

    errs() << Def->getName() << "\n";
    PrintFatalError("Dag/Def that isn't a DiffeRet!!");
  } else if (DefInit *DefArg = dyn_cast<DefInit>(arg)) {
    auto Def = DefArg->getDef();
    if (Def->isSubClassOf("DiffeRetIndex")) {
      os << "{dif}";
    } else if (Def->isSubClassOf("adj")) {
      auto name = Def->getValueAsString("name");
      os << "{d_" << name;
      size_t argPosition = (size_t)(-1);
      for (size_t i = 0; i < rule.nameVec.size(); i++) {
        if (rule.nameVec[i] == name) {
          argPosition = i;
          break;
        }
      }
      if (argPosition == (size_t)(-1)) {
        errs() << "couldn't find name: " << name << " ap=" << argPosition
               << "\n";
        PrintFatalError("arg not in inverted nameMap!");
      }
      auto ty = rule.argTypesFull.lookup(argPosition);
      auto incName = rule.nameVec[argPosition + 1];
      if (ty == ArgType::vincData || ty == ArgType::mldData)
        os << ", arg_" << incName;
      else
        assert(ty == ArgType::fp || ty == ArgType::ap);
      os << "}";
    } else if (Def->isSubClassOf("input")) {
      auto name = Def->getValueAsString("name");
      os << "{input_" << name;
      size_t argPosition = (size_t)(-1);
      for (size_t i = 0; i < rule.nameVec.size(); i++) {
        if (rule.nameVec[i] == name) {
          argPosition = i;
          break;
        }
      }
      if (argPosition == (size_t)(-1)) {
        errs() << "couldn't find name: " << name << " ap=" << argPosition
               << "\n";
        PrintFatalError("arg not in inverted nameMap!");
      }
      auto ty = rule.argTypesFull.lookup(argPosition);
      auto incName = rule.nameVec[argPosition + 1];
      if (ty == ArgType::vincData)
        os << ", (cache_" << name << " ? const_one : arg_" << incName << ")";
      else
        assert(ty == ArgType::fp || ty == ArgType::ap ||
               ty == ArgType::mldData);
      os << "}";
    } else if (Def->isSubClassOf("use")) {
      auto name = Def->getValueAsString("name");
      os << "{mat_" << name << "}";
    } else if (Def->isSubClassOf("Constant")) {
      auto val = Def->getValueAsString("value");
      os << "{to_blas_fp_callconv(Builder2, ConstantFP::get(fpType, " << val
         << "), byRef, blasFPType, allocationBuilder, \"constant.fp." << val
         << "\")}";
    } else if (Def->isSubClassOf("Char")) {
      auto val = Def->getValueAsString("value");
      if (val == "N") {
        os << "{to_blas_callconv(Builder2, valueN, byRef, cublas, nullptr, "
              "allocationBuilder, \"constant.char.N\")}";
      } else if (val == "T") {
        os << "{to_blas_callconv(Builder2, valueT, byRef, cublas, nullptr, "
              "allocationBuilder, \"constant.char.T\")}";
      } else if (val == "G") {
        os << "{to_blas_callconv(Builder2, valueG, byRef, cublas, nullptr, "
              "allocationBuilder, \"constant.char.G\")}";
        // C is not supported yet
        //} else if (val == "C") {
      } else {
        errs() << "unknown char: " << val << "\n";
        PrintFatalError("unknown char");
      }
    } else if (Def->isSubClassOf("Alloca")) {
      auto val = Def->getValueAsInt("value");
      (void)val;
      assert(val == 1);
      os << "{allocationBuilder.CreateAlloca(intType)}";
    } else if (Def->isSubClassOf("ConstantInt")) {
      auto val = Def->getValueAsInt("value");
      os << "{to_blas_callconv(Builder2, ConstantInt::get(intType, " << val
         << "), byRef, cublas, intType, allocationBuilder, \"constant.int."
         << val << "\")}";
    } else if (Def->isSubClassOf("transpose")) {
      auto name = Def->getValueAsString("name");
      os << "{(arg_transposed_" << name << " = arg_transposed_" << name
         << " ? arg_transposed_" << name << " : "
         << "transpose(Builder2, arg_" << name
         << ", byRef, cublas, charType, allocationBuilder, \"" << name
         << "\"))}";
    } else {
      errs() << Def->getName() << "\n";
      PrintFatalError("Def that isn't a DiffeRet!");
    }
  } else {
    auto name = ruleDag->getArgNameStr(pos);
    if (name == "") {
      PrintFatalError("arg has no name!" + std::to_string(pos));
      assert(name != "");
    }
    // get the position of the argument in the primary blas call
    if (nameMap.count(name) != 1) {
      errs() << "couldn't find name: " << name << "\n";
      PrintFatalError("arg not in nameMap!");
    }
    assert(nameMap.count(name) == 1);
    auto argPosition = nameMap.lookup(name);
    // and based on that get the fp/int + scalar/vector type
    auto ty = typeMap.lookup(argPosition);

    switch (ty) {
    case ArgType::cblas_layout:
    case ArgType::len:
    case ArgType::fp:
    case ArgType::ap:
    case ArgType::trans:
    case ArgType::diag:
    case ArgType::uplo:
    case ArgType::side:
    case ArgType::vincInc:
    case ArgType::vincData:
    case ArgType::mldData: {
      os << "{";
      if (argPosition == actArg) {
        os << "d_" << name;
      } else {
        os << "arg_" << name;
      }
      if (ty == ArgType::vincData) {
        auto incName = rule.nameVec[argPosition + 1];
        os << ", (cache_" << name << " ? const_one : arg_" << incName << ")";
      }
      if (ty == ArgType::mldData) {
        auto ldName = rule.nameVec[argPosition + 1];
        if (argPosition == actArg) {
          os << ", true_" << ldName;
        } else {
          // if this matrix got cached, we need more complex logic
          // to determine the next arg. Thus handle it once we reach it
        }
      }

      os << "}";
      return;
    }
    default:
      errs() << "name: " << name << " typename: " << ty << "\n";
      llvm_unreachable("unimplemented input type in reverse mode!\n");
    }
  }
}

// fill the result string and return the number of added args
void rev_call_args(StringRef argName, Rule &rule, size_t actArg,
                   raw_ostream &os, int subRule, StringRef func) {

  const auto nameMap = rule.getArgNameMap();

  auto ruleDag = rule.getRuleDag();
  size_t numArgs = ruleDag->getNumArgs();

  if (subRule != -1) {
    // handle Seq
    ruleDag = cast<DagInit>(ruleDag->getArg(subRule));
    numArgs = ruleDag->getNumArgs();
  }

  os << "        std::vector<Value *>" << argName << ";\n";

  // layout exist only under the cBLas ABI and not for all fncs.
  bool fncHasLayout = (ruleDag->getArgNameStr(0) == "layout");
  if (fncHasLayout) {
    // Fnc has a layout if cBLAS, that makes it more complex.
    // Distinguish later trough byRef if it is cblas (thus has layout)
    os << "        if (cblas) " << argName << ".push_back(arg_layout);\n";
  }
  os << "        if (cublas) " << argName << ".push_back(arg_handle);\n";

  for (size_t pos = fncHasLayout ? 1 : 0; pos < numArgs; pos++) {
    os << "        for (auto item : ";
    rev_call_arg(ruleDag, rule, actArg, pos, os);
    os << ") " << argName << ".push_back(item);\n";
  }
  os << "        if (byRef) {\n";
  int n = 0;
  if (func == "gemv" || func == "lascl")
    n = 1;
  if (func == "gemm")
    n = 2;
  for (int i = 0; i < n; i++)
    os << "           " << argName
       << ".push_back(ConstantInt::get(intType, 1));\n";
  os << "        }\n";
}

void emit_fret_call(StringRef dfnc_name, StringRef argName, StringRef name,
                    StringRef bb, raw_ostream &os) {
  os << "{\n";
  if (dfnc_name == "inner_prod") {
    os << "    auto derivcall_inner_prod = \n"
          "      getorInsertInnerProd("
       << bb
       << ", "
          "*gutils->oldFunc->getParent(), blas, intType, type_vec_like, "
          "type_n, fpType, "
       << argName << ", Defs, byRef, cublas, julia_decl);\n"
       << "        CallInst *cubcall = "
          "cast<CallInst>(derivcall_inner_prod);\n";
  } else {
    os << "      SmallVector<Type*, 1> tys; for (auto arg : " << argName
       << ") tys.push_back(arg->getType());\n";
    std::string dfnc_ret_ty = get_blas_ret_ty(dfnc_name);
    os << "    llvm::FunctionType *FT" << dfnc_name << " = FunctionType::get("
       << dfnc_ret_ty << ", tys, false);\n";
    os << "    auto derivcall_" << dfnc_name
       << " = gutils->oldFunc->getParent()->getOrInsertFunction(\n"
       << "  blas.prefix + blas.floatType + \"" << dfnc_name
       << "\" + blas.suffix, FT" << dfnc_name << ");\n";

    os << "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name
       << ".getCallee()))\n"
       << "    {\n"
       << "      attribute_" << dfnc_name << "(blas, F);\n"
       << "    }\n\n";
    os << "        CallInst *cubcall = "
          "cast<CallInst>("
       << bb << ".CreateCall(derivcall_" << dfnc_name << ", " << argName
       << ", Defs));\n";
  }
  os << "        if (byRef) {\n"
     << "          ((DiffeGradientUtils *)gutils)"
     << "->addToInvertedPtrDiffe(&call, nullptr, fpType, 0, "
     << "(called->getParent()->getDataLayout().getTypeSizeInBits(fpType)/8), "
        "orig_"
     << name << ", cubcall, " << bb << ");\n"
     << "        } else {\n"
     << "          addToDiffe(orig_" << name << ", cubcall, " << bb
     << ", fpType);\n"
     << "        }\n";
  os << "}\n";
}

// todo: update rt_active_<X> to use actual dag requirements,
// possibly by or-ing them
void emit_runtime_condition(DagInit *ruleDag, StringRef name, StringRef tab,
                            StringRef B, bool isFP, raw_ostream &os) {
  os << tab << "BasicBlock *nextBlock_" << name << " = nullptr;\n"
     << tab << "if (EnzymeRuntimeActivityCheck && cacheMode"
     << (isFP ? " && byRef" : "") << ") {\n"
     << tab << "  BasicBlock *current = Builder2.GetInsertBlock();\n"
     << tab << "  auto activeBlock = gutils->addReverseBlock(current,"
     << "bb_name + \"." << name << ".active\");\n"
     << tab << "  nextBlock_" << name << " = gutils->addReverseBlock("
     << "activeBlock, bb_name + \"." << name << ".done\");\n"
     << tab << "  " << B << ".CreateCondBr(rt_inactive_" << name
     << ", nextBlock_" << name << ", activeBlock);\n"
     << tab << "  " << B << ".SetInsertPoint(activeBlock);\n"
     << tab << "}\n";
}

void emit_runtime_continue(DagInit *ruleDag, StringRef name, StringRef tab,
                           StringRef B, bool isFP, raw_ostream &os) {
  os << tab << "if (nextBlock_" << name << (isFP ? " && byRef" : "") << ") {\n"
     << tab << "  " << B << ".CreateBr(nextBlock_" << name << ");\n"
     << tab << "  " << B << ".SetInsertPoint(nextBlock_" << name << ");\n"
     << tab << "}\n";
}

void if_rule_condition_inner(DagInit *ruleDag, StringRef name, StringRef tab,
                             raw_ostream &os, llvm::StringSet<> &seen) {
  for (size_t pos = 0; pos < ruleDag->getNumArgs();) {
    Init *arg = ruleDag->getArg(pos);
    if (DefInit *DefArg = dyn_cast<DefInit>(arg)) {
      auto Def = DefArg->getDef();
      if (Def->isSubClassOf("adj")) {
        auto name = Def->getValueAsString("name");
        seen.insert(name);
      }
    } else if (auto sub_Dag = dyn_cast<DagInit>(arg)) {
      if_rule_condition_inner(sub_Dag, name, tab, os, seen);
    }
    pos++;
  }
}

// primal arguments are always available,
// shadow arguments (d_<X>) might not, so check if they are active
void emit_if_rule_condition(DagInit *ruleDag, StringRef name, StringRef tab,
                            raw_ostream &os) {
  llvm::StringSet<> seen = llvm::StringSet<>();

  if_rule_condition_inner(ruleDag, name, tab, os, seen);

  // this will only run once, at the end of the outermost call
  os << tab << "if (active_" << name;
  for (auto name : seen.keys()) {
    os << " && d_" << name.str();
  }
  os << ") {\n";
}

void emit_rev_rewrite_rules(const StringMap<TGPattern> &patternMap,
                            TGPattern &pattern, raw_ostream &os) {

  const auto nameVec = pattern.getArgNames();
  const auto typeMap = pattern.getArgTypeMap();
  const auto rules = pattern.getRules();
  const auto activeArgs = pattern.getActiveArgs();
  const bool lv23 = pattern.isBLASLevel2or3();

  // If any of the rule uses DiffeRet, the primary function has a ret val
  // and we should emit the code for handling it.
  bool hasDiffeRetVal = false;
  for (auto derivOp : rules) {
    hasDiffeRetVal |= hasDiffeRet(derivOp.getRuleDag());
  }

  os << "  /* rev-rewrite */                                 \n"
     << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
     << "      Mode == DerivativeMode::ReverseModeGradient) {\n"
     << "    Value *alloc = nullptr;\n"
     << "    if (byRef && !cublas) {\n"
     << "      alloc = allocationBuilder.CreateAlloca(fpType, nullptr, "
        "\"ret\");\n"
     << "    }\n\n";

  if (hasDiffeRetVal) {
    os << "    Value *dif = cublas ? gutils->invertPointerM(call.getArgOperand("
       << typeMap.size() << " + offset), Builder2) : diffe(&call, Builder2);\n";
  }

  // We only emit one derivcall per blass call type.
  // This verifies that we don't end up with multiple declarations.
  StringSet<> handled{};
  for (auto rule : rules) {
    emit_deriv_rule(patternMap, rule, handled, os);
  }

  for (size_t i = 0; i < nameVec.size(); i++) {
    const auto name = nameVec[i];
    const auto ty = typeMap.lookup(i);
    if (isVecLikeArg(ty)) {
      os << "    Value *d_" << name << " = active_" << name << "\n"
         << "     ? lookup(gutils->invertPointerM(orig_" << name
         << ", Builder2), Builder2)\n"
         << "     : nullptr;\n";
    } else if (ty == ArgType::fp) {
      os << "    Value *d_" << name << " = UndefValue::get(fpType);\n";
    }
  }

  // We need to lookup all args which we haven't cached or overwritten and which
  // are required.
  for (size_t i = 0; i < nameVec.size(); i++) {
    const auto name = nameVec[i];
    const auto ty = typeMap.lookup(i);
    // those do have special handling
    if (ty != ArgType::cblas_layout) {
      os << "    if (!cache_" << name << " && need_" << name << ")\n"
         << "      arg_" << name << " = lookup(arg_" << name
         << ", Builder2);\n";
    }
  }

  os << "  if(EnzymeRuntimeActivityCheck && cacheMode) {\n";
  for (size_t i = 0; i < activeArgs.size(); i++) {
    auto name = nameVec[activeArgs[i]];

    // floats are passed by calue, except of the Fortran Abi (byRef)
    auto ty = typeMap.lookup(activeArgs[i]);
    os << "    if (";
    if (ty == ArgType::fp)
      os << "byRef && ";
    os << "active_" << name << ") {\n"
       << "      rt_inactive_" << name << " = lookup(rt_inactive_" << name
       << ", Builder2);\n"
       << "    }\n";
  }
  os << "  }\n";

  // now we can use it to transpose our trans arguments if they exist
  for (size_t i = (lv23 ? 1 : 0); i < nameVec.size(); i++) {
    auto name = nameVec[i];
    if (typeMap.lookup(i) == ArgType::trans) {
      os << "    llvm::Value* arg_transposed_" << name << " = nullptr;\n";
    }
  }

  os << "    applyChainRule(\n"
     << "      Builder2,\n"
     << "      [&](";
  bool first = true;
  for (auto arg : activeArgs) {
    const auto name = nameVec[arg];
    const auto ty = typeMap.lookup(arg);
    // We don't pass in shadows of fp values,
    // we just create and struct-return the shadows
    if (ty == ArgType::fp)
      continue;
    os << ((first) ? "" : ", ") << "Value *"
       << "d_" + name;
    first = false;
  }

  if (hasDiffeRetVal) {
    os << ((first) ? "" : ", ") << "Value *dif) {\n"
       << "        if (byRef && !cublas) {\n"
       << "          Builder2.CreateStore(dif, alloc);\n"
       << "          dif = alloc;\n"
       << "        }\n";
  } else {
    os << ") {\n";
  }

  // just make this const one available now to have less variable name repition
  os << "Value * const_one = to_blas_callconv(Builder2, "
        "ConstantInt::get(intType, 1), "
     << "byRef, cublas, intType, allocationBuilder, \"int.one\");\n";

  os << "      auto bb_name = Builder2.GetInsertBlock()->getName();\n";
  for (size_t i = 0; i < activeArgs.size(); i++) {
    auto rule = rules[i];
    const size_t actArg = activeArgs[i];
    const auto ruleDag = rule.getRuleDag();
    const auto name = nameVec[actArg];
    const auto nameMap = rule.getArgNameMap();
    const auto ty = typeMap.lookup(actArg);
    const auto valueTypes = ValueType_helper(pattern, actArg);
    const auto opName = ruleDag->getOperator()->getAsString();
    const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();

    if (Def->isSubClassOf("DiffeRetIndex")) {
      os << "      if (active_" << name << ") {\n"
         << "        Value *toadd = dif;\n"
         << "        addToDiffe(arg_" << name << ", toadd, Builder2, "
         << ", type_" << name << ");\n"
         << "      }\n";
    } else if (Def->isSubClassOf("b")) {
      emit_if_rule_condition(ruleDag, name, "      ", os);
      emit_runtime_condition(ruleDag, name, "        ", "Builder2",
                             (ty == ArgType::fp), os);
      const auto dfnc_name = Def->getValueAsString("s");
      rev_call_args("args1", rule, actArg, os, -1, dfnc_name);
      os << "        const auto Defs = gutils->getInvertedBundles(&call, {"
         << valueTypes << "}, Builder2, /* lookup */ true);\n";

      if (ty == ArgType::fp) {
        // extra handling, since we will update only a fp scalar as part of the
        // return struct it's presumably done by setting it to the value
        // returned by this call
        os << "      if (!cublas) {\n";
        emit_fret_call(dfnc_name, "ArrayRef<Value *>(args1)", name, "Builder2",
                       os);
        os << "      } else {\n";
      } else {
        os << "    SmallVector<Type*, 1> tys; for (auto arg : args1) "
              "tys.push_back(arg->getType());\n";
        std::string dfnc_ret_ty = get_blas_ret_ty(dfnc_name);

        os << "    llvm::FunctionType *FT" << dfnc_name << ";\n";
        os << "    if (cublas) {\n"
           << "      FT" << dfnc_name
           << " = FunctionType::get(cublas_retty, tys, false);\n"
           << "    } else {\n"
           << "      FT" << dfnc_name << " = FunctionType::get(" << dfnc_ret_ty
           << ", tys, false);\n"
           << "    }\n";
        os << "    auto derivcall_" << dfnc_name
           << " = gutils->oldFunc->getParent()->getOrInsertFunction(\n"
           << "  blas.prefix + blas.floatType + \"" << dfnc_name
           << "\" + blas.suffix, FT" << dfnc_name << ");\n";

        os << "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name
           << ".getCallee()))\n"
           << "    {\n"
           << "      attribute_" << dfnc_name << "(blas, F);\n"
           << "    }\n\n";
        os << "        Builder2.CreateCall(derivcall_" << dfnc_name
           << ", args1, Defs);\n";
      }
      if (ty == ArgType::fp)
        os << "      }\n";
      emit_runtime_continue(ruleDag, name, "        ", "Builder2",
                            (ty == ArgType::fp), os);
      os << "      }\n";
    } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "noop") {
    } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "inactive") {
      os << "      assert(!active_" << name << ");\n";
    } else if (Def->isSubClassOf("DiagUpdateSPMV")) {
      os << "      // DiagUpdateSPMV\n";
      emit_if_rule_condition(ruleDag, name, "      ", os);
      emit_runtime_condition(ruleDag, name, "        ", "Builder2", true, os);
      rev_call_args("args1", rule, actArg, os, -1, "");
      os << "        const auto Defs = gutils->getInvertedBundles(&call, {"
         << valueTypes << "}, Builder2, /* lookup */ true);\n";
      // Now that we have the defs, we can create the call
      assert(ty == ArgType::ap);
      os << "callSPMVDiagUpdate(Builder2, *gutils->oldFunc->getParent(), blas, "
            "intType, blasCharType, blasFPType, type_vec_like, type_n, fpType, "
            "ArrayRef<Value *>(args1), "
            "Defs, byRef, julia_decl);\n";
      emit_runtime_continue(ruleDag, name, "        ", "Builder2", true, os);
      os << "      }\n";
    } else if (Def->isSubClassOf("FrobInnerProd")) {
      assert(ty == ArgType::fp);
      os << "      // FrobInnerProd\n";
      emit_if_rule_condition(ruleDag, name, "      ", os);
      emit_runtime_condition(ruleDag, name, "        ", "Builder2", true, os);
      rev_call_args("args1", rule, actArg, os, -1, "");
      os << "        const auto Defs = gutils->getInvertedBundles(&call, {"
         << valueTypes << "}, Builder2, /* lookup */ true);\n";
      // Now that we have the defs, we can create the call
      emit_fret_call("inner_prod", "ArrayRef<Value *>(args1)", name, "Builder2",
                     os);
      emit_runtime_continue(ruleDag, name, "        ", "Builder2", true, os);
      os << "      }\n";
    } else if (Def->isSubClassOf("Seq")) {
      os << "      // Seq\n";
      // (Currently) we only need advanced rules for differentiating
      // wrt. scalar or ap. Make this more generic once we have more testcases.
      assert(ty == ArgType::fp || ty == ArgType::ap);
      emit_if_rule_condition(ruleDag, name, "      ", os);
      emit_runtime_condition(ruleDag, name, "        ", "Builder2", true, os);

      // We might need to create a tmp vec or matrix
      emit_tmp_creation(Def, os, "Builder2");

      os << "        const auto Defs = gutils->getInvertedBundles(&call, {"
         << valueTypes << "}, Builder2, /* lookup */ true);\n";

      // handle seq rules
      for (size_t i = 0; i < ruleDag->getNumArgs(); i++) {
        Init *subArg = ruleDag->getArg(i);
        DagInit *sub_Dag = cast<DagInit>(subArg);
        if (auto sub_def = dyn_cast<DefInit>(sub_Dag->getOperator())) {
          const auto sub_Def = sub_def->getDef();
          if (sub_Def->isSubClassOf("b")) {
            const auto dfnc_name = sub_Def->getValueAsString("s");
            std::string argName = "args" + std::to_string(i);
            rev_call_args(argName, rule, actArg, os, i, dfnc_name);
            os << "    //handling nested blas: " << std::to_string(i) << "\n";
            // emit_deriv_blas_call(sub_Dag, patternMap, handled, os);
            if (get_blas_ret_ty(dfnc_name) == "fpType") {
              // returns, so assume it's the last step of the sequence
              // and update the diffe accordingly
              assert(i == ruleDag->getNumArgs() - 1);
              os << "    if (cublas) assert(false && "
                    "\"cublas not implemented\");\n";
              emit_fret_call(dfnc_name, argName, name, "Builder2", os);
            } else {
              os << "    SmallVector<Type*, 1> tys; for (auto arg : " << argName
                 << ") tys.push_back(arg->getType());\n";
              std::string dfnc_ret_ty = get_blas_ret_ty(dfnc_name);
              os << "    llvm::FunctionType *FT" << dfnc_name
                 << " = FunctionType::get(" << dfnc_ret_ty
                 << ", tys, false);\n";
              os << "    auto derivcall_" << dfnc_name
                 << " = gutils->oldFunc->getParent()->getOrInsertFunction(\n"
                 << "  blas.prefix + blas.floatType + \"" << dfnc_name
                 << "\" + blas.suffix, FT" << dfnc_name << ");\n";

              os << "    if (auto F = dyn_cast<Function>(derivcall_"
                 << dfnc_name << ".getCallee()))\n"
                 << "    {\n"
                 << "      attribute_" << dfnc_name << "(blas, F);\n"
                 << "    }\n\n";
              os << "        Builder2.CreateCall(derivcall_" << dfnc_name
                 << ", " << argName << ", Defs);\n";
            }
            os << "    //handled nested blas: " << std::to_string(i) << "\n";
          } else if (sub_Def->isSubClassOf("FrobInnerProd")) {
            std::string argName = "args" + std::to_string(i);
            rev_call_args(argName, rule, actArg, os, i, "");
            assert(sub_Dag->getNumArgs() == 4);
            assert(ty == ArgType::fp);
            emit_fret_call("inner_prod", argName, name, "Builder2", os);
          } else if (sub_Def->isSubClassOf("DiagUpdateSPMV")) {
            std::string argName = "args" + std::to_string(i);
            rev_call_args(argName, rule, actArg, os, i, "");
            assert(sub_Dag->getNumArgs() == 6);
            assert(ty == ArgType::ap);
            os << "callSPMVDiagUpdate(Builder2, *gutils->oldFunc->getParent(), "
                  "blas, intType, blasCharType, blasFPType, type_vec_like, "
                  "type_n, fpType, "
                  "ArrayRef<Value *>("
               << argName << "), Defs, byRef, julia_decl);\n";
          }
        }
      }
      emit_tmp_free(Def, os, "Builder2");
      emit_runtime_continue(ruleDag, name, "        ", "Builder2", true, os);
      os << "      }\n";
    } else {
      errs() << Def->getName() << "\n";
      PrintFatalError("Unhandled blas-rev case!");
    }
  }
  if (hasDiffeRetVal) {
    os << "    if (cublas)\n";
    os << "      Builder2.CreateStore(Constant::getNullValue(fpType), dif);\n";
  }

  os << "    },\n"
     << "    ";

  first = true;
  for (auto arg : activeArgs) {
    const auto name = nameVec[arg];
    const auto ty = typeMap.lookup(arg);
    // We don't pass in shadows of fp values,
    // we just create and struct-return the shadows
    if (ty == ArgType::fp)
      continue;
    os << ((first) ? "" : ", ") << "d_" + name;
    first = false;
  }
  if (hasDiffeRetVal) {
    os << ((first) ? "" : ", ") << "dif);\n";
    os << "  if (!cublas)\n"
       << "    setDiffe(\n"
       << "      &call,\n"
       << "      "
          "Constant::getNullValue(gutils->getShadowType(call.getType())),\n"
       << "      Builder2);\n";
  } else {
    os << "  );\n";
  }

  os << "  }\n";
}

// Further optimization: re-use / share caches where possible

/*
 * We create the following variables:
 */
void emitBlasDerivatives(const RecordKeeper &RK, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &blasPatterns = RK.getAllDerivedDefinitions("CallBlasPattern");

  SmallVector<TGPattern, 8> newBlasPatterns;
  StringMap<TGPattern> patternMap;
  for (auto &&pattern : blasPatterns) {
    auto parsedPattern = TGPattern(pattern);
    newBlasPatterns.push_back(TGPattern(pattern));
    auto newEntry = std::pair<std::string, TGPattern>(parsedPattern.getName(),
                                                      parsedPattern);
    patternMap.insert(newEntry);
  }

  // Make sure that we only call blass function b for calculating the derivative
  // of a iff we have defined b and pass the right amount of parameters.
  // TODO: type check params, as far as possible
  checkBlasCalls(RK, blasPatterns);
  // //checkBlasCalls2(newBlasPatterns);
  emit_handleBLAS(newBlasPatterns, os);
  // // emitEnumMatcher(blas_modes, os);

  // https://docs.altimesh.com/api/Hybridizer.Runtime.CUDAImports.cublasOperation_t.html
  os << "enum cublasOperation_t {\n"
     << "  CUBLAS_OP_N = 0,\n"
     << "  CUBLAS_OP_T = 1,\n"
     << "  CUBLAS_OP_C = 2,\n"
     << "};\n";

  for (auto &&newPattern : newBlasPatterns) {
    bool hasNonInactive = false;
    for (Rule rule : newPattern.getRules()) {
      const auto ruleDag = rule.getRuleDag();
      const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
      if (Def->isSubClassOf("MagicInst") && Def->getName() == "inactive")
        continue;
      hasNonInactive = true;
      break;
    }
    if (!hasNonInactive)
      continue;

    emit_beginning(newPattern, os);
    emit_helper(newPattern, os);
    emit_scalar_types(newPattern, os);

    emit_caching(newPattern, os);
    emit_extract_calls(newPattern, os);

    emit_fwd_rewrite_rules(newPattern, os);
    emit_rev_rewrite_rules(patternMap, newPattern, os);

    //// writeEnums(pattern, blas_modes, os);
    emit_free_and_ending(newPattern, os);
  }
}

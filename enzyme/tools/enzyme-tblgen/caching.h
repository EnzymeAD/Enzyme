//===- enzyme-tblgen.cpp - Top-Level TableGen implementation for Enzyme
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
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

// clang-format off
//

using namespace llvm;
void emit_vinc_caching(Record *pattern, std::vector<size_t> actArgs, 
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {

  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  size_t argPosition = 0;
  std::vector<std::string> cacheVars{};

  // Debug
  // for (size_t i = 0; i < 6; i++) {
  //   os << "arg " << i << " is used by: ";
  //   llvm::SmallSet<size_t, 5> x = argUsers.lookup(i);
  //   for (auto val : x) 
  //     os << val << " ";
  //   os << "\n";
  // }

  for (auto val : inputTypes) {
    if (val->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto vecPosition = argPosition;
      auto vecUsers = argUsers.lookup(vecPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      auto incPosition = argPosition + 1;
      auto incUsers = argUsers.lookup(incPosition);
      os 
<< "  bool cache_" << vecName
<< "  = Mode != DerivativeMode::ForwardMode &&\n"
<< "          uncacheable_" << vecName;
      for (size_t user: vecUsers) {
        auto name = argOps->getArgNameStr(user);
        os 
<< " && active_" << name;
      }
      os 
<< ";\n"
<< "  bool cache_" << incName << " = false;\n";
      cacheVars.push_back("cache_" + vecName.str());
      // xinc is needed to be preserved if
      // 1) it is potentially overwritten AND EITHER
      //     a) x is active (for performing the shadow increment) or
      //     b) we're not caching x and need xinc to compute the
      //     derivative of a different variable
      os 
<< "  bool need_" << incName << " = (active_" << vecName;
      if (incUsers.size() > 0) {
        os 
<< "  || (!cache_" << vecName << " && (";
        bool first = true;
        for (size_t user: incUsers) {
          auto name = argOps->getArgNameStr(user);
          if (!first)
            os << " || ";
          os 
<< "active_" << name;
          first = false;
        }
        os 
<< ")";
      }
      os 
<< ");\n"
<< "  if (byRef && uncacheable_" << incName << " && need_" << incName << ") {\n"
<< "    cacheTypes.push_back(intType);\n"
<< "    cache_" << incName << " = true;\n "
<< "  }\n\n";
    } // end vinc
    argPosition += val->getValueAsInt("nelem");
  }
  os 
<< "  int numCached = ";
  for (size_t i = 0; i < cacheVars.size(); i++) {
    if (i > 0)
      os << " + ";
    os << "(int) " << cacheVars[i];
  }
  os 
<< ";\n"
<< "  bool anyCache = (numCached > 0);\n";
}

void emit_count_caching(Record *pattern, std::vector<size_t> actArgs,
                        raw_ostream &os) {
  os 
<< "  // count must be preserved if overwritten\n";
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  size_t argPosition = 0;
  for (auto val : inputTypes) {
    if (val->getName() == "len") {
      auto name = argOps->getArgNameStr(argPosition);
      os 
<< "  bool cache_" << name << " = false;\n"
<< "  if (byRef && uncacheable_" << name << ") {\n"
<< "    cacheTypes.push_back(intType);\n"
<< "    cache_" << name << " = true;\n"
<< "  }\n";
    }
    argPosition += val->getValueAsInt("nelem");
  }
}

void emit_cache_for_reverse(Record *pattern, std::vector<size_t> actArgs, 
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {
  os 
<< "  if ((Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "       Mode == DerivativeMode::ReverseModePrimal) && cachetype) {\n"
<< "    SmallVector<Value *, 2> cacheValues;\n";
  
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");

  size_t argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "len") {
      auto name = argOps->getArgNameStr(argPosition);
      os 
<< "    Value *count = gutils->getNewFromOriginal(arg_" << name << ");\n";
      break;
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  argPosition = 0;
  for (auto inputType : inputTypes) {

    // cache count if needed
    if (inputType->getName() == "len") {
      os
<< "    if (byRef) {\n"
<< "      count = BuilderZ.CreatePointerCast(count, PointerType::getUnqual(intType));\n"
<< "#if LLVM_VERSION_MAJOR > 7\n"
<< "      count = BuilderZ.CreateLoad(intType, count);\n"
<< "#else\n"
<< "      count = BuilderZ.CreateLoad(count);\n"
<< "#endif\n"
<< "      if (countcache)\n"
<< "        cacheValues.push_back(count);\n"
<< "    }\n";
    } else if (inputType->getName() == "vinc") {
      //auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      //auto incPosition = argPosition + 1;

      // cache vinc's if needed.
      os 
<< "    Value *" << incName << " = gutils->getNewFromOriginal(arg_" << incName <<");\n"
<< "    if (byRef) {\n"
<< "      " << incName << " = BuilderZ.CreatePointerCast(" << incName << ", PointerType::getUnqual(intType));\n"
<< "#if LLVM_VERSION_MAJOR > 7\n"
<< "      " << incName << " = BuilderZ.CreateLoad(intType, " << incName << ");\n"
<< "#else\n"
<< "      " << incName << " = BuilderZ.CreateLoad(" << incName << ");\n"
<< "#endif\n"
<< "      if (cache_" << incName << ")\n"
<< "        cacheValues.push_back(" << incName << ");\n"
<< "    }\n";
    } else {
      // handle fp
      continue;
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  size_t i = 0;
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      os
<< "    if (cache_" << vecName << ") {\n"
<< "      auto dmemcpy = getOrInsertMemcpyStrided(\n"
<< "          *gutils->oldFunc->getParent(), cast<PointerType>(castvals[" << i << "]),\n"
<< "          count->getType(), 0, 0);\n"
<< "      auto malins = CreateAllocation(BuilderZ, innerType, count);\n"
<< "      Value *arg = BuilderZ.CreateBitCast(malins, castvals[" << i << "]);\n"
<< "      Value *args[4] = {arg,\n"
<< "                         gutils->getNewFromOriginal(arg_" << vecName << "),\n"
<< "                         count, " << incName << "};\n"
<< "      if (args[1]->getType()->isIntegerTy())\n"
<< "        args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[" << i << "]);\n"
<< "      BuilderZ.CreateCall(dmemcpy, args,\n"
<< "          gutils->getInvertedBundles(&call,{\n";
      bool first = true;
      for (size_t j = 0; j < argOps->arg_size(); j++) {
        if (!first)
          os << ", ";
        os 
<< ((j == argPosition) ? "ValueType::Shadow" : "ValueType::None");
        first = false;
      }
      //<< "                                          ValueType::None, ValueType::Shadow,\n"
      //<< "                                           ValueType::None, ValueType::None,\n"
      //<< "                                           ValueType::None\n"
      os
<< "    },BuilderZ, /*lookup*/ false));\n"
<< "      cacheValues.push_back(arg);\n"
<< "    }\n";
      i++;
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  os
<< "    if (cacheValues.size() == 1) {\n"
<< "      cacheval = cacheValues[0];\n"
<< "    } else {\n"
<< "      cacheval = UndefValue::get(cachetype);\n"
<< "      for (auto tup : llvm::enumerate(cacheValues))\n"
<< "        cacheval = BuilderZ.CreateInsertValue(cacheval, tup.value(), tup.index());\n"
<< "    }\n"
<< "    gutils->cacheForReverse(BuilderZ, cacheval,\n"
<< "                            getIndex(&call, CacheType::Tape));\n"
<< "  }\n"
<< "  unsigned cacheidx = 0;\n"
<< "  Value *count = gutils->getNewFromOriginal(call.getArgOperand(0));\n"; // todo adjust idx
 
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto incName = argOps->getArgNameStr(argPosition + 1);
      os 
<< "  Value *true_" << incName << " = gutils->getNewFromOriginal(arg_" << incName << ");\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  os
<< "  IRBuilder<> Builder2(call.getParent());\n"
<< "  switch (Mode) {\n"
<< "    case DerivativeMode::ReverseModeCombined:\n"
<< "    case DerivativeMode::ReverseModeGradient:\n"
<< "      getReverseBuilder(Builder2);\n"
<< "      break;\n"
<< "    case DerivativeMode::ForwardMode:\n"
<< "    case DerivativeMode::ForwardModeSplit:\n"
<< "      Builder2.SetInsertPoint(BuilderZ.GetInsertBlock(),\n"
<< "                              BuilderZ.GetInsertPoint());\n"
<< "      Builder2.setFastMathFlags(BuilderZ.getFastMathFlags());\n"
<< "      break;\n"
<< "    case DerivativeMode::ReverseModePrimal:\n"
<< "      break;\n"
<< "  }\n\n";
}

void emit_caching(Record *pattern, std::vector<size_t> actArgs,
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {

  // 1. No caching for fwd-mode
  // 2. Deactivate caching for uncacheable_args
  // 3. Only caching if we do need the primary for an active gradient.
  os 
<< "  SmallVector<Type *, 2> cacheTypes;\n\n";

  emit_count_caching(pattern, actArgs, os);
  // emit_fp_caching(pattern, actArgs, os);
  emit_vinc_caching(pattern, actArgs, argUsers, os);

  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  for (auto actEn : llvm::enumerate(actArgs)) {
    auto name = argOps->getArgNameStr(actEn.value());
    os 
<< "  if (" << name << "cache)\n"
<< "    cacheTypes.push_back(castvals[" << actEn.index() << "]);\n";
  }
  os
<< "  Type *cachetype = nullptr;\n"
<< "  switch (cacheTypes.size()) {\n"
<< "  case 0:\n"
<< "    break;\n"
<< "  case 1:\n"
<< "    cachetype = cacheTypes[0];\n"
<< "    break;\n"
<< "  default:\n"
<< "    cachetype = StructType::get(call.getContext(), cacheTypes);\n"
<< "    break;\n"
<< "  }\n\n";

  emit_cache_for_reverse(pattern, actArgs, argUsers, os);
}


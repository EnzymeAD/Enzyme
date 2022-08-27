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
          os 
<< ((first) ? "" : " || ") << "active_" << name;
          first = false;
        }
        os 
<< "))";
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
    os << ((i > 0) ? " + " : "" ) << "(int) " << cacheVars[i];
  }
  os 
<< ";\n";
}

void emit_scalar_caching(Record *pattern, std::vector<size_t> actArgs,
                        raw_ostream &os) {
  os 
<< "  // len, fp must be preserved if overwritten\n";
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  size_t argPosition = 0;
  for (auto val : inputTypes) {
    auto typeName = val->getName();
    if (typeName == "len" || typeName == "fp") {
      auto scalarType = (typeName == "len") ? "intType" : "fpType";
      auto name = argOps->getArgNameStr(argPosition);
      os 
<< "  bool cache_" << name << " = false;\n"
<< "  if (byRef && uncacheable_" << name << ") {\n"
<< "    cacheTypes.push_back(" << scalarType << ");\n"
<< "    cache_" << name << " = true;\n"
<< "  }\n";
    }
    argPosition += val->getValueAsInt("nelem");
  }
}

void emit_cache_for_reverse(Record *pattern, std::vector<size_t> actArgs, 
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {
  llvm::errs() << "AAAA\n";
  os 
<< "  if ((Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "       Mode == DerivativeMode::ReverseModePrimal) && cachetype) {\n"
<< "    SmallVector<Value *, 2> cacheValues;\n";
  llvm::errs() << "AAAA\n";
  
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");

  size_t argPosition = 0;
  for (auto inputType : inputTypes) {
    auto typeName = inputType->getName();

    if (typeName == "len") {
      auto name = argOps->getArgNameStr(argPosition);
      auto lenName = "len_" + name;
      os
<< "    Value *" << lenName << " = gutils->getNewFromOriginal(arg_" << name <<");\n"
<< "    if (byRef) {\n"
<< "      " << lenName << " = BuilderZ.CreatePointerCast(" << lenName <<", PointerType::getUnqual(intType));\n"
<< "#if LLVM_VERSION_MAJOR > 7\n"
<< "      " << lenName << " = BuilderZ.CreateLoad(intType, " << lenName << ");\n"
<< "#else\n"
<< "      " << lenName << " = BuilderZ.CreateLoad(" << lenName << ");\n"
<< "#endif\n"
<< "      if (cache_" << name << ")\n"
<< "        cacheValues.push_back(" << lenName << ");\n"
<< "    }\n";
    } else if (typeName == "vinc") {
      auto incName = argOps->getArgNameStr(argPosition + 1);
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
    } else if (typeName == "fp") {
      auto name = argOps->getArgNameStr(argPosition);
      auto fpName = "fp_" + name;
      os 
<< "    Value *" << fpName << " = gutils->getNewFromOriginal(arg_" << name <<");\n"
<< "    if (byRef) {\n"
<< "      " << fpName << " = BuilderZ.CreatePointerCast(" << fpName <<", PointerType::getUnqual(fpType));\n"
<< "#if LLVM_VERSION_MAJOR > 7\n"
<< "      " << fpName << " = BuilderZ.CreateLoad(fpType, " << fpName << ");\n"
<< "#else\n"
<< "      " << fpName << " = BuilderZ.CreateLoad(" << fpName << ");\n"
<< "#endif\n"
<< "      if (cache_" << name << ")\n"
<< "        cacheValues.push_back(" << fpName << ");\n"
<< "    }\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  llvm::errs() << "AAAA\n";

  size_t i = 0;
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      // TODO: remove last hardcoded len_n, type_n usages to support blas lv2/3 
      os
<< "    if (cache_" << vecName << ") {\n"
<< "      auto dmemcpy = getOrInsertMemcpyStrided(\n"
<< "          *gutils->oldFunc->getParent(), cast<PointerType>(castvals[" << i << "]),\n"
<< "          type_n, 0, 0);\n"
<< "      auto malins = CreateAllocation(BuilderZ, fpType, len_n);\n"
<< "      Value *arg = BuilderZ.CreateBitCast(malins, castvals[" << i << "]);\n"
<< "      Value *args[4] = {arg,\n"
<< "                         gutils->getNewFromOriginal(arg_" << vecName << "),\n"
<< "                         len_n, " << incName << "};\n"
<< "      if (args[1]->getType()->isIntegerTy())\n"
<< "        args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[" << i << "]);\n"
<< "      BuilderZ.CreateCall(dmemcpy, args,\n"
<< "          gutils->getInvertedBundles(&call,{\n";
      bool first = true;
      for (size_t j = 0; j < argOps->arg_size(); j++) {
        os 
<< ((first) ? "" : ", ")
<< ((j == argPosition) ? "ValueType::Shadow" : "ValueType::None");
        first = false;
      }
      os
<< "},\n"
<< "          BuilderZ, /*lookup*/ false));\n"
<< "      cacheValues.push_back(arg);\n"
<< "    }\n";
      i++;
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  llvm::errs() << "AAAA\n";

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
<< "  unsigned cacheidx = 0;\n";
 
  argPosition = 0;
  for (auto inputType : inputTypes) {
    auto typeName = inputType->getName();
    if (typeName == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      os 
<< "  Value *true_" << incName << " = gutils->getNewFromOriginal(arg_" << incName << ");\n"
<< "  Value *" << incName << " = true_" << incName << ";\n"
<< "  Value *data_" << vecName << " = gutils->getNewFromOriginal(arg_" << vecName << ");\n"
<< "  Value *data_ptr_" << vecName << " = nullptr;\n";
    } else if (typeName == "len") {
      auto lenName = argOps->getArgNameStr(argPosition);
      os
<< "  Value *len_" << lenName << " = gutils->getNewFromOriginal(arg_" << lenName << ");\n";
    } else if (typeName == "fp") {
      auto fpName = argOps->getArgNameStr(argPosition);
      os
<< "  Value *fp_" << fpName << " = gutils->getNewFromOriginal(arg_" << fpName << ");\n"; 
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  llvm::errs() << "AAAA\n";

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

  emit_scalar_caching(pattern, actArgs, os);
  emit_vinc_caching(pattern, actArgs, argUsers, os);

  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  for (auto actEn : llvm::enumerate(actArgs)) {
    auto name = argOps->getArgNameStr(actEn.value());
    os 
<< "  if (cache_" << name << ")\n"
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


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

// clang-format off
//

using namespace llvm;

enum ActionType { GenDerivatives, GenBlasDerivatives };

static cl::opt<ActionType>
    action(cl::desc("Action to perform:"),
           cl::values(clEnumValN(GenBlasDerivatives, "gen-blas-derivatives",
                                 "Generate BLAS derivatives")),
           cl::values(clEnumValN(GenDerivatives, "gen-derivatives",
                                 "Generate instruction derivative")));

bool hasDiffeRet(Init *resultTree) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "DiffeRet" || Def->isSubClassOf("DiffeRet")) {
      return true;
    }
    for (auto zp :
         llvm::zip(resultRoot->getArgs(), resultRoot->getArgNames())) {
      if (hasDiffeRet(std::get<0>(zp)))
        return true;
    }
  }
  return false;
}

void getFunction(raw_ostream &os, std::string callval, std::string FT,
                 std::string cconv, Init *func) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(func)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "SameFunc" || Def->isSubClassOf("SameFunc")) {
      os << "#if LLVM_VERSION_MAJOR >= 11\n";
      os << "  auto " << callval << " = orig->getCalledOperand();\n";
      os << "#else\n";
      os << "  auto " << callval << " = orig->getCalledValue();\n";
      os << "#endif\n";
      os << "  auto " << FT << " = orig->getFunctionType();\n";
      os << "  auto " << cconv << " = orig->getCallingConv();\n";
      return;
    }
    if (opName == "SameTypesFunc" || Def->isSubClassOf("SameTypesFunc")) {
      os << "  auto " << FT << " = orig->getFunctionType();\n";
      os << "  auto " << callval
         << " = gutils->oldFunc->getParent()->getOrInsertFunction(";
      os << Def->getValueInit("name")->getAsString();
      os << ", " << FT << ", called->getAttributes())\n";
      os << "#if LLVM_VERSION_MAJOR >= 9\n";
      os << "  .getCallee()\n";
      os << "#endif\n";
      os << ";\n";
      os << "  auto " << cconv << " = orig->getCallingConv();\n";
      return;
    }
  }
  assert(0 && "Unhandled function");
}

// Returns whether value generated is a vector value or not.
bool handle(raw_ostream &os, Record *pattern, Init *resultTree,
            std::string builder, StringMap<std::string> &nameToOrdinal,
            bool lookup) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "DiffeRet" || Def->isSubClassOf("DiffeRet")) {
      os << "dif";
      return true;
    } else if (opName == "ConstantFP" || Def->isSubClassOf("ConstantFP")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op constant supported");

      auto *argument = resultRoot->getArg(0);

      auto value = dyn_cast<StringInit>(Def->getValueInit("value"));
      if (!value)
        PrintFatalError(pattern->getLoc(), Twine("'value' not defined in ") +
                                               resultTree->getAsString());

      os << "ConstantFP::get(";
      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto ord = nameToOrdinal.find(name);
        if (ord == nameToOrdinal.end())
          PrintFatalError(pattern->getLoc(), Twine("unknown named operand '") +
                                                 name + "'" +
                                                 resultTree->getAsString());
        os << ord->getValue();
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in constantfp") +
                            resultTree->getAsString());
      os << "->getType(), \"" << value->getValue() << "\")";
      return false;
    } else if (opName == "Shadow" || Def->isSubClassOf("Shadow")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op constant supported");

      auto *argument = resultRoot->getArg(0);
      if (lookup)
        os << "lookup(";
      os << "gutils->invertPointerM(" << builder << ", ";

      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto ord = nameToOrdinal.find(name);
        if (ord == nameToOrdinal.end())
          PrintFatalError(pattern->getLoc(), Twine("unknown named operand '") +
                                                 name + "'" +
                                                 resultTree->getAsString());
        os << ord->getValue();

      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in shadow") +
                            resultTree->getAsString());
      os << ")";
      if (lookup)
        os << ", " << builder << ")";
      return true;
    }

    os << " ({\n";
    os << "    Value* args[" << resultRoot->getArgs().size() << "];\n";

    SmallVector<bool, 1> vectorValued;
    bool anyVector = false;

    size_t idx = 0;
    StringMap<std::string> oldMaps;
    for (auto zp :
         llvm::zip(resultRoot->getArgs(), resultRoot->getArgNames())) {
      os << " args[" << idx << "] = ";
      idx++;
      if (isa<UnsetInit>(std::get<0>(zp)) && std::get<1>(zp)) {
        auto name = std::get<1>(zp)->getAsUnquotedString();
        auto ord = nameToOrdinal.find(name);
        if (ord == nameToOrdinal.end())
          PrintFatalError(pattern->getLoc(), Twine("unknown named operand '") +
                                                 name + "'" +
                                                 resultTree->getAsString());
        if (!StringRef(ord->getValue()).startswith("__tmp_")) {
          if (lookup)
            os << "lookup(";
          os << "gutils->getNewFromOriginal(";
        }
        os << ord->getValue();
        if (!StringRef(ord->getValue()).startswith("__tmp_")) {
          os << ")";
          if (lookup)
            os << ", " << builder << ")";
        }
        os << " ;\n";
        vectorValued.push_back(false);
        continue;
      }
      vectorValued.push_back(
          handle(os, pattern, std::get<0>(zp), builder, nameToOrdinal, lookup));
      os << " ;\n";
      if (std::get<1>(zp)) {
        auto name = std::get<1>(zp)->getAsUnquotedString();
        oldMaps.try_emplace(name, nameToOrdinal[name]);
        nameToOrdinal[name] = "__tmp_" + name;
        os << " Value* __tmp_" << name << " = args[" << (idx - 1) << "];\n";
      }

      anyVector |= vectorValued.back();
    }
    for (auto &pair : oldMaps) {
      if (pair.second.size())
        nameToOrdinal[pair.getKey()] = pair.second;
      else
        nameToOrdinal.erase(pair.getKey());
    }

    if (opName == "Call" || Def->isSubClassOf("Call")) {
      getFunction(os, "callval", "FT", "cconv", Def->getValueInit("func"));
    }

    os << " Value *res = nullptr;\n";

    if (anyVector) {
      os << " if (gutils->getWidth() == 1) { \n";
    }

    if (opName == "Call" || Def->isSubClassOf("Call")) {
      os << " CallInst *cubcall = cast<CallInst>(" << builder
         << ".CreateCall(FT, callval, ArrayRef<Value*>({";
    } else {
      os << "   res = " << builder << ".Create" << opName << "(";
    }
    for (size_t i = 0; i < idx; i++) {
      if (i > 0)
        os << ", ";
      os << "args[" << i << "]";
    }
    if (opName == "Call" || Def->isSubClassOf("Call")) {
      os << "})";
      os << ")";
    }
    os << ")";
    os << ";\n";
    if (opName == "Call" || Def->isSubClassOf("Call")) {
      os << " cubcall->setDebugLoc(gutils->getNewFromOriginal(orig->"
            "getDebugLoc()));\n";
      os << " cubcall->setCallingConv(cconv);\n";
      for (auto *attr : *cast<ListInit>(Def->getValueAsListInit("fnattrs"))) {
        auto attrDef = cast<DefInit>(attr)->getDef();
        os << "#if LLVM_VERSION_MAJOR >= 14\n"
           << " cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, "
           << "Attribute::"
           << attrDef->getValueInit("name")->getAsUnquotedString() << ");\n";
        os << "#else\n"
           << " cubcall->addAttribute(AttributeList::FunctionIndex, "
           << "Attribute::"
           << attrDef->getValueInit("name")->getAsUnquotedString() << ");\n";
        os << "#endif\n";
      }
      os << " res = cubcall;\n";
    }
    if (anyVector) {
      os << " } else {\n";
      os << " for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";

      if (opName == "Call" || Def->isSubClassOf("Call")) {
        os << " CallInst *V = cast<CallInst>(" << builder
           << ".CreateCall(FT, callval, ArrayRef<Value*>({";
      } else {
        os << "   Value *V = " << builder << ".Create" << opName << "(";
      }
      for (size_t i = 0; i < idx; i++) {
        if (i > 0)
          os << ", ";
        if (vectorValued[i])
          os << builder << ".CreateExtractValue(args[" << i << "], {idx})";
        else
          os << "args[" << i << "]";
      }
      if (opName == "Call" || Def->isSubClassOf("Call"))
        os << "})";
      os << ")";
      if (opName == "Call" || Def->isSubClassOf("Call")) {
        os << ")";
      }
      os << ";\n";

      if (opName == "Call" || Def->isSubClassOf("Call")) {
        os << "   "
              "V->setDebugLoc(gutils->getNewFromOriginal(orig->getDebugLoc()));"
              "\n";
        os << "   V->setCallingConv(cconv);\n";
        for (auto *attr : *cast<ListInit>(Def->getValueAsListInit("fnattrs"))) {
          auto attrDef = cast<DefInit>(attr)->getDef();
          os << "#if LLVM_VERSION_MAJOR >= 14\n"
             << "   V->addAttributeAtIndex(AttributeList::FunctionIndex, "
                "Attribute::"
             << attrDef->getValueInit("name")->getAsUnquotedString() << ");\n";
          os << "#else \n"
             << "   V->addAttribute(AttributeList::FunctionIndex, "
                "Attribute::"
             << attrDef->getValueInit("name")->getAsUnquotedString() << ");\n";
          os << "#endif \n";
        }
      }
      os << "   if (res == nullptr) res = "
            "UndefValue::get(ArrayType::get(V->getType(), "
            "gutils->getWidth()));\n";
      os << "   res = " << builder << ".CreateInsertValue(res, V, {idx});\n";
      os << " }\n }\n";
    }
    os << " res; })";
    return anyVector;
  }

  PrintFatalError(pattern->getLoc(), Twine("unknown dag"));
}

void emitFullDerivatives(const RecordKeeper &RK,
                         raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = RK.getAllDerivedDefinitions("CallPattern");
  // Ensure unique patterns simply by appending unique suffix.
  unsigned rewritePatternCount = 0;
  std::string baseRewriteName = "GeneratedConvert";
  for (Record *pattern : patterns) {
    DagInit *tree = pattern->getValueAsDag("PatternToMatch");

    StringMap<std::string> nameToOrdinal;
    for (int i = 0, e = tree->getNumArgs(); i != e; ++i)
      nameToOrdinal[tree->getArgNameStr(i)] =
          "orig->getOperand(" + std::to_string(i) + ")";

    if (tree->getNameStr().str().size())
      nameToOrdinal[tree->getNameStr().str()] = "orig";

    for (auto arg : tree->getArgs()) {
      if (isa<DagInit>(arg))
        PrintFatalError(pattern->getLoc(),
                        "only single pattern inputs supported");
    }

    // Emit RewritePattern for Pattern.
    ListInit *argOps = pattern->getValueAsListInit("ArgDerivatives");

    os << "  if (";

    bool prev = false;
    for (auto *nameI : *cast<ListInit>(pattern->getValueAsListInit("names"))) {
      if (prev)
        os << " ||\n      ";
      os << "funcName == " << cast<StringInit>(nameI)->getAsString() << "";
      prev = true;
    }
    os << " ){\n";
    os << "    if (gutils->knownRecomputeHeuristic.find(orig) !=\n";
    os << "        gutils->knownRecomputeHeuristic.end()) {\n";
    os << "        if (!gutils->knownRecomputeHeuristic[orig]) {\n";
    os << "          gutils->cacheForReverse(BuilderZ, newCall,\n";
    os << "                                  getIndex(orig, "
          "CacheType::Self));\n";
    os << "        }\n";
    os << "    }\n";

    os << "    eraseIfUnused(*orig);\n";
    os << "    if (gutils->isConstantInstruction(orig))\n";
    os << "      return;\n";

    os << "    switch (Mode) {\n";
    os << "      case DerivativeMode::ForwardModeSplit:\n";
    os << "      case DerivativeMode::ForwardMode:{\n";
    os << "        IRBuilder<> Builder2(&call);\n";
    os << "        getForwardBuilder(Builder2);\n";
    // TODO

    os << "        Value *res = nullptr;\n";

    for (auto argOpEn : llvm::enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();
      os << "        if (!gutils->isConstantValue(orig->getArgOperand("
         << argIdx << "))) {\n";
      os << "          Value *dif = diffe(orig->getArgOperand(" << argIdx
         << "), Builder2);\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      os << "          Value *tmp = ";

      bool vectorValued = handle(os, pattern, resultTree, "Builder2",
                                 nameToOrdinal, /*lookup*/ false);
      os << ";\n";

      os << "          if (res == nullptr) res = tmp;\n";
      os << "          else if (gutils->getWidth() == 1) res = "
            "Builder2.CreateFAdd(res, tmp);\n";
      os << "          else {\n";
      if (vectorValued)
        os << "            Value *out = UndefValue::get(res->getType());\n";
      else
        os << "            Value *out = "
              "UndefValue::get(gutils->getShadowType(res->getType()));\n";

      os << "            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; "
            "idx++) {\n";
      os << "              Value *V = "
            "Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), ";
      if (vectorValued)
        os << "Builder2.CreateExtractValue(tmp, {idx})";
      else
        os << "tmp";
      os << ");\n";
      os << "              out = Builder2.CreateInsertValue(out, V, {idx});\n";
      os << "            }\n";
      os << "            res = out;\n";
      os << "          }\n";
      os << "        }\n";
    }

    os << "        setDiffe(orig, res, Builder2);\n";

    os << "        break;\n";
    os << "      }\n";

    os << "      case DerivativeMode::ReverseModeGradient:\n";
    os << "      case DerivativeMode::ReverseModeCombined:{\n";
    os << "        IRBuilder<> Builder2(&call);\n";
    os << "        getReverseBuilder(Builder2);\n";
    // TODO vector

    os << "        Value *dif = nullptr;\n";
    bool seen = false;
    for (auto argOpEn : llvm::enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();
      os << "        ";
      if (seen)
        os << "} else ";
      seen = true;
      os << "if (!dif && !gutils->isConstantValue(orig->getArgOperand("
         << argIdx << "))) {\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      if (hasDiffeRet(resultTree)) {
        os << "          dif = diffe(orig, Builder2);\n";
        os << "          setDiffe(orig, "
              "Constant::getNullValue(gutils->getShadowType(orig->getType())), "
              "Builder2);\n";
      }
    }
    if (seen)
      os << "        }\n";

    for (auto argOpEn : llvm::enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();
      DagInit *resultTree = cast<DagInit>(argOpEn.value());

      os << "        if (!gutils->isConstantValue(orig->getArgOperand("
         << argIdx << "))) {\n";
      os << "          Value *tmp = ";
      bool vectorValued = handle(os, pattern, resultTree, "Builder2",
                                 nameToOrdinal, /*lookup*/ true);
      os << ";\n";
      os << "          Value *toadd = tmp;\n";

      if (!vectorValued) {
        os << "          if (gutils->getWidth() > 1) {\n";
        os << "            toadd = "
              "UndefValue::get(gutils->getShadowType(tmp->getType()));\n";
        os << "            for(unsigned int idx=0, W=gutils->getWidth(); "
              "idx<W; idx++) {\n";
        os << "              toadd = Builder2.CreateInsertValue(toadd, tmp, "
              "{idx});\n";
        os << "            }\n";
        os << "          }\n";
      }

      os << "          addToDiffe(orig->getArgOperand(" << argIdx << "), toadd";
      os << ", Builder2, orig->getArgOperand(" << argIdx << ")->getType());\n";
      os << "        }\n";
    }

    os << "        break;\n";
    os << "      }\n";

    os << "      case DerivativeMode::ReverseModePrimal:{\n";
    // TODO
    os << "        break;\n";
    os << "      }\n";
    os << "    }\n";

    os << "    return;\n  }\n";
  }
}


/////////// blas from here on


void emitEnumMatcher(const std::vector<Record *> &blas_modes, raw_ostream &os) {
  for (auto mode : blas_modes) {
    auto name = mode->getName();
    auto sub_modes = mode->getValueAsListOfStrings("modes");
    os 
<< "std::string read_" << name
<< "(llvm::CallInst &call, size_t pos) {\n"
<< "  std::string s = call.getArgOperand(pos)->getValue();\n";
    for (auto sub_mode : sub_modes) {
      os 
<< "  if (s == \"" << sub_mode << "\")\n"
<< "    return \"" << sub_mode << "\";\n";
    }
    os 
<< "  assert(false && \"failed reading " << name << "\");\n"
<< "}\n\n";
  }
}

void writeEnums(Record *pattern, const std::vector<Record *> &blas_modes,
                raw_ostream &os) {
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  for (auto inputType : inputTypes) {
    if (inputType->isSubClassOf("blas_modes")) {
      os << inputType->getName() << ": ";
      for (auto a : inputType->getValueAsListOfStrings("modes")) {
        os << a << " ";
      }
      os << "\n";
    }
  }
  DagInit *tree = pattern->getValueAsDag("PatternToMatch");
  for (int i = 0, e = tree->getNumArgs(); i != e; ++i) {
    // os << tree->getArgNameStr(i) << " ";
    //  auto optns = blas_arg->getValueAsListOfStrings("modes");
    //  for (auto optn : optns)
    //    os << optn << " ";
    //  }
  }
}

void emit_castvals(Record *pattern, std::vector<size_t> activeArgs,
                   raw_ostream &os) {
  os 
<< "  /* beginning castvalls */\n"
<< "  Type *castvals[" << activeArgs.size() << "];\n";
  
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");

  for (auto argPos : llvm::enumerate(activeArgs)) {
    auto name = argOps->getArgNameStr(argPos.value());
    size_t argIdx = argPos.index();
    os 
<< "  if (auto PT = dyn_cast<PointerType>(type_" << name << "))\n"
<< "    castvals[" << argIdx << "] = PT;\n"
<< "  else\n"
<< "    castvals[" << argIdx << "] = PointerType::getUnqual(fpType);\n";
  }
  os 
<< "  Value *cacheval;\n\n"
<< "  /* ending castvalls */\n";
}

void emit_scalar_types(Record *pattern, raw_ostream &os) {
  // We only look at the type of the first integer showing up.
  // This allows to learn if we use Fortran abi (byRef) or cabi
  size_t pos = 0;
  bool foundInt = false;
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");

  for (auto val : inputTypes) {
    if (val->getName() == "len") {
      foundInt = true;
      break;
    }
    pos += val->getValueAsInt("nelem");
  }
  auto name = argOps->getArgNameStr(pos);
  assert(foundInt && "no int type found in blas call");

  os
//<< "  Type *fpType = fpType;\n" // already given by blas type (s, d, c, z)
<< "  IntegerType *intType = dyn_cast<IntegerType>(type_" << name << ");\n"
<< "  bool byRef = false;\n" // Fortran Abi?
<< "  if (!intType) {\n"
<< "    auto PT = cast<PointerType>(type_" << name << ");\n"
<< "    if (blas.suffix.contains(\"64\"))\n"
<< "      intType = IntegerType::get(PT->getContext(), 64);\n"
<< "    else\n"
<< "      intType = IntegerType::get(PT->getContext(), 32);\n"
<< "    byRef = true;\n"
<< "  }\n\n";
}


void emit_beginning(Record *pattern, raw_ostream &os) {
  auto name = pattern->getName();
  os
<< "\nbool handle_" << name
<< "(BlasInfo blas, llvm::CallInst &call, Function *called,\n"
<< "    const std::map<Argument *, bool> &uncacheable_args, Type *fpType) {\n"
<< "  \n"
<< "  CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));\n"
<< "  IRBuilder<> BuilderZ(newCall);\n"
<< "  BuilderZ.setFastMathFlags(getFast());\n"
<< "  IRBuilder<> allocationBuilder(gutils->inversionAllocs);\n"
<< "  allocationBuilder.setFastMathFlags(getFast());\n"
<< "  auto &DL = gutils->oldFunc->getParent()->getDataLayout();\n";
}

std::vector<size_t> getPossiblyActiveArgs(Record *pattern) {
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  int numTypes = 0;
  std::vector<size_t> activeArgs;
  for (auto val : inputTypes) {
    if (val->getValueAsBit("active"))
      activeArgs.push_back(numTypes);
    numTypes += val->getValueAsInt("nelem");
  }

  // verify correctness of declarations in td file
  auto name = pattern->getName();
  DagInit *tree = pattern->getValueAsDag("PatternToMatch");
  int lenDagArgs = tree->getNumArgs();
  assert(numTypes == lenDagArgs);
  return activeArgs;
}

// only for testing
#include "llvm/IR/Type.h"


void emit_free_and_ending(Record *pattern, std::vector<size_t> actArgs,
                  raw_ostream &os) {
    os
<< "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "      Mode == DerivativeMode::ReverseModeGradient ||\n"
<< "      Mode == DerivativeMode::ForwardModeSplit) {\n"
<< "    if (shouldFree()) {\n";
  size_t argPosition = 0;
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto name = argOps->getArgNameStr(argPosition);
      os
<< "      if (cache_" << name << ") {\n"
<< "        CreateDealloc(Builder2, data_ptr_" << name << ");\n"
<< "      }\n";
    }
    argPosition += inputType->getValueAsInt("nelem");

  }
  os
<< "    }\n"
<< "  }\n"
<< "  if (gutils->knownRecomputeHeuristic.find(&call) !=\n"
<< "    gutils->knownRecomputeHeuristic.end()) {\n"
<< "    if (!gutils->knownRecomputeHeuristic[&call]) {\n"
<< "    gutils->cacheForReverse(BuilderZ, newCall,\n"
<< "     getIndex(&call, CacheType::Self));\n"
<< "    }\n"
<< "  }\n"
<< "  return true;\n"
<< "}\n\n";
}

void emit_extract_calls(Record *pattern, std::vector<size_t> actArgs, 
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {
  size_t argPosition = 0;
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
    // TODO: adjust count / getArgOperand(0) based on first int?
  os 
<< "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "      Mode == DerivativeMode::ReverseModeGradient ||\n"
<< "      Mode == DerivativeMode::ForwardModeSplit) {\n"
<< "\n"
<< "    if (cachetype) {\n"
<< "      if (Mode != DerivativeMode::ReverseModeCombined) {\n"
<< "        cacheval = BuilderZ.CreatePHI(cachetype, 0);\n"
<< "      }\n"
<< "      cacheval = gutils->cacheForReverse(\n"
<< "          BuilderZ, cacheval, getIndex(&call, CacheType::Tape));\n"
<< "      if (Mode != DerivativeMode::ForwardModeSplit)\n"
<< "        cacheval = lookup(cacheval, Builder2);\n"
<< "    }\n"
<< "\n"
<< "    if (byRef) {\n";
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto incName = argOps->getArgNameStr(argPosition + 1);
      os
<< "      if (cache_" << incName << ") {\n"
<< "        true_" << incName << " =\n"
<< "            (cacheTypes.size() == 1)\n"
<< "                ? cacheval\n"
<< "                : Builder2.CreateExtractValue(cacheval, {cacheidx});\n"
<< "        auto alloc = allocationBuilder.CreateAlloca(intType);\n"
<< "        Builder2.CreateStore(true_" << incName << ", alloc);\n"
<< "        true_" << incName << " = Builder2.CreatePointerCast(\n"
<< "            alloc, call.getArgOperand(0)->getType());\n"
<< "        " << incName << " = true_" << incName << ";\n"
<< "        cacheidx++;\n"
<< "      } else if (need_" << incName << ") {\n"
<< "        if (Mode != DerivativeMode::ForwardModeSplit) {\n"
<< "          true_" << incName <<" = lookup(true_" << incName << ", Builder2);\n"
<< "          " << incName << " = true_" << incName << ";\n"
<< "        }\n"
<< "      }\n"
<< "\n";
    } else if (inputType->getName() == "len") {
      auto lenName = argOps->getArgNameStr(argPosition);
      os
<< "      if (cache_" << lenName << ") {\n"
<< "        len_" << lenName << " = (cacheTypes.size() == 1)\n"
<< "                    ? cacheval\n"
<< "                    : Builder2.CreateExtractValue(cacheval, {cacheidx});\n"
<< "        auto alloc = allocationBuilder.CreateAlloca(intType);\n"
<< "        Builder2.CreateStore(len_" << lenName << ", alloc);\n"
<< "        len_" << lenName << " = Builder2.CreatePointerCast(\n"
<< "            alloc, call.getArgOperand(0)->getType());\n"
<< "        cacheidx++;\n"
<< "      } else {\n"
<< "        if (Mode != DerivativeMode::ForwardModeSplit)\n"
<< "          len_" << lenName << " = lookup(len_" << lenName << ", Builder2);\n"
<< "      }\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  os
<< "    } else if (Mode != DerivativeMode::ForwardModeSplit) {\n";
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto incName = argOps->getArgNameStr(argPosition + 1);
  os
<< "      if (cache_" << incName << ") {\n"
<< "        true_" << incName << " = lookup(true_" << incName <<", Builder2);\n"
<< "        " << incName << " = true_" << incName << ";\n"
<< "      }\n";
    } else if (inputType->getName() == "len") {
      auto lenName = argOps->getArgNameStr(argPosition);
      os
<< "      len_" << lenName << " = lookup(len_" << lenName << ", Builder2);\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  os 
<< "    }\n";
  
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto vecPosition = argPosition;
      auto vecUsers = argUsers.lookup(vecPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      os
<< "    if (cache_" << vecName << ") {\n"
<< "      data_ptr_" << vecName << " = data_" << vecName << " =\n"
<< "          (cacheTypes.size() == 1)\n"
<< "              ? cacheval\n"
<< "              : Builder2.CreateExtractValue(cacheval, {cacheidx});\n"
<< "      cacheidx++;\n"
<< "      " << incName << " = ConstantInt::get(intType, 1);\n"
<< "      if (byRef) {\n"
<< "        auto alloc = allocationBuilder.CreateAlloca(intType);\n"
<< "        Builder2.CreateStore(" << incName << ", alloc);\n"
<< "        " << incName << " = Builder2.CreatePointerCast(\n"
<< "            alloc, call.getArgOperand(0)->getType());\n"
<< "      }\n"
<< "      if (type_" << vecName << "->isIntegerTy())\n"
<< "        data_" << vecName << " = Builder2.CreatePtrToInt(data_" 
<< vecName << ", type_" << vecName << ");\n"
<< "    }";

      if (vecUsers.size() > 0) {
        os 
<< "   else if (";
        bool first = true;
        // TODO: fix this, probably one else if for each possible user?
        for (auto user: vecUsers) {
          auto name = argOps->getArgNameStr(user);
          os << ((first) ? "" : " || ") << "active_" << name;
          first = false;
        }
        os << ") {\n"
<< "      data_" << vecName << " = lookup(gutils->getNewFromOriginal(arg_" 
<< vecName << "), Builder2);\n"
<< "    }\n";
      }
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  os 
<< "  } else {\n"
<< "\n";
  
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
  os
<< "    if (type_" << vecName << "->isIntegerTy())\n"
<< "      data_" << vecName << " = Builder2.CreatePtrToInt(data_" << vecName << ", type_" << vecName << ");\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  os 
<< "  }\n";
}

void findArgPositions(const std::vector<StringRef> toFind,
                      const DagInit *toSearch,
                      llvm::SmallSet<size_t, 5> &toInsert) {
  for (size_t i = 0; i < toSearch->getNumArgs(); i++) {
    if (DagInit *arg = dyn_cast<DagInit>(toSearch->getArg(i))) {
      // os << " Recursing. Magic!\n";
      findArgPositions(toFind, arg, toInsert);
    } else {
      auto name = toSearch->getArgNameStr(i);
      for (size_t i = 0; i < toFind.size(); i++) {
        if (name == toFind[i])
          toInsert.insert(i);
      }
    }
  }
}

llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> getUsedInputs(
    Record *pattern, std::vector<size_t> posActArgs) {

  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  std::vector<StringRef> inputs;
  for (size_t i = 0; i < argOps->getNumArgs(); i++) {
    inputs.push_back(argOps->getArgNameStr(i));
  }

  llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers{};

  // For each Gradient (say possibly active arg)
  ListInit *gradOps = pattern->getValueAsListInit("ArgDerivatives");
  assert(posActArgs.size() == gradOps->size() && "tblgen error");
  for (size_t i = 0; i < posActArgs.size(); i++) {
    auto val = gradOps->getElement(i);
    DagInit *resultRoot = cast<DagInit>(val);
    llvm::SmallSet<size_t, 5> set{};
    // collect all uses 
    findArgPositions(inputs, resultRoot, set);

    llvm::errs() << "Gradient " << i << " uses: ";

    for (auto position : set) {
      llvm::errs() << position <<" ";
      llvm::SmallSet<size_t, 5> val = argUsers.lookup(position);
      val.insert(posActArgs[i]);
      // assert(val.size() != 2);
      // if posActArgs[i] is active, 
      // then it will need to use the argument at position
      auto newVal = std::make_pair<>(position, val);
      argUsers.erase(position);
      argUsers.insert(newVal);
    }
  }
  return argUsers;
}


void emit_helper(Record *pattern, std::vector<size_t> actArgs,
                 raw_ostream &os) {
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  for (size_t i = 0; i < argOps->getNumArgs(); i++) {
    auto name = argOps->getArgNameStr(i);
    os 
<< "  auto arg_" << name << " = call.getArgOperand(" << i << ");\n"
<< "  auto type_" << name << " = arg_" << name << "->getType();\n"
<< "  bool uncacheable_" << name << " = uncacheable_args.find(called->getArg("  << i << "))->second;\n";
    if (std::count(actArgs.begin(), actArgs.end(), i)) {
      os 
<< "  bool active_" << name << " = !gutils->isConstantValue(arg_"
<< name << ");\n";
    }
    os 
<< "\n";
  }
}

llvm::SmallString<80> ValueType_helper(DagInit *argOps, 
    llvm::DenseMap<StringRef, StringRef> typeOfArgName, size_t actPos) {
  llvm::SmallString<80> valueTypes{};
  for (size_t pos = 0; pos < argOps->getNumArgs();) {
    auto name = argOps->getArgNameStr(pos);
    auto typeName = typeOfArgName.lookup(name);
    
    if (pos > 0) {
      valueTypes.append(", ");
    }

    if (typeName == "len") {
      valueTypes.append("ValueType::None");
    } else if (typeName == "fp") {
      auto floatName = argOps->getArgNameStr(pos);
      if (pos == actPos) {
        valueTypes.append("ValueType::Shadow");
      } else {
        valueTypes.append((Twine("cache_") + floatName + " ? ValueType::None : ValueType::Primal").str());
      }
    } else if (typeName == "vincData") {
      auto nextName = argOps->getArgNameStr(pos + 1);
      auto nextTypeName = typeOfArgName.lookup(nextName);
      assert(nextTypeName == "vincInc");
      auto vecName = argOps->getArgNameStr(pos);
      if (pos == actPos) {
        valueTypes.append("ValueType::Shadow, ValueType::None");
      } else {
        valueTypes.append((Twine("cache_") + vecName + " ? ValueType::None : ValueType::Primal, ValueType::None").str());
      }
      pos++; // extra inc, since vector cover two args
    } else {
      llvm::errs() << typeName << "\n";
      PrintFatalError("Unknown type!");
    }
    pos++;
  }
  return valueTypes;
}

llvm::SmallString<40> call_arg_helper(DagInit *argOps,
    llvm::DenseMap<StringRef, StringRef> typeOfArgName, llvm::StringRef actArg) {
  llvm::SmallString<40> result{};
  llvm::errs() << "call_arg_helper: " << argOps->getNumArgs() << "\n";
  for (size_t pos = 0; pos < argOps->getNumArgs();) {
    if (pos > 0) {
      result.append(", ");
    }

    auto arg = argOps->getArg(pos);
    if (DefInit *DefArg = dyn_cast<DefInit>(arg)) {
      auto Def = DefArg->getDef();
      if (Def->isSubClassOf("DiffeRet")) {
        result.append("dif");
      } else {
      PrintFatalError("Def that isn't a DiffeRet!");
      }
    } else {
      auto name = argOps->getArgNameStr(pos);
      llvm::errs() << "call_arg_helper: name: " << name << "\n";
      auto typeName = typeOfArgName.lookup(name);
      if (typeName == "len") {
        auto out = (Twine("len_") + name).str();
        llvm::errs() << "call_arg_helper: len: " << out << "\n";
        result.append((Twine("len_") + name).str());
      } else if (typeName == "fp") {
        if (name == actArg) {
          result.append((Twine("d_") + name).str());
        } else {
          result.append((Twine("fp_") + name).str());
        }
      } else if (typeName == "vincData") {
        auto nextName = argOps->getArgNameStr(pos+1);
        auto nextType = typeOfArgName.lookup(nextName);
        assert(nextType == "vincInc");
        if (name == actArg) {
          result.append((Twine("d_") + name + ", true_" + nextName).str());
        } else {
          result.append((Twine("data_") + name + ", " + nextName).str());
        }
        pos++; // extra ++ due to also handlinc vincInc
      } else if (typeName == "vincInc") {
        // might come without vincData, e.g. after DiffeRet
        result.append(name);
      } else {
        llvm::errs() << "name: " << name << " typename: " << typeName << "\n";
        llvm_unreachable("unimplemented input type!");
      }
    }
    pos++;
  }

  return result;
}

void emit_deriv_fnc(DagInit *resultTree, llvm::DenseMap<StringRef, StringRef> typeOfArgName,
    llvm::StringSet<> &handled, raw_ostream &os) {
  auto opName = resultTree->getOperator()->getAsString();
  auto Def = cast<DefInit>(resultTree->getOperator())->getDef();
  if (Def->isSubClassOf("b")) {
    auto dfnc_name = Def->getValueAsString("s");
    auto full_dfnc_name = llvm::Twine("(blas.prefix + \"") + dfnc_name + "\" + blas.suffix).str()";
    llvm::errs() << "found blas fnc: " << dfnc_name << "\n";
    if (handled.find(dfnc_name) != handled.end())
      return;
    else 
      handled.insert(dfnc_name);
    os 
<< "    auto derivcall_" << dfnc_name << " = gutils->oldFunc->getParent()->getOrInsertFunction(\n"
<< "      (blas.prefix +\"" << dfnc_name << "\" + blas.suffix).str(), Builder2.getVoidTy(),\n";
      // insert arg types based on .td file 
      bool first = true;
      std::vector<StringRef> usedArgStrs{};
      for (size_t i = 0; i < resultTree->getNumArgs(); i++) {
        Init* subArg = resultTree->getArg(i);
        if (DefInit *def = dyn_cast<DefInit>(subArg)) {
          usedArgStrs.push_back(""); // no need to process later
          if (def->getDef()->isSubClassOf("DiffeRet")) {
            os 
<< ((first) ? "" : ", ") << "byRef ? PointerType::getUnqual(call.getType()) : call.getType()\n";
          } else {
            PrintFatalError(Def->getLoc(), "PANIC! Unsupported Definit");
          }
        } else {
          auto argStr = resultTree->getArgNameStr(i);
          os 
<< ((first) ? "" : ", ") << "type_" << argStr; 
          usedArgStrs.push_back(argStr);
        }
        first = false;
        }
      os 
<< ");\n"
<< "#if LLVM_VERSION_MAJOR >= 9\n"
<< "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name << ".getCallee()))\n"
<< "#else\n"
<< "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name << "))\n"
<< "#endif\n"
<< "    {\n"
<< "      F->addFnAttr(Attribute::ArgMemOnly);\n"
<< "      if (byRef) {\n";
  for (size_t argPos = 0; argPos < usedArgStrs.size(); argPos++) {
    StringRef argName = usedArgStrs[argPos];
    auto argType = typeOfArgName.lookup(argName);
    if (argType == "len" || argType == "vincInc") {
      os 
<< "        F->addParamAttr(" << argPos << ", Attribute::ReadOnly);\n"
<< "        F->addParamAttr(" << argPos << ", Attribute::NoCapture);\n";
    }
  }


  os
<< "      }\n"
<< "    }\n\n";
 } else {
   PrintFatalError("Unhandled deriv Rule!");
 }
}

void emit_rev_rewrite_rules(Record *pattern, llvm::DenseMap<StringRef, StringRef> typeOfArgName,
    std::vector<size_t> actArgs,
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {
  
  ListInit *derivOps = pattern->getValueAsListInit("ArgDerivatives"); // correct
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  std::vector<Record *> inputTypes =
    pattern->getValueAsListOfDefs("inputTypes");

  os 
<< "  /* rev-rewrite */                                 \n"
<< "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "      Mode == DerivativeMode::ReverseModeGradient) {\n"
<< "    Value *dif = diffe(&call, Builder2);\n"
<< "    Value *alloc = nullptr;\n"
<< "    if (byRef) {\n"
<< "      alloc = allocationBuilder.CreateAlloca(fpType);\n"
<< "    }\n\n";

  llvm::StringSet handled{}; // We only emit one derivcall per blass call type
  llvm::errs() << "Number of grad defs: " << derivOps->size() << "\n";
  for (auto derivOp : *derivOps) {
    DagInit *resultTree = cast<DagInit>(derivOp); // correct
    emit_deriv_fnc(resultTree, typeOfArgName, handled, os);
  }
  os
<< "    // Vector Mode not handled yet\n";
 
  auto argPosition = 0;
  for (auto inputType : inputTypes) {
    auto typeName = inputType->getName();
    if (typeName == "vinc" || typeName == "fp") {
      auto name = argOps->getArgNameStr(argPosition);
  os
<< "    Value *d_" << name << " = active_" << name << "\n"
<< "     ? lookup(gutils->invertPointerM(arg_" << name << ", Builder2), Builder2)\n"
<< "     : nullptr;\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  os 
<< "    applyChainRule(\n"
<< "      Builder2,\n"
<< "      [&](";
  bool first = true;
  for (auto act : actArgs) {
    os 
<< ((first) ? "" : ", ") << "Value *" << "d_" + argOps->getArgNameStr(act);
    first = false;
  }
  os 
<< ") {\n"
<< "        if (byRef) {\n"
<< "          Builder2.CreateStore(dif, alloc);\n"
<< "          dif = alloc;\n"
<< "        }\n";


  for (size_t i = 0; i < derivOps->size(); i++) {
    auto actArg = actArgs[i];
    auto actName = argOps->getArgNameStr(actArg);
    auto derivOp = derivOps->getElement(i);
    DagInit *resultTree = cast<DagInit>(derivOp); // correct
    auto args = call_arg_helper(resultTree, typeOfArgName, actName);
    auto valueTypes = ValueType_helper(argOps, typeOfArgName, actArg);
    auto opName = resultTree->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultTree->getOperator())->getDef();
    if (!Def->isSubClassOf("b"))
      continue;// check
    auto dfnc_name = Def->getValueAsString("s");
    os
<< "      if (active_" << actName << ") {\n"
<< "        Value *args1[] = {" << args << "};\n"
<< "        Builder2.CreateCall(\n"
<< "          derivcall_" << dfnc_name << ", args1,\n"
<< "          gutils->getInvertedBundles(\n"
<< "            &call,\n"
<< "            {" << valueTypes << "},\n"
<< "            Builder2, /* lookup */ true));\n"
<< "      }\n";
  }
  os 
<< "    },\n"
<< "    ";

  first = true;
  for (auto act : actArgs) {
    os << ((first) ? "" : ", ") << "d_" + argOps->getArgNameStr(act);
    first = false;
  }
  os 
<< "    );\n"
<< "  setDiffe(\n"
<< "    &call,\n"
<< "    Constant::getNullValue(gutils->getShadowType(call.getType())),\n"
<< "    Builder2);\n"
<< "  }\n";
}

void emit_fwd_rewrite_rules(const Record *pattern, 
    const llvm::DenseMap<StringRef, StringRef> typeOfArgName, const std::vector<size_t> actArgs,
    const llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, 
                 raw_ostream &os) {
  os 
<< "  /* fwd-rewrite */                                 \n"
<< "  if (Mode == DerivativeMode::ForwardMode ||        \n"
<< "      Mode == DerivativeMode::ForwardModeSplit) {   \n"
<< "                                                    \n"
<< "#if LLVM_VERSION_MAJOR >= 11                        \n"
<< "    auto callval = call.getCalledOperand();         \n"
<< "#else                                               \n"
<< "    auto callval = call.getCalledValue();           \n"
<< "#endif                                            \n\n";

  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  auto argPosition = 0;
  for (auto inputType : inputTypes) {
    llvm::errs() << "LOOPING\n";
    auto typeName = inputType->getName();
    if (typeName == "vinc" || typeName == "fp") {
      auto name = argOps->getArgNameStr(argPosition);
  os
<< "    Value *d_" << name << " = active_" << name << "\n"
<< "     ? gutils->invertPointerM(arg_" << name << ", Builder2)\n"
<< "     : nullptr;\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  os
<< "    Value *dres = applyChainRule(\n"
<< "        call.getType(), Builder2,\n"
<< "        [&](";
  bool first = true;
  for (auto act : actArgs) {
    auto name = argOps->getArgNameStr(act);
    os 
<< ((first) ? "" : ", ") << "Value *d_" << name; 
    first = false;
  }
  os
<< "  ) {\n"
<< "      Value *dres = nullptr;\n";

  
  first = true;
  for (auto act : actArgs) {
    auto actName = argOps->getArgNameStr(act);
    auto dcallArgs = call_arg_helper(argOps, typeOfArgName, actName);
    auto valueTypes = ValueType_helper(argOps, typeOfArgName, act);
    os
<< "      if(active_" << actName << ") {\n"
<< "        Value *args1[] = {" << dcallArgs << "};\n\n"
<< "        auto Defs = gutils->getInvertedBundles(\n"
<< "          &call, {" << valueTypes << "}, Builder2, /* lookup */ false);\n";
  if (first) {
    os 
<< "#if LLVM_VERSION_MAJOR > 7\n"
<< "          dres = Builder2.CreateCall(call.getFunctionType(), callval, args1, Defs);\n"
<< "#else\n"
<< "          dres = Builder2.CreateCall(callval, args1, Defs);\n"
<< "#endif\n";
  } else {
    os 
<< "#if LLVM_VERSION_MAJOR > 7\n"
<< "        Value *nextCall = Builder2.CreateCall(\n"
<< "          call.getFunctionType(), callval, args1, Defs);\n"
<< "#else\n"
<< "        Value *nextCall = Builder2.CreateCall(callval, args1, Defs);\n"
<< "#endif\n"
<< "        if (dres)\n"
<< "          dres = Builder2.CreateFAdd(dres, nextCall);\n"
<< "        else\n"
<< "          dres = nextCall;\n";
  }
  os  
<< "      }\n";
  first = false;
  }
  os 
<< "      return dres;\n"
<< "    },\n"
<< "    ";

  first = true;
  for (auto act : actArgs) {
    os 
<< ((first) ? "" : ", ") << "d_" + argOps->getArgNameStr(act);
    first = false;
  }
  os 
<< ");\n"
<< "    setDiffe(&call, dres, Builder2);\n"
<< "  }\n";
}



void emit_handleBLAS(const std::vector<Record *> &blasPatterns, raw_ostream &os) {
  std::string handledBlasFunctions = "";
  bool first = true;
  for (auto blasPattern : blasPatterns) {
    auto newName = Twine((first) ? "" : ", ") + "\"" + blasPattern->getName() + "\"";
    handledBlasFunctions.append(newName.str());
    first = false;
  }
  os 
<<"struct BlasInfo {\n"
<<"  StringRef floatType;\n"
<<"  StringRef prefix;\n"
<<"  StringRef suffix;\n"
<<"  StringRef function;\n"
<<"};\n"
<<"\n"
<<"llvm::Optional<BlasInfo> extractBLAS(StringRef in) {\n"
<<"  llvm::Twine floatType[] = {\"s\", \"d\"}; // c, z\n"
<<"  llvm::Twine extractable[] = {" << handledBlasFunctions << "};\n"
<<"  llvm::Twine prefixes[] = {\"\", \"cblas_\", \"cublas_\"};\n"
<<"  llvm::Twine suffixes[] = {\"\", \"_\", \"_64_\"};\n"
<<"  for (auto t : floatType) {\n"
<<"    for (auto f : extractable) {\n"
<<"      for (auto p : prefixes) {\n"
<<"        for (auto s : suffixes) {\n"
<<"          if (in == (p + t + f + s).str()) {\n"
<<"            return llvm::Optional<BlasInfo>(BlasInfo{\n"
<<"                t.getSingleStringRef(),\n"
<<"                p.getSingleStringRef(),\n"
<<"                s.getSingleStringRef(),\n"
<<"                f.getSingleStringRef(),\n"
<<"            });\n"
<<"          }\n"
<<"        }\n"
<<"      }\n"
<<"    }\n"
<<"  }\n"
<<"  return llvm::NoneType();\n"
<<"}\n"
<<"\n"
<< "bool handleBLAS(llvm::CallInst &call, Function *called, BlasInfo blas,\n"
<< "                const std::map<Argument *, bool> &uncacheable_args) { \n"
<< "                                                                      \n"
<< "  bool result = true;                                                 \n"
<< "  if (!gutils->isConstantInstruction(&call)) {                        \n"
<< "    Type *fpType;                                                  \n"
<< "    if (blas.floatType == \"d\") {                                    \n"
<< "      fpType = Type::getDoubleTy(call.getContext());               \n"
<< "    } else if (blas.floatType == \"s\") {                             \n"
<< "      fpType = Type::getFloatTy(call.getContext());                \n"
<< "    } else {                                                          \n"
<< "      assert(false && \"Unreachable\");                               \n"
<< "    }                                                                 \n";
  first = true;
  for (auto pattern : blasPatterns) {
    auto name = pattern->getName();
    os
<< "    " << ((first) ? "" : "} else ") 
<< " if (blas.function == \"" << name << "\") {                           \n"
<< "      result = handle_" << name 
<< "(blas, call, called, uncacheable_args, fpType);                    \n";
    first = false;
  }
  os 
<< "    } else {                                                          \n"
<< "      llvm::errs() << \" fallback?\\n\";                              \n"
<< "      return false;                                                   \n"
<< "    }                                                                 \n"
<< "  }                                                                   \n"
<< "                                                                      \n"
<< "  if (Mode == DerivativeMode::ReverseModeGradient) {                  \n"
<< "    eraseIfUnused(call, /*erase*/ true, /*check*/ false);             \n"
<< "  } else {                                                            \n"
<< "    eraseIfUnused(call);                                              \n"
<< "  }                                                                   \n"
<< "                                                                      \n"
<< "  return result;                                                      \n"
<< "}                                                                     \n";
}

llvm::DenseMap<StringRef, StringRef> getArgTypes(const Record *pattern) {
  llvm::DenseMap<StringRef, StringRef> res{};
  std::vector<Record *> inputTypes =
    pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  size_t pos = 0;
  for (auto val : inputTypes) {
    auto argName = argOps->getArgNameStr(pos);
    if (val->getName() == "len") {
      res.insert(std::make_pair(argName, "len"));
    } else if (val->getName() == "fp") {
      res.insert(std::make_pair(argName, "fp"));
    } else if (val->getName() == "vinc") {
      res.insert(std::make_pair(argName, "vincData"));
      res.insert(std::make_pair(argOps->getArgNameStr(pos+1), "vincInc"));
    } else {
      PrintFatalError("Unknown type!");
      //TODO: panic
    }
    pos += val->getValueAsInt("nelem");
  }
  for (auto en : res) {
    llvm::errs() << " key: " << en.getFirst() << " val: " << en.getSecond() << "\n";
  }
  return res;
}


static void checkBlasCallsInDag(const RecordKeeper &RK,
                                const std::vector<Record *> blasPatterns,
                                const StringRef blasName,
                                const DagInit *toSearch) {

  for (size_t i = 0; i < toSearch->getNumArgs(); i++) {
    if (DagInit *arg = dyn_cast<DagInit>(toSearch->getArg(i))) {
      checkBlasCallsInDag(RK, blasPatterns, blasName, arg);
    }
  }

  auto Def = cast<DefInit>(toSearch->getOperator())->getDef();
  if (Def->isSubClassOf("b")) {
    auto numArgs = toSearch->getNumArgs();

    auto opName = Def->getValueAsString("s");
    auto CalledBlas = RK.getDef(opName);
    assert(CalledBlas);
    auto expectedNumArgs =
        CalledBlas->getValueAsDag("PatternToMatch")->getNumArgs();
    if (expectedNumArgs != numArgs) {
      llvm::errs() << "failed calling " << opName << " in the derivative of "
                   << blasName << " incorrect number of params. Expected "
                   << expectedNumArgs << " but got " << numArgs << "\n";
      assert(expectedNumArgs == numArgs);
    }
  }
}

/// Here we check that all the Blas derivatives who call another
/// blas function will use the correct amount of args
/// Later we might check for "types" too.
static void checkBlasCalls(const RecordKeeper &RK,
                           std::vector<Record *> blasPatterns) {
  for (auto pattern : blasPatterns) {
    ListInit *argOps = pattern->getValueAsListInit("ArgDerivatives");
    // for each possibly active parameter
    for (auto argOp : *argOps) {
      DagInit *resultRoot = cast<DagInit>(argOp);
      checkBlasCallsInDag(RK, blasPatterns, pattern->getName(), resultRoot);
    }
  }
}

/*
 * We create the following variables:
 *
 * vec: 
 * data_<vecName>
 * data_ptr_<vecName>
 * inc_<vecName>
 * true_<incName>
 * need_<incName>
 *
 * arg_<Name>
 * type_<Name>
 * active_<Name>
 * uncacheable_<Name>
 * d_<Name>
 *
 */
void emitBlasDerivatives(const RecordKeeper &RK, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = RK.getAllDerivedDefinitions("CallPattern");
  const auto &blasPatterns = RK.getAllDerivedDefinitions("CallBlasPattern");
  const auto &blas_modes = RK.getAllDerivedDefinitions("blas_modes");
  Record *attrClass = RK.getClass("Attr");

  // Make sure that we only call blass function b for calculating the derivative
  // of a iff we have defined b and pass the right amount of parameters.
  // TODO: type check params, as far as possible
  // TODO: assert unique input names.
  // TODO: assert mutable args are input names.
  // TODO: assert args in deriv defs exist
  checkBlasCalls(RK, blasPatterns);
  emit_handleBLAS(blasPatterns, os);
  // emitEnumMatcher(blas_modes, os);
  for (auto pattern : blasPatterns) {
    std::vector<size_t> posActArgs = getPossiblyActiveArgs(pattern);

    // For each input arg, we store a set including all users (by index).
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers = getUsedInputs(pattern, posActArgs);
    llvm::DenseMap<StringRef, StringRef> typeOfArgName = getArgTypes(pattern);

    emit_beginning(pattern, os);
    emit_helper(pattern, posActArgs, os);
    emit_castvals(pattern, posActArgs, os);
    emit_scalar_types(pattern, os);

    emit_caching(pattern, posActArgs, argUsers, os);
    emit_extract_calls(pattern, posActArgs, argUsers, os);

    emit_fwd_rewrite_rules(pattern, typeOfArgName, posActArgs, argUsers, os);
    emit_rev_rewrite_rules(pattern, typeOfArgName, posActArgs, argUsers, os);

    // writeEnums(pattern, blas_modes, os);
    emit_free_and_ending(pattern, posActArgs, os);
  }
}

//static void emitDerivatives(RecordKeeper &RK, raw_ostream &os) {
//  emitSourceFileHeader("Rewriters", os);
//  const auto &patterns = RK.getAllDerivedDefinitions("CallPattern");
//  const auto &blasPatterns = RK.getAllDerivedDefinitions("CallBlasPattern");
//  const auto &blas_modes = RK.getAllDerivedDefinitions("blas_modes");
//  Record *attrClass = RK.getClass("Attr");
//
//  emitFullDerivatives(patterns, os);
//}

static bool EnzymeTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
    case GenDerivatives:
      emitFullDerivatives(records, os); 
      return false;
    case GenBlasDerivatives:
      emitBlasDerivatives(records, os); 
      return false;
  }
  return true; // Not sure here?
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &EnzymeTableGenMain);
}

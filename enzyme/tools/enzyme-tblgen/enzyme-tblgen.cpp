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

enum ActionType { GenDerivatives };

static cl::opt<ActionType>
    action(cl::desc("Action to perform:"),
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

void emitFullDerivatives(const std::vector<Record *> &patterns,
                         raw_ostream &os) {
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
    for (auto *nameI : *cast<ListInit>(pattern->getValueAsListInit("name"))) {
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
<< "    castvals[" << argIdx
<< "  ] = PointerType::getUnqual(innerType);\n";
  }
  os 
<< "  Value *cacheval;\n\n"
<< "  /* ending castvalls */\n";
}

void emit_inttype(Record *pattern, raw_ostream &os) {
  // We only look at the type of the first integer showing up.
  // This allows to learn if we use Fortran abi (byRef) or cabi
  size_t firstIntPos = 0;
  bool found = false;
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  for (auto val : inputTypes) {
    if (val->getName() == "len") {
      found = true;
      // os << "first integer at: " << firstIntPos << "\n";
      break;
    }
    firstIntPos += val->getValueAsInt("nelem");
  }
  auto name = argOps->getArgNameStr(firstIntPos);
  assert(found && "no int type found in blas call");

  os
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
<< "    const std::map<Argument *, bool> &uncacheable_args, Type *innerType) {\n"
<< "  \n"
<< "  CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));\n"
<< "  IRBuilder<> BuilderZ(newCall);\n"
<< "  BuilderZ.setFastMathFlags(getFast());\n"
<< "  IRBuilder<> allocationBuilder(gutils->inversionAllocs);\n"
<< "  allocationBuilder.setFastMathFlags(getFast());\n"
<< "  auto &DL = gutils->oldFunc->getParent()->getDataLayout();\n"
<< "  auto *called = call.getCalledFunction();\n\n";
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
  llvm::errs() << activeArgs.size() << name;
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
  os 
    // TODO: adjust count / getArgOperand(0) based on first int?
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
<< "    if (byRef) {\n"
<< "      if (countcache) {\n"
<< "        count = (cacheTypes.size() == 1)\n"
<< "                    ? cacheval\n"
<< "                    : Builder2.CreateExtractValue(cacheval, {cacheidx});\n"
<< "        auto alloc = allocationBuilder.CreateAlloca(intType);\n"
<< "        Builder2.CreateStore(count, alloc);\n"
<< "        count = Builder2.CreatePointerCast(\n"
<< "            alloc, call.getArgOperand(0)->getType());\n"
<< "        cacheidx++;\n"
<< "      } else {\n"
<< "        if (Mode != DerivativeMode::ForwardModeSplit)\n"
<< "          count = lookup(count, Builder2);\n"
<< "      }\n"
<< "\n";
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
<< "        cacheidx++;\n"
<< "      } else if (need_" << incName << ") {\n"
<< "        if (Mode != DerivativeMode::ForwardModeSplit)\n"
<< "          true_" << incName <<" = lookup(true_" << incName << ", Builder2);\n"
<< "      }\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  os
<< "    } else if (Mode != DerivativeMode::ForwardModeSplit) {\n"
<< "      count = lookup(count, Builder2);\n"
<< "\n";
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto incName = argOps->getArgNameStr(argPosition + 1);
  os
<< "      if (cache_" << incName << ") \n"
<< "        true_" << incName << " = lookup(true_" << incName <<", Builder2);\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  os 
<< "    }\n" << "\n";

  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
  os
<< "    Value *data_" << vecName << " = gutils->getNewFromOriginal(arg_" << vecName << ");\n"
<< "    Value *data_ptr_" << vecName << " = nullptr;\n"
<< "    Value *" << incName << " = true_" << incName << ";\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto vecPosition = argPosition;
      auto vecUsers = argUsers.lookup(vecPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      os // todo update numbers
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
        for (auto user: vecUsers) {
          auto name = argOps->getArgNameStr(user);
          os << "active_" << name;
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
<< "  } else {\n";
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      os
<< "    Value *data_" << vecName << " = gutils->getNewFromOriginal(arg_" << vecName << ");\n"
<< "    Value *data_ptr_" << vecName << " = nullptr;\n"
<< "    Value *" << incName << " = true_" << incName << ";\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  os << "\n";
  
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

void emit_new_vars(Record *pattern, std::vector<size_t> actArgs,
                   raw_ostream &os) {
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");

  for (auto act : actArgs) {
    auto name = argOps->getArgNameStr(act);
    os 
<< "  auto new_" << name << " = lookup(gutils->getNewFromOriginal(arg_" << name << "), Builder2),\n";
  }
}

void emit_blas_call(Record *pattern, std::vector<size_t> actArgs,
                    raw_ostream &os) {
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  // auto blassCall = gutils->oldFunc->getParent()->getOrInsertFunction(
  //     axpyName, Builder2.getVoidTy(), type_n, innerType, type_y, type_incy,
  //     type_x, type_incx);
  // SmallVector<Value *, 6> args = {new_n,  ConstantFP::get(innerType, 1.0),
  //                                 diff_y, new_incy,
  //                                 diff_x, new_incx};
  // Builder2.CreateCall(axpyCall, args,
  //                     gutils->getInvertedBundles(
  //                         &call,
  //                         {ValueType::None, ValueType::Shadow,
  //                         ValueType::None,
  //                          ValueType::Shadow, ValueType::None},
  //                         Builder2, true));
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

void emit_fwd_rewrite_rules(Record *pattern, std::vector<size_t> actArgs,
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, 
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

// TODO
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  auto argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
  os
<< "    Value *d_" << vecName << " = active_" << vecName << "\n"
<< "     ? gutils->invertPointerM(arg_" << vecName << ", Builder2)\n"
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
    if (!first) 
      os << ", ";
    auto name = argOps->getArgNameStr(act);
    os << "Value *d_" << name; 
    first = false;
  }
  os
<< "  ) {\n"
<< "      value *dres = nullptr;\n";

  std::vector<llvm::Twine> d_args{}; // TODO inc from active vec becomes trueXinc
  for (auto act : actArgs) {
    auto actName = argOps->getArgNameStr(act);
    d_args.push_back("d_" + actName);
  }
  
  first = true;
  for (auto act : actArgs) {
    auto actName = argOps->getArgNameStr(act);
    os
<< "      if(active_" << actName << ") {\n"
<< "        Value *args1[] = {";
  bool first2 = true;
  for (auto arg : d_args) {
    if (!first2) 
      os << ", ";
    os 
<< arg; 
    first2 = false;
  }
  os 
<< "};\n\n"
<< "        auto Defs = gutils->getInvertedBundles(\n"
<< "          &call, {/* currently unused, to be fixed */}, Builder2, /* lookup */ false);\n";
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
<< "    },\n";
  first = true;
  for (auto arg : d_args) {
    if (!first)
      os << ", ";
    os 
<< arg;
    first = false;
  }
  os 
<< ");\n"
<< "    setDiffe(&call, dres, Builder2);\n"
<< "  }\n";
}



void emit_handleBLAS(const std::vector<Record *> &blasPatterns, raw_ostream &os) {
  os 
<< "bool handleBLAS(llvm::CallInst &call, Function *called, BlasInfo blas,\n"
<< "                const std::map<Argument *, bool> &uncacheable_args) { \n"
<< "                                                                      \n"
<< "  bool result = true;                                                 \n"
<< "  if (!gutils->isConstantInstruction(&call)) {                        \n"
<< "    Type *innerType;                                                  \n"
<< "    if (blas.floatType == \"d\") {                                    \n"
<< "      innerType = Type::getDoubleTy(call.getContext());               \n"
<< "    } else if (blas.floatType == \"s\") {                             \n"
<< "      innerType = Type::getFloatTy(call.getContext());                \n"
<< "    } else {                                                          \n"
<< "      assert(false && \"Unreachable\");                               \n"
<< "    }                                                                 \n";
  bool first = true;
  for (auto pattern : blasPatterns) {
    auto name = pattern->getName();
    if (!first) {
      os 
<< "    } else if (blas.function == \"" << name << "\") {                 \n";
    } else {
      os 
<< "    if (blas.function == \"" << name << "\") {                        \n";
    }
    os 
<< "      result = handle" << name 
<< "(blas, call, called, uncacheable_args, innerType);\n";
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
 * new_<Name>
 * d_<Name>
 *
 */
void emitBlasDerivatives(const std::vector<Record *> &blasPatterns,
                         const std::vector<Record *> &blas_modes,
                         raw_ostream &os) {
  emit_handleBLAS(blasPatterns, os);
  // emitEnumMatcher(blas_modes, os);
  for (auto pattern : blasPatterns) {
    if (pattern->getName() != "dot") 
      continue;
    // TODO: assert unique input names.
    std::vector<size_t> posActArgs = getPossiblyActiveArgs(pattern);

    // For each input arg, we store a set including all users (by index).
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers = getUsedInputs(pattern, posActArgs);

    // std::vector<std::vector<size_t>> cacheArgPos =
    //     getToCachePos(pattern, posActArgs);
    //  For each active arg we want to have a list of input args required.
    //  We use it to find out if we need to cache them.
    // assert(posActArgs.size() == cacheArgPos.size());
    

    emit_beginning(pattern, os);
    emit_helper(pattern, posActArgs, os);
    emit_castvals(pattern, posActArgs, os);
    emit_inttype(pattern, os);

    emit_caching(pattern, posActArgs, argUsers, os);
    emit_extract_calls(pattern, posActArgs, argUsers, os);

    emit_fwd_rewrite_rules(pattern, posActArgs, argUsers, os);
    //emit_rev_rewrite_rules(pattern, posActArgs, argUsers, os);

    // writeEnums(pattern, blas_modes, os);
    emit_free_and_ending(pattern, posActArgs, os);
  }
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

static void emitDerivatives(RecordKeeper &RK, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = RK.getAllDerivedDefinitions("CallPattern");
  const auto &blasPatterns = RK.getAllDerivedDefinitions("CallBlasPattern");
  const auto &blas_modes = RK.getAllDerivedDefinitions("blas_modes");
  Record *attrClass = RK.getClass("Attr");

  // Make sure that we only call blass function b for calculating the derivative
  // of a iff we have defined b and pass the right amount of parameters.
  // TODO: type check params, as far as possible
  checkBlasCalls(RK, blasPatterns);

  // We have full access to the source code to differentiate it
  // emitFullDerivatives(patterns, os);

  // Improve UX / comp-time by handling Blas calls extra.
  emitBlasDerivatives(blasPatterns, blas_modes, os);
}

static bool EnzymeTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case GenDerivatives:
    emitDerivatives(records, os); return false;
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &EnzymeTableGenMain);
}

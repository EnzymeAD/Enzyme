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


void emitEnumMatcher(const std::vector<Record *> &blas_modes, raw_ostream &os) {
  for (auto mode : blas_modes) {
    auto name = mode->getName();
    auto sub_modes = mode->getValueAsListOfStrings("modes");
    llvm::errs() << "std::string read_" << name
                 << "(llvm::CallInst &call, size_t pos) {\n"
                 << "  std::string s = call.getArgOperand(pos)->getValue();\n";
    for (auto sub_mode : sub_modes) {
      llvm::errs() << "  if (s == \"" << sub_mode << "\")\n"
                   << "    return \"" << sub_mode << "\";\n";
    }
    llvm::errs() << "  assert(false && \"failed reading " << name << "\");\n"
                 << "}\n\n";
  }
}

void writeEnums(Record *pattern, const std::vector<Record *> &blas_modes,
                raw_ostream &os) {
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  for (auto inputType : inputTypes) {
    if (inputType->isSubClassOf("blas_modes")) {
      llvm::errs() << inputType->getName() << ": ";
      for (auto a : inputType->getValueAsListOfStrings("modes")) {
        llvm::errs() << a << " ";
      }
      llvm::errs() << "\n";
    }
  }
  DagInit *tree = pattern->getValueAsDag("PatternToMatch");
  for (int i = 0, e = tree->getNumArgs(); i != e; ++i) {
    // llvm::errs() << tree->getArgNameStr(i) << " ";
    //  auto optns = blas_arg->getValueAsListOfStrings("modes");
    //  for (auto optn : optns)
    //    llvm::errs() << optn << " ";
    //  }
  }
}

void emit_castvals(Record *pattern, std::vector<size_t> activeArgs,
                   raw_ostream &os) {
  llvm::errs() << "Type *castvalls[" << activeArgs.size() << "];\n";

  for (auto argPos : llvm::enumerate(activeArgs)) {
    size_t argIdx = argPos.index();
    llvm::errs() << "if (auto PT = dyn_cast<PointerType>(call.getArgOperand("
                 << argPos.value() << ")->getType()))\n"
                 << "  castvals[" << argIdx << "] = PT;\n"
                 << "else\n"
                 << "  castvals[" << argIdx
                 << "] = PointerType::getUnqual(innerType);\n";
  }
  //              << "Value *undefinit = UndefValue::get(cachetype);\n"
  llvm::errs() << "Value *cacheval;\n\n";
}

void emit_inttype(Record *pattern, raw_ostream &os) {
  // We only look at the type of the first integer showing up.
  // This allows to learn if we use Fortran abi (byRef) or cabi
  size_t firstIntPos = 0;
  bool found = false;
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  for (auto val : inputTypes) {
    if (val->getName() == "len") {
      found = true;
      // llvm::errs() << "first integer at: " << firstIntPos << "\n";
      break;
    }
    firstIntPos += val->getValueAsInt("nelem");
  }
  assert(found && "no int type found in blas call");

  llvm::errs()
      << "IntegerType *intType = dyn_cast<IntegerType>(call.getOperand("
      << firstIntPos << ")->getType());\n"
      << "bool byRef = false;\n" // Fortran Abi?
      << "if (!intType) {\n"
      << "  auto PT = cast<PointerType>(call.getOperand(" << firstIntPos
      << ")->getType());\n"
      << "  if (blas.suffix.contains(\"64\"))\n"
      << "    intType = IntegerType::get(PT->getContext(), 64);\n"
      << "  else\n"
      << "    intType = IntegerType::get(PT->getContext(), 32);\n"
      << "  byRef = true;\n"
      << "}\n\n";
}


void emit_beginning(Record *pattern, raw_ostream &os) {
  auto name = pattern->getName();
  llvm::errs()
      << "\nbool handle_" << name
      << "(BlasInfo blas, llvm::CallInst &call, "
         "Function *called,\n"
      << "const std::map<Argument *, bool> &uncacheable_args,\n"
      << "Type *innerType) {\n"
      << "CallInst *const newCall = "
         "cast<CallInst>(gutils->getNewFromOriginal(&call));\n"
      << "IRBuilder<> BuilderZ(newCall);\n"
      << "BuilderZ.setFastMathFlags(getFast());\n"
      << "IRBuilder<> allocationBuilder(gutils->inversionAllocs);\n"
      << "allocationBuilder.setFastMathFlags(getFast());\n\n"
      << "auto &DL = gutils->oldFunc->getParent()->getDataLayout();\n\n";
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

void emit_ending(Record *pattern, raw_ostream &os) {

  llvm::errs() << "if (gutils->knownRecomputeHeuristic.find(&call) !=\n"
               << "gutils->knownRecomputeHeuristic.end()) {\n"
               << "if (!gutils->knownRecomputeHeuristic[&call]) {\n"
               << "gutils->cacheForReverse(BuilderZ, newCall,\n"
               << " getIndex(&call, CacheType::Self));\n"
               << "}\n"
               << "}\n";

  llvm::errs() << "if (Mode == DerivativeMode::ReverseModeGradient) {\n"
               << "  eraseIfUnused(*orig, /*erase*/ true, /*check*/ false);\n"
               << "} else {\n"
               << "  eraseIfUnused(*orig);\n"
               << "}\n"
               << "return true;\n"
               << "}\n\n";
}

void emit_vinc_caching(Record *pattern, std::vector<size_t> actArgs, 
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {

  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  size_t argPosition = 0;
  std::vector<std::string> cacheVars{};

  // Debug
  // for (size_t i = 0; i < 6; i++) {
  //   llvm::errs() << "arg " << i << " is used by: ";
  //   llvm::SmallSet<size_t, 5> x = argUsers.lookup(i);
  //   for (auto val : x) 
  //     llvm::errs() << val << " ";
  //   llvm::errs() << "\n";
  // }

  for (auto val : inputTypes) {
    if (val->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto vecPosition = argPosition;
      auto vecUsers = argUsers.lookup(vecPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      auto incPosition = argPosition + 1;
      auto incUsers = argUsers.lookup(incPosition);
      llvm::errs() << " bool cache_" << vecName
                   << " = Mode != DerivativeMode::ForwardMode &&\n"
                   << "         uncacheable_" << vecName;
      for (size_t user: vecUsers) {
        auto name = argOps->getArgNameStr(user);
        llvm::errs() << " && active_" << name;
      }
      llvm::errs() << ";\n";
      llvm::errs() << " bool cache_" << incName << " = false;\n";
      cacheVars.push_back("cache_" + vecName.str());
      // xinc is needed to be preserved if
      // 1) it is potentially overwritten AND EITHER
      //     a) x is active (for performing the shadow increment) or
      //     b) we're not caching x and need xinc to compute the
      //     derivative of a different variable
      llvm::errs() << " bool need_" << incName << " = (active_" << vecName;
      if (incUsers.size() > 0) {
        llvm::errs() << " || (!cache_" << vecName << " && (";
        bool first = true;
        for (size_t user: incUsers) {
          auto name = argOps->getArgNameStr(user);
          if (!first)
            llvm::errs() << " || ";
          llvm::errs() << "active_" << name;
          first = false;
        }
        llvm::errs() << "));\n";
      }

      llvm::errs() 
        << " if (byRef && uncacheable_" << incName << " && need_" << incName << ") {\n"
        << "   cacheTypes.push_back(intType);\n"
        << "   cache_" << incName << " = true;\n "
        << " }\n\n";

    }
    argPosition += val->getValueAsInt("nelem");
  }
  llvm::errs() << "int numCached = ";
  for (size_t i = 0; i < cacheVars.size(); i++) {
    if (i > 0)
      llvm::errs() << " + ";
    llvm::errs() << "(int) " << cacheVars[i];
  }
  llvm::errs() << ";\n";
  llvm::errs() << "bool anyCache = (numCached > 0);\n";
}

// This impl is slightly suboptimal as for lv3 blas it might cache one or two
// integers that might not be needed under all circumstances. Shouldn't matter.
void emit_count_caching(Record *pattern, std::vector<size_t> actArgs,
                        raw_ostream &os) {
  llvm::errs() << "   // count must be preserved if overwritten\n";
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  size_t argPosition = 0;
  for (auto val : inputTypes) {
    if (val->getName() == "len") {
      auto name = argOps->getArgNameStr(argPosition);
      llvm::errs() << " bool cache_" << name << " = false;\n";
      llvm::errs() << " if (byRef && uncacheable_" << name << ") {\n"
                   << "     cacheTypes.push_back(intType);\n"
                   << "     cache_" << name << " = true;\n"
                   << "   }\n";
    }
    argPosition += val->getValueAsInt("nelem");
  }
}

void emit_cache_for_reverse(Record *pattern, std::vector<size_t> actArgs, 
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {
  llvm::errs() << "if ((Mode == DerivativeMode::ReverseModeCombined ||\n"
               << "         Mode == DerivativeMode::ReverseModePrimal) && cachetype) {\n";

  llvm::errs() << "SmallVector<Value *, 2> cacheValues;\n";
  llvm::errs() << "auto size = ConstantInt::get(intType, "
                  "DL.getTypeSizeInBits(innerType) / 8);\n";
  
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");

  size_t argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "len") {
      auto name = argOps->getArgNameStr(argPosition);
      llvm::errs() << "Value *count = gutils->getNewFromOriginal(arg_" << name << ");\n";
      break;
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  argPosition = 0;
  for (auto inputType : inputTypes) {

    // cache count if needed
    if (inputType->getName() == "len") {
      llvm::errs()
        << "          if (byRef) {\n"
        << "            count = BuilderZ.CreatePointerCast(count, PointerType::getUnqual(intType));\n"
        << "#if LLVM_VERSION_MAJOR > 7\n"
        << "            count = BuilderZ.CreateLoad(intType, count);\n"
        << "#else\n"
        << "            count = BuilderZ.CreateLoad(count);\n"
        << "#endif\n"
        << "            if (countcache)\n"
        << "              cacheValues.push_back(count);\n"
        << "          }\n";
    } else if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
      auto incPosition = argPosition + 1;

      // cache vinc's if needed.
      llvm::errs() 
        << "          Value *" << incName << " = gutils->getNewFromOriginal(" << incName <<");\n"
        << "          if (byRef) {\n"
        << "            " << incName << " = BuilderZ.CreatePointerCast(" << incName << ",\n"
           "PointerType::getUnqual(intType));\n"
        << "#if LLVM_VERSION_MAJOR > 7\n"
        << "            " << incName << " = BuilderZ.CreateLoad(intType, " << incName << ");\n"
        << "#else\n"
        << "            " << incName << " = BuilderZ.CreateLoad(" << incName << ");\n"
        << "#endif\n"
        << "            if (cache_" << incName << ")\n"
        << "              cacheValues.push_back(" << incName << ");\n"
        << "          }\n";
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
      llvm::errs()
      << " if (cache_" << vecName << ") {\n"
      << "           auto dmemcpy = getOrInsertMemcpyStrided(\n"
      << "               *gutils->oldFunc->getParent(), cast<PointerType>(castvals[" << i << "]),\n"
      << "               size->getType(), 0, 0);\n"
      << "           auto malins = CallInst::CreateMalloc(\n"
      << "               gutils->getNewFromOriginal(&call), size->getType(), innerType,\n"
      << "               size, count, nullptr, \"\");\n"
      << "           Value *arg = BuilderZ.CreateBitCast(malins, castvals[" << i << "]);\n"
      << "           Value *args[4] = {arg,\n"
         "                              gutils->getNewFromOriginal(call.getArgOperand(" << argPosition << ")),\n"
      << "                              count, " << incName << "};\n"

      << "           if (args[1]->getType()->isIntegerTy())\n"
      << "             args[1] = BuilderZ.CreateIntToPtr(args[1], castvals[" << i << "]);\n"

      << "           BuilderZ.CreateCall(\n"
      << "               dmemcpy, args,\n"
      << "               gutils->getInvertedBundles(&call,\n"
      << "                                          {ValueType::None, ValueType::Shadow,\n"
      << "                                           ValueType::None, ValueType::None,\n"
      << "                                           ValueType::None},\n"
      << "                                          BuilderZ, /*lookup*/ false));\n"
      << "           cacheValues.push_back(arg);\n"
      << "         }\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
    i++;
  }

  llvm::errs()
      << "      if (cacheValues.size() == 1) {\n"
      << "        cacheval = cacheValues[0];\n"
      << "      } else {\n"
      << "        cacheval = UndefValue::get(cachetype);\n"
      << "        for (auto tup : llvm::enumerate(cacheValues))\n"
      << "          cacheval = BuilderZ.CreateInsertValue(cacheval, "
         "tup.value(), tup.index());\n"
      << "      }\n"
      << "      gutils->cacheForReverse(BuilderZ, cacheval,\n"
      << "                              getIndex(&call, CacheType::Tape));\n"
      << " }\n";

  llvm::errs()
      << "    unsigned cacheidx = 0;\n"
      << "    Value *count = gutils->getNewFromOriginal(call.getArgOperand(0));\n"; // todo adjust idx
 
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto incName = argOps->getArgNameStr(argPosition + 1);
      llvm::errs() << "    Value *true_" << incName 
        << " = gutils->getNewFromOriginal(arg_" << incName << ");\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  llvm::errs()
      << "    IRBuilder<> Builder2(call.getParent());\n"
      << "    switch (Mode) {\n"
      << "      case DerivativeMode::ReverseModeCombined:\n"
      << "      case DerivativeMode::ReverseModeGradient:\n"
      << "        getReverseBuilder(Builder2);\n"
      << "        break;\n"
      << "      case DerivativeMode::ForwardMode:\n"
      << "      case DerivativeMode::ForwardModeSplit:\n"
      << "        Builder2.SetInsertPoint(BuilderZ.GetInsertBlock(),\n"
      << "                                BuilderZ.GetInsertPoint());\n"
      << "        Builder2.setFastMathFlags(BuilderZ.getFastMathFlags());\n"
      << "        break;\n"
      << "      case DerivativeMode::ReverseModePrimal:\n"
      << "        break;\n"
      << "    }\n\n";
}

void emit_free(Record *pattern, std::vector<size_t> actArgs,
                  raw_ostream &os) {
    llvm::errs()
      << "          if (Mode == DerivativeMode::ReverseModeCombined ||\n"
      << "            Mode == DerivativeMode::ReverseModeGradient ||\n"
      << "            Mode == DerivativeMode::ForwardModeSplit) {\n"
      << "          if (shouldFree()) {\n";
  size_t argPosition = 0;
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto name = argOps->getArgNameStr(argPosition);
      llvm::errs()
        << "            if (cache_" << name << ") {\n"
        << "              auto ci = cast<CallInst>(CallInst::CreateFree(\n"
        << "                  Builder2.CreatePointerCast(\n"
        << "                      data_ptr_" << name <<", Type::getInt8PtrTy(data_ptr_" << name << "->getContext())),\n"
        << "                  Builder2.GetInsertBlock()));\n"
        << "#if LLVM_VERSION_MAJOR >= 14\n"
        << "              ci->addAttributeAtIndex(AttributeList::FirstArgIndex,\n"
        << "                                      Attribute::NonNull);\n"
        << "#else\n"
        << "              ci->addAttribute(AttributeList::FirstArgIndex,\n"
        << "                               Attribute::NonNull);\n"
        << "#endif\n"
        << "              if (ci->getParent() == nullptr) {\n"
        << "                Builder2.Insert(ci);\n"
        << "              }\n"
        << "            }\n";
    }
    argPosition += inputType->getValueAsInt("nelem");

    }
    llvm::errs()
        << "          }\n"
        << "        }\n";
}

void emit_extract_calls(Record *pattern, std::vector<size_t> actArgs, 
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {
  size_t argPosition = 0;
  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  llvm::errs() 
<< "        if (Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "            Mode == DerivativeMode::ReverseModeGradient ||\n"
<< "            Mode == DerivativeMode::ForwardModeSplit) {\n"
<< "\n"
<< "        if (cachetype) {\n"
<< "          if (Mode != DerivativeMode::ReverseModeCombined) {\n"
<< "            cacheval = BuilderZ.CreatePHI(cachetype, 0);\n"
<< "          }\n"
<< "          cacheval = gutils->cacheForReverse(\n"
<< "              BuilderZ, cacheval, getIndex(&call, CacheType::Tape));\n"
<< "          if (Mode != DerivativeMode::ForwardModeSplit)\n"
<< "            cacheval = lookup(cacheval, Builder2);\n"
<< "        }\n"
<< "\n"
<< "        if (byRef) {\n"
<< "          if (countcache) {\n"
<< "            count = (cacheTypes.size() == 1)\n"
<< "                        ? cacheval\n"
<< "                        : Builder2.CreateExtractValue(cacheval, {cacheidx});\n"
<< "            auto alloc = allocationBuilder.CreateAlloca(intType);\n"
<< "            Builder2.CreateStore(count, alloc);\n"
<< "            count = Builder2.CreatePointerCast(\n"
<< "                alloc, call.getArgOperand(0)->getType());\n"
<< "            cacheidx++;\n"
<< "          } else {\n"
<< "            if (Mode != DerivativeMode::ForwardModeSplit)\n"
<< "              count = lookup(count, Builder2);\n"
<< "          }\n"
<< "\n";
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto incName = argOps->getArgNameStr(argPosition + 1);
  llvm::errs()
<< "          if (cache_" << incName << ") {\n"
<< "            true_" << incName << " =\n"
<< "                (cacheTypes.size() == 1)\n"
<< "                    ? cacheval\n"
<< "                    : Builder2.CreateExtractValue(cacheval, {cacheidx});\n"
<< "            auto alloc = allocationBuilder.CreateAlloca(intType);\n"
<< "            Builder2.CreateStore(true_" << incName << ", alloc);\n"
<< "            true_" << incName << " = Builder2.CreatePointerCast(\n"
<< "                alloc, call.getArgOperand(0)->getType());\n"
<< "            cacheidx++;\n"
<< "          } else if (need_" << incName << ") {\n"
<< "            if (Mode != DerivativeMode::ForwardModeSplit)\n"
<< "              true_" << incName <<" = lookup(true_" << incName << ", Builder2);\n"
<< "          }\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  llvm::errs()
<< "        } else if (Mode != DerivativeMode::ForwardModeSplit) {\n"
<< "          count = lookup(count, Builder2);\n"
<< "\n";
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto incName = argOps->getArgNameStr(argPosition + 1);
  llvm::errs()
<< "          if (cache_" << incName << ") \n"
<< "            true_" << incName << " = lookup(true_" << incName <<", Builder2);\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }
  llvm::errs() << "        }\n" << "\n";

  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
  llvm::errs()
<< "        Value *data_" << vecName << " = gutils->getNewFromOriginal(arg_" << vecName << ");\n"
<< "        Value *data_ptr_" << vecName << " = nullptr;\n"
<< "        Value *" << incName << " = true_" << incName << ";\n"
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
      llvm::errs() // todo update numbers
<< "        if (cache_" << vecName << ") {\n"
<< "          data_ptr_" << vecName << " = data_" << vecName << " =\n"
<< "              (cacheTypes.size() == 1)\n"
<< "                  ? cacheval\n"
<< "                  : Builder2.CreateExtractValue(cacheval, {cacheidx});\n"
<< "          cacheidx++;\n"
<< "          " << incName << " = ConstantInt::get(intType, 1);\n"
<< "          if (byRef) {\n"
<< "            auto alloc = allocationBuilder.CreateAlloca(intType);\n"
<< "            Builder2.CreateStore(" << incName << ", alloc);\n"
<< "            " << incName << " = Builder2.CreatePointerCast(\n"
<< "                alloc, call.getArgOperand(0)->getType());\n"
<< "          }\n"
<< "          if (type_" << vecName << "->isIntegerTy())\n"
<< "            data_" << vecName << " = Builder2.CreatePtrToInt(data_" 
<< vecName << ", type_" << vecName << ");\n"
<< "        }";

      if (vecUsers.size() > 0) {
        llvm::errs() << " else if (";
        for (auto user: vecUsers) {
          auto name = argOps->getArgNameStr(user);
          llvm::errs() << "active_" << name;
        }
        llvm::errs() << ") {\n"
<< "          data_" << vecName << " = lookup(gutils->getNewFromOriginal(arg_" 
<< vecName << "), Builder2);\n"
<< "        }\n";
      }
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

// << "        if (ycache) {\n"
// << "          ydata_ptr = ydata =\n"
// << "              (cacheTypes.size() == 1)\n"
// << "                  ? cacheval\n"
// << "                  : Builder2.CreateExtractValue(cacheval, {cacheidx});\n"
// << "          cacheidx++;\n"
// << "          yinc = ConstantInt::get(intType, 1);\n"
// << "          if (byRef) {\n"
// << "            auto alloc = allocationBuilder.CreateAlloca(intType);\n"
// << "            Builder2.CreateStore(yinc, alloc);\n"
// << "            yinc = Builder2.CreatePointerCast(\n"
// << "                alloc, call.getArgOperand(0)->getType());\n"
// << "          }\n"
// << "          if (call.getArgOperand(3)->getType()->isIntegerTy())\n"
// << "            ydata = Builder2.CreatePtrToInt(ydata,\n"
// << "                                            call.getArgOperand(3)->getType());\n"
// << "        } else if (!gutils->isConstantValue(call.getArgOperand(1))) {\n"
// << "          ydata = lookup(gutils->getNewFromOriginal(call.getArgOperand(3)),\n"
// << "                         Builder2);\n"
// << "        }\n"
  llvm::errs() << "        } else {\n";
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
  llvm::errs()
<< "        Value *data_" << vecName << " = gutils->getNewFromOriginal(arg_" << vecName << ");\n"
<< "        Value *data_ptr_" << vecName << " = nullptr;\n"
<< "        Value *" << incName << " = true_" << incName << ";\n"
<< "\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  llvm::errs() << "\n";
  
  argPosition = 0;
  for (auto inputType : inputTypes) {
    if (inputType->getName() == "vinc") {
      auto vecName = argOps->getArgNameStr(argPosition);
      auto incName = argOps->getArgNameStr(argPosition + 1);
  llvm::errs()
    << "          if (type_" << vecName << "->isIntegerTy())\n"
    << "            data_" << vecName << " = Builder2.CreatePtrToInt(data_" << vecName << ",\n"
    << "                                           type_" << vecName << ");\n";
    }
    argPosition += inputType->getValueAsInt("nelem");
  }

  llvm::errs() << "        }\n";
}

void emit_caching(Record *pattern, std::vector<size_t> actArgs,
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers, raw_ostream &os) {

  // 1. No caching for fwd-mode
  // 2. Deactivate caching for uncacheable_args
  // 3. Only caching if we do need the primary for an active gradient.
  llvm::errs() << "SmallVector<Type *, 2> cacheTypes;\n\n";

  emit_count_caching(pattern, actArgs, os);
  // emit_fp_caching(pattern, actArgs, os);
  emit_vinc_caching(pattern, actArgs, argUsers, os);

  DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  for (auto actEn : llvm::enumerate(actArgs)) {
    auto name = argOps->getArgNameStr(actEn.value());
    llvm::errs() << " if (" << name << "cache)\n"
                 << "   cacheTypes.push_back(castvals[" << actEn.index()
                 << "]);\n";
  }
  llvm::errs()
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

void findArgPositions(const std::vector<StringRef> toFind,
                      const DagInit *toSearch,
                      llvm::SmallSet<size_t, 5> &toInsert) {
  for (size_t i = 0; i < toSearch->getNumArgs(); i++) {
    if (DagInit *arg = dyn_cast<DagInit>(toSearch->getArg(i))) {
      // llvm::errs() << " Recursing. Magic!\n";
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
    llvm::errs() << " auto new_" << name
                 << " = lookup(gutils->getNewFromOriginal(arg_" << name
                 << "), Builder2),\n";
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
  llvm::errs() << "auto *called = call.getCalledFunction();\n";
  for (size_t i = 0; i < argOps->getNumArgs(); i++) {
    auto name = argOps->getArgNameStr(i);
    llvm::errs() << "auto arg_" << name << " = call.getArgOperand(" << i << ");\n";
    llvm::errs() << "auto type_" << name << " = arg_" << name << "->getType();\n";
    llvm::errs() << "bool uncacheable_" << name 
      << " = uncacheable_args.find(called->getArg("  << i << "))->second;\n\n";
  }
  for (auto act : actArgs) {
    auto name = argOps->getArgNameStr(act);
    llvm::errs() << "bool active_" << name << " = !gutils->isConstantValue(arg_"
                 << name << ");\n";
  }
}

void emitBlasDerivatives(const std::vector<Record *> &blasPatterns,
                         const std::vector<Record *> &blas_modes,
                         raw_ostream &os) {
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
    emit_castvals(pattern, posActArgs, os);
    emit_inttype(pattern, os);
    emit_helper(pattern, posActArgs, os);

    // new:
    emit_caching(pattern, posActArgs, argUsers, os);
    emit_extract_calls(pattern, posActArgs, argUsers, os);

    // emit_new_vars(pattern, posActArgs, os);
    // emit_blas_call(pattern, posActArgs, os);

    // emit_ending(pattern, os);
    // writeEnums(pattern, blas_modes, os);
    emit_free(pattern, posActArgs, os);
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
    emitDerivatives(records, os);
    return false;
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &EnzymeTableGenMain);
}

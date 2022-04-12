//===- enzyme-tblgen.cpp - Top-Level TableGen implementation for Enzyme -------===//
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

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

enum ActionType { GenDerivatives };

static cl::opt<ActionType>
    action(cl::desc("Action to perform:"),
           cl::values(clEnumValN(GenDerivatives, "gen-derivatives",
                                 "Generate instruction derivative")));

bool hasDiffeRet(Init * resultTree) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator()) ->getDef();
    if (opName == "DiffeRet" || Def->isSubClassOf("DiffeRet")) {
      return true;
    }
    for (auto zp : llvm::zip(resultRoot->getArgs(), resultRoot->getArgNames())) {
      if (hasDiffeRet(std::get<0>(zp)))
        return true;
    }
  }
  return false;
}

void handle(raw_ostream &os, Record *pattern, Init * resultTree, std::string builder, StringMap<std::string> &nameToOrdinal, bool lookup) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    llvm::errs() << " daginit\n";
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator()) ->getDef();
    llvm::errs() << " opname: " << opName << " - " << Def->getName() << " subc: " << Def->isSubClassOf("ConstantFP") << "\n";
    if (opName == "DiffeRet" || Def->isSubClassOf("DiffeRet")) {
      os << "dif";
      return;
    } else if (opName == "ConstantFP" || Def->isSubClassOf("ConstantFP")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op constant supported");
      
      auto *argument = resultRoot->getArg(0);

        auto value = dyn_cast<StringInit>(  Def->getValueInit("value"));
        if (!value)
          PrintFatalError(pattern->getLoc(), Twine("'value' not defined in ") +
                                                 resultTree->getAsString());


    os << " ({ Type* T = ";
      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto ord = nameToOrdinal.find(name);
        if (ord == nameToOrdinal.end())
          PrintFatalError(pattern->getLoc(),
                          Twine("unknown named operand '") + name + "'" + resultTree->getAsString());
        os << ord->getValue();
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in constantfp") + resultTree->getAsString());
      os << "->getType();\n";
      os << " Type* ET = T;\n";
      os << " if (gutils->getWidth() > 1) ET = cast<ArrayType>(T)->getElementType();\n";

      os << "Constant *res = ConstantFP::get(ET, \"" << value->getValue() << "\");\n";
      os << "if (gutils->getWidth() > 1) {\n";
      os << "  SmallVector<Constant*, 2> vals(gutils->getWidth(), res);\n";
      os << "  res = ConstantArray::get(cast<ArrayType>(T), vals);\n";
      os << "}\n";
      os << "res; })";
      return;
    } else if (opName == "Shadow" || Def->isSubClassOf("Shadow")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op constant supported");
      
        auto *argument = resultRoot->getArg(0);
        if (lookup) os << "lookup(";
        os << "gutils->invertPointerM(" << builder << ", ";

      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto ord = nameToOrdinal.find(name);
        if (ord == nameToOrdinal.end())
          PrintFatalError(pattern->getLoc(),
                          Twine("unknown named operand '") + name + "'" + resultTree->getAsString());
        os << ord->getValue();

      } else 
          PrintFatalError(pattern->getLoc(),
                          Twine("unknown named operand in shadow") + resultTree->getAsString());
      os << ")";
      if (lookup) os << ", " << builder << ")";

      return;
    }

    os << " ({\n";
    os << "    Value* args[" << resultRoot->getArgs().size() << "];\n";

    size_t idx = 0;
    for (auto zp : llvm::zip(resultRoot->getArgs(), resultRoot->getArgNames())) {
      os << " args[" << idx << "] = ";
      idx++;
      if (std::get<1>(zp)) {
        auto name = std::get<1>(zp)->getAsUnquotedString();
        auto ord = nameToOrdinal.find(name);
        if (ord == nameToOrdinal.end())
          PrintFatalError(pattern->getLoc(),
                          Twine("unknown named operand '") + name + "'" + resultTree->getAsString());
        if (lookup) os << "lookup(";
        os << "gutils->getNewFromOriginal(";
        os << ord->getValue();
        os << ")";
        if (lookup) os << ", " << builder << ")";
        os << " ;\n";
        continue;
      }
      handle(os, pattern, std::get<0>(zp), builder, nameToOrdinal, lookup);
      os << " ;\n";
    }

    os << " Value *res = nullptr;\n";
    os << " if (gutils->getWidth() == 1) res = " << builder << ".Create" << opName << "(";
    for (size_t i=0; i<idx; i++) {
      if (i > 0) os << ", ";
      os << "args[" << i << "]";
    }
    os << ");\n";
    os << " else {\n";
    os << " for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";
    os << "   Value *V = " << builder << ".Create" << opName << "(";
    for (size_t i=0; i<idx; i++) {
      if (i > 0) os << ", ";
      os << builder << ".CreateExtractValue(args[" << i << "], {idx})";
    }
    os << ");\n";
    os << "   if (res == nullptr) res = UndefValue::get(ArrayType::get(V->getType(), gutils->getWidth()));\n";
    os << "   res = " << builder << ".CreateInsertValue(res, V, {idx});\n";
    os << " }\n }\n";
    os << " res; })";
    return;
  }

  PrintFatalError(pattern->getLoc(),
                    Twine("unknown dag"));
}

static void emitDerivatives(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = recordKeeper.getAllDerivedDefinitions("CallPattern");
  Record *attrClass = recordKeeper.getClass("Attr");

  // Ensure unique patterns simply by appending unique suffix.
  unsigned rewritePatternCount = 0;
  std::string baseRewriteName = "GeneratedConvert";
  for (Record *pattern : patterns) {
    DagInit *tree = pattern->getValueAsDag("PatternToMatch");

    StringMap<std::string> nameToOrdinal;
    for (int i = 0, e = tree->getNumArgs(); i != e; ++i)
      nameToOrdinal[tree->getArgNameStr(i)] = "orig->getOperand(" + std::to_string(i) + ")";

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
      if (prev) os << " ||\n      ";
      os << "funcName == " << cast<StringInit>(nameI)->getAsString() << "";
      prev = true;
    }
    os << " ){\n";
    os << "    if (gutils->knownRecomputeHeuristic.find(orig) !=\n";
    os << "        gutils->knownRecomputeHeuristic.end()) {\n";
    os << "        if (!gutils->knownRecomputeHeuristic[orig]) {\n";
    os << "          gutils->cacheForReverse(BuilderZ, newCall,\n";
    os << "                                  getIndex(orig, CacheType::Self));\n";
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
      os << "        if (!gutils->isConstantValue(orig->getArgOperand(" << argIdx << "))) {\n";
      os << "          Value *dif = diffe(orig->getArgOperand(" << argIdx << "), Builder2);\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      os << "          Value *tmp = ";
      handle(os, pattern, resultTree, "Builder2", nameToOrdinal, /*lookup*/false);
      os << ";\n";

      os << "          if (res == nullptr) res = tmp;\n";
      os << "          else if (gutils->getWidth() == 1) res = Builder2.CreateFAdd(res, tmp);\n";
      os << "          else {\n";
      os << "            Value *out = UndefValue::get(res->getType());\n";
      os << "            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";
      os << "              Value *V = Builder2.CreateFAdd(Builder2.CreateExtractValue(res, {idx}), Builder2.CreateExtractValue(tmp, {idx}));\n";
      os << "              out = Builder2.CreateInsertValue(out, V, {idx});\n";
      os << "            }\n";
      os << "            res = out;\n";
      os << "          }\n";
      os << "        }\n";
    }

    os << "        setDiffe(orig, res, Builder2);\n";
    /*
    os << "          Value *x = gutils->getNewFromOriginal(orig->getArgOperand(0));\n";
    os << "          Value *onePx2 = Builder2.CreateFAdd(\n";
    os << "              ConstantFP::get(x->getType(), 1.0), Builder2.CreateFMul(x, x));\n";

    os << "          Value *op = diffe(orig->getArgOperand(0), Builder2);\n";

    os << "          auto rule = [&](Value *op) {\n";
    os << "            return Builder2.CreateFDiv(op, onePx2);\n";
    os << "          };\n";

    os << "          Value *dif0 = applyChainRule(call.getType(), Builder2, rule, op);\n";
    os << "          return;\n";
    */

  #if 0
    os << "        Value *dif = nullptr;\n";
    bool seen = false;
    for (auto argOpEn : llvm::enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();
      os << "        ";
      if (seen) os << "} else ";
      seen = true;
      os << "if (!dif && !gutils->isConstantValue(orig->getArgOperand(" << argIdx << "))) {\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      if (hasDiffeRet(resultTree))
        os << "          dif = diffe(orig, Builder2);\n";
    }
    if (seen) os << "        }\n";

    for (auto argOpEn : llvm::enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();
      os << "        if (!gutils->isConstantValue(orig->getArgOperand(" << argIdx << "))) {\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      os << "          addToDiffe(orig->getArgOperand(" << argIdx << "), ";
      handle(os, pattern, resultTree, "Builder2", nameToOrdinal, /*lookup*/true);
      os << ");\n";
      os << "        }\n";
    }

    os << "        auto rule = [&](";
    
    Value *op) {
            return Builder2.CreateFDiv(op, onePx2);
          };

          Value *dif0 = applyChainRule(call.getType(), Builder2, rule, op);
          setDiffe(orig, dif0, Builder2);
  #endif

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
      if (seen) os << "} else ";
      seen = true;
      os << "if (!dif && !gutils->isConstantValue(orig->getArgOperand(" << argIdx << "))) {\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      if (hasDiffeRet(resultTree))
        os << "          dif = diffe(orig, Builder2);\n";
    }
    if (seen) os << "        }\n";

    for (auto argOpEn : llvm::enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();
      os << "        if (!gutils->isConstantValue(orig->getArgOperand(" << argIdx << "))) {\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      os << "          addToDiffe(orig->getArgOperand(" << argIdx << "), ";
      handle(os, pattern, resultTree, "Builder2", nameToOrdinal, /*lookup*/true);
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

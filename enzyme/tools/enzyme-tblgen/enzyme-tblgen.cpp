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
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "datastructures.h"

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
    for (auto arg : resultRoot->getArgs()) {
      if (hasDiffeRet(arg))
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
      os << "  auto " << callval << " = call.getCalledOperand();\n";
      os << "#else\n";
      os << "  auto " << callval << " = call.getCalledValue();\n";
      os << "#endif\n";
      os << "  auto " << FT << " = call.getFunctionType();\n";
      os << "  auto " << cconv << " = call.getCallingConv();\n";
      return;
    }
    if (opName == "SameTypesFunc" || Def->isSubClassOf("SameTypesFunc")) {
      os << " auto " << FT << " = call.getFunctionType();\n";
      os << " auto " << callval
         << " = gutils->oldFunc->getParent()->getOrInsertFunction(";
      os << Def->getValueInit("name")->getAsString();
      os << ", " << FT << ", called->getAttributes())\n";
      os << "#if LLVM_VERSION_MAJOR >= 9\n";
      os << "  .getCallee()\n";
      os << "#endif\n";
      os << ";\n";
      os << "  auto " << cconv << " = call.getCallingConv();\n";
      return;
    }
  }
  assert(0 && "Unhandled function");
}
void getIntrinsic(raw_ostream &os, std::string callval, std::string FT,
                  std::string cconv, StringRef intrName, ListInit *typeInit,
                  StringMap<std::string> &nameToOrdinal) {
  os << " Type *tys[] = {";
  bool first = true;
  for (auto intrType : *typeInit) {
    auto arg = cast<DagInit>(intrType);
    assert(arg->getNumArgs() == 1 && "Only one arg allowed");
    auto name = arg->getArgNameStr(0);
    auto num = nameToOrdinal.lookup(name);
    os << ((first) ? "" : ", ") << num << "->getType()";
    first = false;
  }
  os << "};\n"
     << " Function *" << callval
     << " = Intrinsic::getDeclaration(called->getParent(), "
        "Intrinsic::"
     << intrName << ", tys);\n";
  os << "  auto " << FT << " = " << callval << "->getFunctionType();\n";
  os << "  auto " << cconv << " = call.getCallingConv();\n";
}

llvm::raw_ostream &operator<<(raw_ostream& os, StringMap<std::string> &C) {
    os << "{";
    bool first = true;
    for (auto &pair : C) {
        if (!first) os << ", ";
        os << pair.first() << ":" << pair.second;
        first = false;
    }
    return os << "}";
}

void initializeNames(raw_ostream &os, Init *resultTree) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
      for (size_t i=0; i<resultRoot->arg_size(); i++) {
        auto arg = resultRoot->getArg(i);
        auto name = resultRoot->getArgName(i);
      if (isa<UnsetInit>(arg) && name) {
        continue;
      }
      if (name) {
        auto namev = name->getAsUnquotedString();
        os << "llvm::Value *__tmp_" + namev << " = nullptr;\n";
      }
      initializeNames(os, arg);
    }
  } else if (ListInit *lst = dyn_cast<ListInit>(resultTree)) {
    for (auto elem : *lst)
        initializeNames(os, elem);
  }
}

// Returns whether value generated is a vector value or not.
bool handle(raw_ostream &os, Record *pattern, Init *resultTree,
            std::string builder, StringMap<std::string> &nameToOrdinal,
            bool lookup, std::vector<unsigned> retidx) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "DiffeRet" || Def->isSubClassOf("DiffeRet")) {
      if (retidx.size() == 0) {
        os << "dif";
      } else {
        os << "({  \n";
        os << "    Type* T = call.getType();\n";
        for (auto i : retidx) {
            os << "if (auto AT = dyn_cast<ArrayType>(T)) T = AT->getElementType();\n";
            os << "else if (auto AT = dyn_cast<VectorType>(T)) T = AT->getElementType();\n";
            os << "else if (auto AT = dyn_cast<StructType>(T)) T = AT->getElementType((unsigned)" << i << ");\n";
        }
            os << "Value *out = UndefValue::get(gutils->getShadowType(T));\n";

            os << "            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";

            os << " Value *prev = (gutils->getWidth() == 1) ? gutils->extractMeta(" << builder << ", dif, ArrayRef<unsigned>({";
      bool first = true;
      for (auto ind : retidx) {
        if (!first)
        os << ", ";
        os << ind;
        first = false;
      }
      os << "})) : gutils->extractMeta(" << builder << ", dif, ArrayRef<unsigned>({idx";
      for (auto ind : retidx) {
        os << ", ";
        os << ind;
        first = false;
      }
      os << "}));\n";
            os << "              out = (gutils->getWidth() > 1) ? Builder2.CreateInsertValue(out, prev, idx) : prev;\n";
            os << "            }\n";
            os << "            out; })\n";
      }
      return true;
    } else if (opName == "ConstantFP" || Def->isSubClassOf("ConstantFP")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op constant supported");

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
    } else if (opName == "Undef" || Def->isSubClassOf("Undef")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op constant supported");

      os << "UndefValue::get(";
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
      os << "->getType())";
      return false;
    } else if (opName == "Shadow" || Def->isSubClassOf("Shadow")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op constant supported");

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
          handle(os, pattern, std::get<0>(zp), builder, nameToOrdinal, lookup, retidx));
      os << " ;\n";
      if (std::get<1>(zp)) {
        auto name = std::get<1>(zp)->getAsUnquotedString();
        if (vectorValued.back())
        PrintFatalError(pattern->getLoc(), Twine("Cannot have vector valued saved arg '") +
                                                 name + "'" + std::get<0>(zp)->getAsString());
        oldMaps.try_emplace(name, nameToOrdinal[name]);
        nameToOrdinal[name] = "__tmp_" + name;
        os << " __tmp_" << name << " = args[" << (idx - 1) << "];\n";
      }

      anyVector |= vectorValued.back();
    }
    for (auto &pair : oldMaps) {
      if (pair.second.size())
        nameToOrdinal[pair.getKey()] = pair.second;
      // else
      //  nameToOrdinal.erase(pair.getKey());
    }

    if (Def->isSubClassOf("InsertValue"))
        opName = "InsertValue";
    if (Def->isSubClassOf("ExtractValue"))
        opName = "ExtractValue";

    bool isCall = opName == "Call" || Def->isSubClassOf("Call");
    bool isIntr = opName == "Intrinsic" || Def->isSubClassOf("Intrinsic");

    if (isCall) {
      getFunction(os, "callval", "FT", "cconv", Def->getValueInit("func"));
    } else if (isIntr) {
      auto intrName = Def->getValueAsString("name");
      auto intrTypes = Def->getValueAsListInit("types");
      getIntrinsic(os, "callval", "FT", "cconv", intrName, intrTypes,
                   nameToOrdinal);
    }

    os << " Value *res = nullptr;\n";

    if (anyVector) {
      os << " if (gutils->getWidth() == 1) { \n";
    }

    if (isCall || isIntr) {
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
    if (opName == "ExtractValue" || opName == "InsertValue") {
      os << ", ArrayRef<unsigned>({";
      bool first = true;
      for (auto *ind : *cast<ListInit>(Def->getValueAsListInit("indices"))) {
        if (!first)
        os << ", ";
        os << "(unsigned)(" << cast<IntInit>(ind)->getValue() << ")";
        first = false;
      }
      os << "})";
    }
    if (isCall || isIntr)
      os << "})";
    os << ")";
    if (isCall || isIntr)
      os << ")";
    os << ";\n";
    if (isCall) {
      os << " cubcall->setDebugLoc(gutils->getNewFromOriginal(call."
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
    } else if (isIntr) {
      os << " cubcall->setDebugLoc(gutils->getNewFromOriginal(call."
            "getDebugLoc()));\n";
      os << " cubcall->setCallingConv(cconv);\n";
      os << " res = cubcall;\n";
    }
    if (anyVector) {
      os << " } else {\n";
      os << " for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";

      if (isCall || isIntr) {
        os << " CallInst *V = cast<CallInst>(" << builder
           << ".CreateCall(FT, callval, ArrayRef<Value*>({";
      } else {
        os << "   Value *V = " << builder << ".Create" << opName << "(";
      }
      for (size_t i = 0; i < idx; i++) {
        if (i > 0)
          os << ", ";
        if (vectorValued[i])
          os << "gutils->extractMeta(" << builder << ", args[" << i << "], idx)";
        else
          os << "args[" << i << "]";
      }
        if (opName == "ExtractValue" || opName == "InsertValue") {
          os << ", ArrayRef<unsigned>({";
          bool first = true;
          for (auto *ind : *cast<ListInit>(Def->getValueAsListInit("indices"))) {
            if (!first)
            os << ", ";
            os << "(unsigned)(" << cast<IntInit>(ind)->getValue() << ")";
            first = false;
          }
          os << "})";
        }
      if (isCall || isIntr)
        os << "})";
      os << ")";
      if (isCall || isIntr) {
        os << ")";
      }
      os << ";\n";

      if (isCall) {
        os << "   "
              "V->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));"
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
      if (isIntr) {
        os << "   "
              "V->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));"
              "\n";
        os << "   V->setCallingConv(cconv);\n";
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

    if (tree->getNameStr().str().size())
      nameToOrdinal[tree->getNameStr().str()] = "&call";

    std::function<void(DagInit*, std::vector<unsigned>)>
        insert = [&](DagInit *ptree, std::vector<unsigned> prev) {
            unsigned i=0;
            for (auto tree : ptree->getArgs()) {
                std::vector<unsigned> next = prev;
                next.push_back(i);
                if (auto dg = dyn_cast<DagInit>(tree))
                    insert(dg, next);

                if (ptree->getArgNameStr(i).size()) {
                    auto op = "call.getOperand(" + std::to_string(next[0]) + ")";
                    if (prev.size() > 0) {
                      op = "gutils->extractMeta(Builder2, " + op + ", ArrayRef<unsigned>({";
                      bool first = true;
                      for (unsigned i=1; i<next.size(); i++) {
                        if (!first) op += ", ";
                        op += std::to_string(next[i]);
                      }
                      op += "}))";
                    }
                    nameToOrdinal[ptree->getArgNameStr(i)] = op;
                }
                i++;
            }
    };

    insert(tree, {});

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
    os << "    if (gutils->knownRecomputeHeuristic.find(&call) !=\n";
    os << "        gutils->knownRecomputeHeuristic.end()) {\n";
    os << "        if (!gutils->knownRecomputeHeuristic[&call]) {\n";
    os << "          gutils->cacheForReverse(BuilderZ, newCall,\n";
    os << "                                  getIndex(&call, "
          "CacheType::Self));\n";
    os << "        }\n";
    os << "    }\n";

    os << "    eraseIfUnused(call);\n";
    os << "    if (gutils->isConstantInstruction(&call))\n";
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
      os << "        if (!gutils->isConstantValue(call.getArgOperand("
         << argIdx << "))) {\n";
      os << "          Value *dif = diffe(call.getArgOperand(" << argIdx
         << "), Builder2);\n";

      initializeNames(os, argOpEn.value());
      std::function<void(std::vector<unsigned>, Init*)> fwdres = [&](std::vector<unsigned> idx, Init* ival) {
        if (DagInit *resultTree = dyn_cast<DagInit>(ival)) {
            if ("ArrayRet" == resultTree->getOperator()->getAsString()) {
                unsigned i=0;
                for (auto r : resultTree->getArgs()) {
                    std::vector<unsigned> next = idx;
                    next.push_back(i);
                    i++;
                    fwdres(next, r);
                }
                return;
            }
            os << "          {\n";
            os << "            Value *itmp = ";
            bool vectorValued = handle(os, pattern, resultTree, "Builder2",
                                 nameToOrdinal, /*lookup*/ false, {});
            os << ";\n";
            if (idx.size() == 0)
            os << "                Value *out = UndefValue::get(gutils->getShadowType(call.getType()));\n";
            else
            os << "                Value *out = res ? res : Constant::getNullValue(gutils->getShadowType(call.getType()));\n";
            os << "                for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";
            
            os << "                  Value *prev = res;\n";

            os << "                  if (prev) prev = gutils->getWidth() == 1 ? prev : gutils->extractMeta(Builder2, prev, idx);\n";
            if (idx.size() != 0) {
            os << "                  if (prev) prev = gutils->extractMeta(Builder2, prev, ArrayRef<unsigned>({";
            bool first = true;
            for (auto v : idx) {
                if (!first) os << ", ";
                first = true;
                os << v;
            }
            os << "}));\n";
            }
            
            os << "                  Value *next = itmp;\n";
           
            if (vectorValued)
            os << "                  if (gutils->getWidth() != 1) next = gutils->extractMeta(Builder2, next, idx);\n";
            os << "                  if (prev) next = Builder2.CreateFAdd(prev, next);\n";
            
            if (idx.size() == 0) {
            os << "                  out = (gutils->getWidth() == 1) ? next : Builder2.CreateInsertValue(out, next, {idx});\n";
            } else {
            os << "                  out = (gutils->getWidth() == 1) ? Builder2.CreateInsertValue(out, next, ArrayRef<unsigned>({";
            bool first = true;
            for (auto v : idx) {
                if (!first) os << ", ";
                first = false;
                os << v;
            }
            os << "})) : Builder2.CreateInsertValue(out, next, ArrayRef<unsigned>({idx,";
            for (auto v : idx) {
                os << v;
            }
            os << "}));\n";
            }
            os << "                }\n";
            os << "                res = out;\n";
            os << "           }\n";
        } else if (ListInit *lst = dyn_cast<ListInit>(ival)) {
                unsigned i=0;
                for (auto r : *lst) {
                    std::vector<unsigned> next = idx;
                    next.push_back(i);
                    i++;
                    fwdres(next, r);
                }
        } else 
          PrintFatalError(pattern->getLoc(), Twine("Unknown subinitialization"));
      };
      fwdres({}, argOpEn.value());
      os << "        }\n";
    }

    os << "        assert(res);\n";
    os << "        setDiffe(&call, res, Builder2);\n";

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
      os << "if (!dif && !gutils->isConstantValue(call.getArgOperand("
         << argIdx << "))) {\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      if (hasDiffeRet(resultTree)) {
        os << "          dif = diffe(&call, Builder2);\n";
        os << "          setDiffe(&call, "
              "Constant::getNullValue(gutils->getShadowType(call.getType())), "
              "Builder2);\n";
      }
    }
    if (seen)
      os << "        }\n";

    std::function<void(size_t, std::vector<unsigned>, Init*)> revres = [&](size_t argIdx, std::vector<unsigned> idx, Init* ival) {
      if (DagInit *resultTree = dyn_cast<DagInit>(ival)) {
        if ("ArrayRet" == resultTree->getOperator()->getAsString()) {
            unsigned i=0;
                for (auto r : resultTree->getArgs()) {
            auto next = idx;
            next.push_back(i);
                    revres(argIdx, next, r);
                    i++;
                }
                return;
        }
        os << "          {\n";
        os << "          Value *tmp = ";
        bool vectorValued = handle(os, pattern, resultTree, "Builder2",
                                 nameToOrdinal, /*lookup*/ true, idx);
        os << ";\n";

            os << "                Value *out = UndefValue::get(gutils->getShadowType(call.getArgOperand(" << argIdx << ")->getType()));\n";

            os << "            for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";

            os << "              Value *prev = toadd ? (gutils->getWidth() == 1 ? toadd : gutils->extractMeta(Builder2, toadd, idx)) : nullptr;\n";
            os << "              Value *next = tmp;\n";
            if (vectorValued)
            os << "              if (gutils->getWidth() > 1) next = gutils->extractMeta(Builder2, next, idx);\n";
            os << "              if (prev) next = Builder2.CreateFAdd(prev, next);\n";
            os << "              out = (gutils->getWidth() > 1) ? Builder2.CreateInsertValue(out, next, idx) : next;\n";
            os << "            }\n";
            os << "            toadd = out;\n";

        os << "          }\n";





      } else if (ListInit* lst = dyn_cast<ListInit>(ival)) {
        unsigned i=0;
        for (auto elem: *lst) {
            auto next = idx;
            next.push_back(i);
            revres(argIdx, next, elem);
            i++;
        }
      } else assert(0);
    };


    for (auto argOpEn : llvm::enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();

      os << "        if (!gutils->isConstantValue(call.getArgOperand("
         << argIdx << "))) {\n";
      initializeNames(os, argOpEn.value());
      os << ";\n";

      os << "          Value *toadd = nullptr;\n";
      revres(argIdx, {}, argOpEn.value());

      os << "          if (toadd) addToDiffe(call.getArgOperand(" << argIdx << "), toadd";
      os << ", Builder2, call.getArgOperand(" << argIdx << ")->getType());\n";
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

static void checkBlasCallsInDag(const RecordKeeper &RK,
                                const std::vector<Record *> blasPatterns,
                                const StringRef blasName,
                                const DagInit *toSearch) {

  // For nested FAdd, ... rules which don't directly call a blass fnc
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

void emit_handleBLAS(const std::vector<TGPattern> &blasPatterns,
                     raw_ostream &os) {
  std::string handledBlasFunctions = "";
  bool first = true;
  for (auto blasPattern : blasPatterns) {
    auto newName =
        Twine((first) ? "" : ", ") + "\"" + blasPattern.getName() + "\"";
    handledBlasFunctions.append(newName.str());
    first = false;
  }
  os << "struct BlasInfo {\n"
     << "  llvm::StringRef floatType;\n"
     << "  llvm::StringRef prefix;\n"
     << "  llvm::StringRef suffix;\n"
     << "  llvm::StringRef function;\n"
     << "};\n"
     << "\n"
     << "llvm::Optional<BlasInfo> extractBLAS(llvm::StringRef in) {\n"
     << "  llvm::Twine floatType[] = {\"s\", \"d\"}; // c, z\n"
     << "  llvm::Twine extractable[] = {" << handledBlasFunctions << "};\n"
     << "  llvm::Twine prefixes[] = {\"\", \"cblas_\", \"cublas_\"};\n"
     << "  llvm::Twine suffixes[] = {\"\", \"_\", \"_64_\"};\n"
     << "  for (auto t : floatType) {\n"
     << "    for (auto f : extractable) {\n"
     << "      for (auto p : prefixes) {\n"
     << "        for (auto s : suffixes) {\n"
     << "          if (in == (p + t + f + s).str()) {\n"
     << "            return llvm::Optional<BlasInfo>(BlasInfo{\n"
     << "                t.getSingleStringRef(),\n"
     << "                p.getSingleStringRef(),\n"
     << "                s.getSingleStringRef(),\n"
     << "                f.getSingleStringRef(),\n"
     << "            });\n"
     << "          }\n"
     << "        }\n"
     << "      }\n"
     << "    }\n"
     << "  }\n"
     << "  return llvm::NoneType();\n"
     << "}\n"
     << "\n"
     << "bool handleBLAS(llvm::CallInst &call, llvm::Function *called,"
        "BlasInfo blas,const std::vector<bool> &overwritten_args) {         \n"
     << "  using llvm::Type;                                                \n"
     << "  if(overwritten_args.size() != called->arg_size()) {              \n"
     << "       llvm::errs() << overwritten_args.size() << \" \" << "
        "called->arg_size() << \"\\n\";\n"
     << "       assert(overwritten_args.size() == called->arg_size());      \n"
     << "  }                                                                \n"
     << "  bool result = true;                                              \n"
     << "  std::map<llvm::Argument *, bool> uncacheable_args;               \n"
     << "  for (size_t i = 0; i < called->arg_size(); i++) {                \n"
     << "       bool overwritten = overwritten_args[i];                     \n"
     << "       llvm::Argument *arg = called->getArg(i);                    \n"
     << "       auto entry = std::pair<llvm::Argument *, "
        "bool>(arg, overwritten);                         \n"
     << "       uncacheable_args.insert(entry);                             \n"
     << "  }                                                                \n"
     << "  if (!gutils->isConstantInstruction(&call)) {                     \n"
     << "    Type *fpType;                                                  \n"
     << "    if (blas.floatType == \"d\") {                                 \n"
     << "      fpType = Type::getDoubleTy(call.getContext());               \n"
     << "    } else if (blas.floatType == \"s\") {                          \n"
     << "      fpType = Type::getFloatTy(call.getContext());                \n"
     << "    } else {                                                       \n"
     << "      assert(false && \"Unreachable\");                            \n"
     << "    }                                                              \n";
  first = true;
  for (auto pattern : blasPatterns) {
    auto name = pattern.getName();
    os << "    " << ((first) ? "" : "} else ") << " if (blas.function == \""
       << name << "\") {                           \n"
       << "      result = handle_" << name
       << "(blas, call, called, uncacheable_args, fpType);                    "
          "\n";
    first = false;
  }
  os << "    } else {                                                       \n"
     << "      llvm::errs() << \" fallback?\\n\";                           \n"
     << "      return false;                                                \n"
     << "    }                                                              \n"
     << "  }                                                                \n"
     << "                                                                   \n"
     << "  if (Mode == DerivativeMode::ReverseModeGradient) {               \n"
     << "    eraseIfUnused(call, /*erase*/ true, /*check*/ false);          \n"
     << "  } else {                                                         \n"
     << "    eraseIfUnused(call);                                           \n"
     << "  }                                                                \n"
     << "                                                                   \n"
     << "  return result;                                                   \n"
     << "}                                                                  \n";
}

void emit_beginning(TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  os << "\nbool handle_" << name
     << "(BlasInfo blas, llvm::CallInst &call, llvm::Function *called,\n"
     << "    const std::map<llvm::Argument *, bool> &uncacheable_args, "
        "llvm::Type *fpType) "
        "{\n"
     << "  \n"
     << "  using namespace llvm;\n"
     << "  CallInst *const newCall = "
        "cast<CallInst>(gutils->getNewFromOriginal(&call));\n"
     << "  IRBuilder<> BuilderZ(newCall);\n"
     << "  BuilderZ.setFastMathFlags(getFast());\n"
     << "  IRBuilder<> allocationBuilder(gutils->inversionAllocs);\n"
     << "  allocationBuilder.setFastMathFlags(getFast());\n";
  // not yet needed for lv-1
  //<< "  auto &DL = gutils->oldFunc->getParent()->getDataLayout();\n";
}

void emit_free_and_ending(TGPattern &pattern, raw_ostream &os) {
  os << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
     << "      Mode == DerivativeMode::ReverseModeGradient ||\n"
     << "      Mode == DerivativeMode::ForwardModeSplit) {\n"
     << "    if (shouldFree()) {\n";

  auto nameVec = pattern.getArgNames();
  auto typeMap = pattern.getArgTypeMap();
  for (size_t i = 0; i < nameVec.size(); i++) {
    if (typeMap.lookup(i) == argType::vincData) {
      auto name = nameVec[i];
      os << "      if (cache_" << name << ") {\n"
         << "        CreateDealloc(Builder2, data_ptr_" << name << ");\n"
         << "      }\n";
    }
  }
  os << "    }\n"
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

void emit_helper(TGPattern &pattern, raw_ostream &os) {
  std::vector<size_t> fp_pos{};
  auto nameVec = pattern.getArgNames();
  assert(nameVec.size() > 0);
  auto argTypeMap = pattern.getArgTypeMap();

  auto actArgs = pattern.getActiveArgs();
  os << "  auto calledArg = called->arg_begin();\n\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto name = nameVec[i];
    os << "  auto arg_" << name << " = call.getArgOperand(" << i << ");\n"
       << "  auto type_" << name << " = arg_" << name << "->getType();\n"
       << "  bool uncacheable_" << name
       << " = uncacheable_args.find(calledArg)->second;\n"
       << "  calledArg++;\n";
    if (std::count(actArgs.begin(), actArgs.end(), i)) {
      os << "  bool active_" << name << " = !gutils->isConstantValue(arg_"
         << name << ");\n";
    }
    os << "\n";
  }

  os << "  int num_active_fp = 0;\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    argType type = argTypeMap.lookup(i);
    if (type == argType::fp) {
      os << "  if (active_" << nameVec[i] << ")\n"
         << "    num_active_fp++;\n";
    }
  }

  for (auto name : llvm::enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto type = argTypeMap.lookup(name.index());
    if (type == argType::vincData) {
      os << "  bool julia_decl = !type_" << name.value()
         << "->isPointerTy();\n";
      return;
    }
  }

  PrintFatalError("Blas function without vector?");
}

void emit_castvals(TGPattern &pattern, raw_ostream &os) {
  auto activeArgs = pattern.getActiveArgs();
  auto nameVec = pattern.getArgNames();
  os << "  /* beginning castvalls */\n"
     << "  Type *castvals[" << activeArgs.size() << "];\n";

  for (size_t i = 0; i < activeArgs.size(); i++) {
    size_t argIdx = activeArgs[i];
    auto name = nameVec[argIdx];
    os << "  if (auto PT = dyn_cast<PointerType>(type_" << name << "))\n"
       << "    castvals[" << i << "] = PT;\n"
       << "  else\n"
       << "    castvals[" << i << "] = PointerType::getUnqual(fpType);\n";
  }
  os << "  Value *cacheval;\n\n"
     << "  /* ending castvalls */\n";
}

void emit_scalar_types(TGPattern &pattern, raw_ostream &os) {
  // We only look at the type of the first integer showing up.
  // This allows to learn if we use Fortran abi (byRef) or cabi
  std::string name = "";
  bool foundInt = false;

  auto inputTypes = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();

  for (auto val : inputTypes) {
    if (val.second == argType::len) {
      foundInt = true;
      name = nameVec[val.first];
      break;
    }
  }
  assert(foundInt && "no int type found in blas call");

  os << "  // fpType already given by blas type (s, d, c, z) \n"
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

#include "caching.h"

void emit_extract_calls(TGPattern &pattern, raw_ostream &os) {
  auto actArgs = pattern.getActiveArgs();
  auto typeMap = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();
  auto argUsers = pattern.getArgUsers();

  // TODO: adjust count / getArgOperand(0) based on first int?
  os << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
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

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto typeOfArg = typeMap.lookup(i);
    auto name = nameVec[i];
    if (typeOfArg == argType::vincInc) {
      os << "      if (cache_" << name << ") {\n"
         << "        true_" << name << " =\n"
         << "            (cacheTypes.size() == 1)\n"
         << "                ? cacheval\n"
         << "                : Builder2.CreateExtractValue(cacheval, "
            "{cacheidx});\n"
         << "        auto alloc = allocationBuilder.CreateAlloca(intType);\n"
         << "        Builder2.CreateStore(true_" << name << ", alloc);\n"
         << "        true_" << name << " = Builder2.CreatePointerCast(\n"
         << "            alloc, call.getArgOperand(0)->getType());\n"
         << "        " << name << " = true_" << name << ";\n"
         << "        cacheidx++;\n"
         << "      } else if (need_" << name << ") {\n"
         << "        if (Mode != DerivativeMode::ForwardModeSplit) {\n"
         << "          true_" << name << " = lookup(true_" << name
         << ", Builder2);\n"
         << "          " << name << " = true_" << name << ";\n"
         << "        }\n"
         << "      }\n"
         << "\n";
    } else if (typeOfArg == argType::len) {
      os << "      if (cache_" << name << ") {\n"
         << "        len_" << name << " = (cacheTypes.size() == 1)\n"
         << "                    ? cacheval\n"
         << "                    : Builder2.CreateExtractValue(cacheval, "
            "{cacheidx});\n"
         << "        auto alloc = allocationBuilder.CreateAlloca(intType);\n"
         << "        Builder2.CreateStore(len_" << name << ", alloc);\n"
         << "        len_" << name << " = Builder2.CreatePointerCast(\n"
         << "            alloc, call.getArgOperand(0)->getType());\n"
         << "        cacheidx++;\n"
         << "      } else {\n"
         << "        if (Mode != DerivativeMode::ForwardModeSplit)\n"
         << "          len_" << name << " = lookup(len_" << name
         << ", Builder2);\n"
         << "      }\n"
         << "\n";
    }
  }

  os << "    } else if (Mode != DerivativeMode::ForwardModeSplit) {\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto typeOfArg = typeMap.lookup(i);
    auto name = nameVec[i];
    if (typeOfArg == argType::vincInc) {
      os << "      if (cache_" << name << ") {\n"
         << "        true_" << name << " = lookup(true_" << name
         << ", Builder2);\n"
         << "        " << name << " = true_" << name << ";\n"
         << "      }\n";
    } else if (typeOfArg == argType::len) {
      os << "      len_" << name << " = lookup(len_" << name << ", Builder2);\n"
         << "\n";
    }
  }
  os << "    }\n";

  for (size_t i = 0; i < nameVec.size(); i++) {
    if (typeMap.lookup(i) != argType::vincData)
      continue;

    auto vecName = nameVec[i];
    auto vecPosition = i;
    auto vecUsers = argUsers.lookup(vecPosition);
    auto incName = nameVec[i + 1];
    os << "    if (cache_" << vecName << ") {\n"
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
      os << "   else if (";
      bool first = true;
      // TODO: verify x isn't user from data_x (as only adjoint of x will be
      // used)
      for (auto user : vecUsers) {
        auto name = nameVec[user];
        if (vecName == name)
          continue; // see above
        os << ((first) ? "" : " || ") << "active_" << name;
        first = false;
      }
      os << ") {\n"
         << "      data_" << vecName
         << " = lookup(gutils->getNewFromOriginal(arg_" << vecName
         << "), Builder2);\n"
         << "    }\n";
    }
  }
  os << "  } else {\n"
     << "\n";

  for (size_t i = 0; i < nameVec.size(); i++) {
    if (typeMap.lookup(i) != vincData)
      continue;
    auto vecName = nameVec[i];
    os << "    if (type_" << vecName << "->isIntegerTy())\n"
       << "      data_" << vecName << " = Builder2.CreatePtrToInt(data_"
       << vecName << ", type_" << vecName << ");\n";
  }

  os << "  }\n";
}

llvm::SmallString<80> ValueType_helper(TGPattern &pattern, size_t actPos) {
  auto nameVec = pattern.getArgNames();
  auto typeMap = pattern.getArgTypeMap();
  llvm::SmallString<80> valueTypes{};

  for (size_t pos = 0; pos < nameVec.size();) {
    auto name = nameVec[pos];
    auto type = typeMap.lookup(pos);

    if (pos > 0) {
      valueTypes.append(", ");
    }

    if (type == argType::len) {
      valueTypes.append("ValueType::None");
    } else if (type == argType::fp) {
      auto floatName = nameVec[pos];
      if (pos == actPos) {
        valueTypes.append("ValueType::Shadow");
      } else {
        valueTypes.append((Twine("cache_") + floatName +
                           " ? ValueType::None : ValueType::Primal")
                              .str());
      }
    } else if (type == argType::vincData) {
      auto nextName = nameVec[pos + 1];
      auto nextType = typeMap.lookup(pos + 1);
      assert(nextType == argType::vincInc);
      auto vecName = nameVec[pos];
      if (pos == actPos) {
        valueTypes.append("ValueType::Shadow, ValueType::None");
      } else {
        valueTypes.append(
            (Twine("cache_") + vecName +
             " ? ValueType::None : ValueType::Primal, ValueType::None")
                .str());
      }
      pos++; // extra inc, since vector cover two args
    } else {
      llvm::errs() << "type: " << type << "\n";
      PrintFatalError("Unhandled type!");
    }
    pos++;
  }
  return valueTypes;
}

// TODO: think about how to handle nested rules which aren't simple calling
// another BLAS fnc.

size_t pattern_call_args(TGPattern &pattern, size_t actArg,
                         llvm::SmallString<40> &result) {
  auto nameVec = pattern.getArgNames();
  auto nameMap = pattern.getArgNameMap();
  auto typeMap = pattern.getArgTypeMap();

  // just replace argOps with rule
  for (size_t pos = 0; pos < nameVec.size();) {
    if (pos > 0) {
      result.append(", ");
    }

    auto name = nameVec[pos];
    // get the position of the argument in the primary blas call
    assert(typeMap.count(pos) == 1);
    // and based on that get the fp/int + scalar/vector type
    auto typeOfArg = typeMap.lookup(pos);
    if (typeOfArg == argType::len) {
      auto out = (Twine("len_") + name).str();
      result.append((Twine("len_") + name).str());
    } else if (typeOfArg == argType::fp) {
      if (pos == actArg) {
        result.append((Twine("d_") + name).str());
      } else {
        result.append((Twine("fp_") + name).str());
      }
    } else if (typeOfArg == argType::vincData) {
      auto nextName = nameVec[pos + 1];
      // get the position of the argument in the primary blas call
      auto nextArgPosition = nameMap.lookup(nextName);
      // and based on that get the fp/int + scalar/vector type
      auto typeOfNextArg = typeMap.lookup(nextArgPosition);
      assert(typeOfNextArg == argType::vincInc);
      if (pos == actArg) {
        result.append((Twine("d_") + name + ", true_" + nextName).str());
      } else {
        result.append((Twine("data_") + name + ", " + nextName).str());
      }
      pos++; // extra ++ due to also handling vincInc
    } else if (typeOfArg == argType::vincInc) {
      // might come without vincData, e.g. after DiffeRet
      result.append(name);
    } else {
      llvm::errs() << "name: " << name << " typename: " << typeOfArg << "\n";
      llvm_unreachable("unimplemented input type!");
    }
    pos++;
  }

  return nameVec.size();
}

void emit_fwd_rewrite_rules(TGPattern &pattern, raw_ostream &os) {
  auto rules = pattern.getRules();
  os << "  /* fwd-rewrite */                                 \n"
     << "  if (Mode == DerivativeMode::ForwardMode ||        \n"
     << "      Mode == DerivativeMode::ForwardModeSplit) {   \n"
     << "                                                    \n"
     << "#if LLVM_VERSION_MAJOR >= 11                        \n"
     << "    auto callval = call.getCalledOperand();         \n"
     << "#else                                               \n"
     << "    auto callval = call.getCalledValue();           \n"
     << "#endif                                            \n\n";

  auto nameVec = pattern.getArgNames();
  auto inputTypes = pattern.getArgTypeMap();
  auto activeArgs = pattern.getActiveArgs();
  for (auto inputType : inputTypes) {
    if (inputType.second == argType::vincData) {
      auto name = nameVec[inputType.first];
      os << "    Value *d_" << name << " = active_" << name << "\n"
         << "     ? gutils->invertPointerM(arg_" << name << ", Builder2)\n"
         << "     : nullptr;\n";
    }
    if (inputType.second == argType::fp) {
      auto name = nameVec[inputType.first];
      os
          // Done: revert Undef to ConstantFP
          //<< "    Value *d_" << name << " = UndefValue::get(fpType);\n";
          << "    Value *d_" << name
          << " = llvm::ConstantFP::get(fpType, 0.0);\n";
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
    auto activeArg = activeArgs[i];
    auto rule = rules[i];
    auto actName = nameVec[activeArg];
    auto dcallArgs = llvm::SmallString<40>();
    size_t numArgs = pattern_call_args(pattern, activeArg, dcallArgs);
    auto valueTypes = ValueType_helper(pattern, activeArg);
    os << "      if(active_" << actName << ") {\n"
       << "        Value *args1[" << numArgs << "] = {" << dcallArgs << "};\n\n"
       << "        auto Defs = gutils->getInvertedBundles(\n"
       << "          &call, {" << valueTypes
       << "}, Builder2, /* lookup */ false);\n";
    if (i == 0) {
      os << "#if LLVM_VERSION_MAJOR > 7\n"
         << "          dres = Builder2.CreateCall(call.getFunctionType(), "
            "callval, args1, Defs);\n"
         << "#else\n"
         << "          dres = Builder2.CreateCall(callval, args1, Defs);\n"
         << "#endif\n";
    } else {
      os << "#if LLVM_VERSION_MAJOR > 7\n"
         << "        Value *nextCall = Builder2.CreateCall(\n"
         << "          call.getFunctionType(), callval, args1, Defs);\n"
         << "#else\n"
         << "        Value *nextCall = Builder2.CreateCall(callval, args1, "
            "Defs);\n"
         << "#endif\n"
         << "        if (dres)\n"
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
  os << ");\n"
     << "    setDiffe(&call, dres, Builder2);\n"
     << "  }\n";
}

void emit_deriv_fnc(StringMap<TGPattern> &patternMap, Rule &rule,
                    llvm::StringSet<> &handled, raw_ostream &os) {
  auto ruleDag = rule.getRuleDag();
  auto typeMap = rule.getArgTypeMap();
  auto opName = ruleDag->getOperator()->getAsString();
  auto nameMap = rule.getArgNameMap();
  auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
  if (Def->isSubClassOf("b")) {
    auto dfnc_name = Def->getValueAsString("s");
    if (patternMap.find(dfnc_name.str()) == patternMap.end()) {
      PrintFatalError("calling unknown Blas function");
    }
    TGPattern calledPattern = patternMap.find(dfnc_name.str())->getValue();
    DenseSet<size_t> mutableArgs = calledPattern.getMutableArgs();

    if (handled.find(dfnc_name) != handled.end())
      return;
    else
      handled.insert(dfnc_name);

    auto retTy = "Builder2.getVoidTy()";
    // TODO: add this to .td file and generate it based on that
    if (dfnc_name == "dot" || dfnc_name == "asum") {
      retTy = "fpType";
    }
    os << "    auto derivcall_" << dfnc_name
       << " = gutils->oldFunc->getParent()->getOrInsertFunction(\n"
       << "      (blas.prefix + blas.floatType + \"" << dfnc_name
       << "\" + blas.suffix).str(), " << retTy << ",\n";
    // insert arg types based on .td file
    bool first = true;
    std::vector<StringRef> usedArgStrs{};
    for (size_t i = 0; i < ruleDag->getNumArgs(); i++) {
      Init *subArg = ruleDag->getArg(i);
      if (DefInit *def = dyn_cast<DefInit>(subArg)) {
        auto Def = def->getDef();
        usedArgStrs.push_back(""); // no need to process later
        std::string typeToAdd = "";
        if (Def->isSubClassOf("DiffeRet")) {
          typeToAdd = "byRef ? PointerType::getUnqual(call.getType()) : "
                      "call.getType()\n";
        } else if (Def->isSubClassOf("input")) {
          auto argStr = Def->getValueAsString("name");
          // assert(mutableArgs.count(i) == 1);
          //  primary and adj have the same type
          typeToAdd = (Twine("type_") + argStr).str();
          usedArgStrs.push_back((Twine("input_") + argStr).str());
        } else if (Def->isSubClassOf("adj")) {
          auto argStr = Def->getValueAsString("name");
          // primary and adj have the same type
          typeToAdd = (Twine("type_") + argStr).str();
          // assert(mutables.count(argStr) == 1);
          usedArgStrs.push_back((Twine("adj_") + argStr).str());
        } else if (Def->isSubClassOf("Constant")) {
          typeToAdd = "fpType";
        } else {
          PrintFatalError(Def->getLoc(), "PANIC! Unsupported Definit");
        }
        os << ((first) ? "" : ", ") << typeToAdd;
      } else {
        auto argStr = ruleDag->getArgNameStr(i);
        os << ((first) ? "" : ", ") << "type_" << argStr;
        usedArgStrs.push_back(argStr);
      }
      first = false;
    }
    os << ");\n";
    if (dfnc_name == "dot") {
      os << "    assert(derivcall_dot.getFunctionType()->getReturnType() == "
            "fpType);\n";
    }
    os << "#if LLVM_VERSION_MAJOR >= 9\n"
       << "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name
       << ".getCallee()))\n"
       << "#else\n"
       << "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name
       << "))\n"
       << "#endif\n"
       << "    {\n"
       << "      F->addFnAttr(Attribute::ArgMemOnly);\n"
       << "      if (byRef) {\n";
    for (size_t argPos = 0; argPos < usedArgStrs.size(); argPos++) {
      auto typeOfArg = typeMap.lookup(argPos);
      if (typeOfArg == argType::len || typeOfArg == argType::vincInc) {
        os << "        F->addParamAttr(" << argPos
           << ", Attribute::ReadOnly);\n"
           << "        F->addParamAttr(" << argPos
           << ", Attribute::NoCapture);\n";
      }
    }
    os << "      }\n"
       << "      // Julia declares double* pointers as Int64,\n"
       << "      //  so LLVM won't let us add these Attributes.\n"
       << "      if (!julia_decl) {\n";
    for (size_t argPos = 0; argPos < usedArgStrs.size(); argPos++) {
      auto typeOfArg = typeMap.lookup(argPos);
      if (typeOfArg == argType::vincData) {
        os << "        F->addParamAttr(" << argPos
           << ", Attribute::NoCapture);\n";
        if (mutableArgs.count(argPos) == 0) {
          // Only emit ReadOnly if the arg isn't mutable
          os << "        F->addParamAttr(" << argPos
             << ", Attribute::ReadOnly);\n";
        }
      }
    }
    os << "      }\n"
       << "    }\n\n";
  } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "noop") {
    // nothing to prepare
  } else if (Def->isSubClassOf("DiffeRet")) {
    // nothing to prepare
  } else if (Def->isSubClassOf("Inst")) {
    // TODO:
    PrintFatalError("Unhandled Inst Rule!");
  } else {
    PrintFatalError("Unhandled deriv Rule!");
  }
}

size_t rule_call_args(Rule &rule, size_t actArg,
                      llvm::SmallString<40> &result) {

  auto nameMap = rule.getArgNameMap();
  auto typeMap = rule.getArgTypeMap();
  auto ruleDag = rule.getRuleDag();
  size_t numArgs = ruleDag->getNumArgs();

  // just replace argOps with rule
  for (size_t pos = 0; pos < ruleDag->getNumArgs();) {
    if (pos > 0) {
      result.append(", ");
    }

    auto arg = ruleDag->getArg(pos);
    if (DefInit *DefArg = dyn_cast<DefInit>(arg)) {
      auto Def = DefArg->getDef();
      if (Def->isSubClassOf("DiffeRet")) {
        result.append("dif");
      } else if (Def->isSubClassOf("adj")) {
        auto name = Def->getValueAsString("name");
        result.append((Twine("d_") + name).str());
      } else if (Def->isSubClassOf("input")) {
        auto name = Def->getValueAsString("name");
        // maybe it should be data_ptr_ ??
        result.append((Twine("data_") + name).str());
        // result.append((Twine("input_") + name).str());
      } else if (Def->isSubClassOf("MagicInst")) {
        llvm::errs() << "MagicInst\n";
      } else if (Def->isSubClassOf("Constant")) {
        auto val = Def->getValueAsString("value");
        result.append((Twine("ConstantFP::get(fpType, ") + val + ")").str());
      } else {
        llvm::errs() << Def->getName() << "\n";
        PrintFatalError("Def that isn't a DiffeRet!");
      }
    } else {
      auto name = ruleDag->getArgNameStr(pos);
      // get the position of the argument in the primary blas call
      assert(nameMap.count(name) == 1);
      auto argPosition = nameMap.lookup(name);
      // and based on that get the fp/int + scalar/vector type
      auto typeOfArg = typeMap.lookup(argPosition);
      if (typeOfArg == argType::len) {
        auto out = (Twine("len_") + name).str();
        result.append((Twine("len_") + name).str());
      } else if (typeOfArg == argType::fp) {
        if (argPosition == actArg) {
          result.append((Twine("d_") + name).str());
        } else {
          result.append((Twine("fp_") + name).str());
        }
      } else if (typeOfArg == argType::vincData) {
        auto nextName = ruleDag->getArgNameStr(pos + 1);
        // get the position of the argument in the primary blas call
        auto nextArgPosition = nameMap.lookup(nextName);
        // and based on that get the fp/int + scalar/vector type
        auto typeOfNextArg = typeMap.lookup(nextArgPosition);
        assert(typeOfNextArg == argType::vincInc);
        if (argPosition == actArg) {
          result.append((Twine("d_") + name + ", true_" + nextName).str());
        } else {
          result.append((Twine("data_") + name + ", " + nextName).str());
        }
        pos++; // extra ++ due to also handling vincInc
      } else if (typeOfArg == argType::vincInc) {
        // might come without vincData, e.g. after DiffeRet
        result.append(name);
      } else {
        llvm::errs() << "name: " << name << " typename: " << typeOfArg << "\n";
        llvm_unreachable("unimplemented input type!");
      }
    }
    pos++;
  }

  return numArgs;
}

void emit_rev_rewrite_rules(StringMap<TGPattern> patternMap, TGPattern &pattern,
                            raw_ostream &os) {

  auto nameVec = pattern.getArgNames();
  auto typeMap = pattern.getArgTypeMap();
  auto rules = pattern.getRules();
  auto activeArgs = pattern.getActiveArgs();

  // If any of the rule uses DiffeRet, the primary function has a ret val
  // and we should emit the code for handling it.
  bool hasDiffeRetVal = false;
  for (auto derivOp : rules) {
    DagInit *resultRoot = derivOp.getRuleDag(); // correct
    for (size_t pos = 0; pos < resultRoot->getNumArgs(); pos++) {
      Init *arg = resultRoot->getArg(pos);
      if (DefInit *DefArg = dyn_cast<DefInit>(arg)) {
        auto Def = DefArg->getDef();
        if (Def->isSubClassOf("DiffeRet")) {
          hasDiffeRetVal = true;
        }
        // DagInit *dagArg = cast<DagInit>(arg);
        // llvm::errs() << "argName: " << dagArg->getName() << "\n";
        // hasDiffeRetVal |= hasDiffeRet(dagArg);
      }
    }
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "DiffeRet" || Def->isSubClassOf("DiffeRet")) {
      hasDiffeRetVal = true;
    }
    for (auto arg : resultRoot->getArgs()) {
      hasDiffeRetVal |= hasDiffeRet(arg);
    }
  }
  // llvm::errs() << "\n\n" << pattern.getName() << hasDiffeRetVal << "\n\n";

  os << "  /* rev-rewrite */                                 \n"
     << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
     << "      Mode == DerivativeMode::ReverseModeGradient) {\n"
     << "    Value *alloc = nullptr;\n"
     << "    if (byRef) {\n"
     << "      alloc = allocationBuilder.CreateAlloca(fpType);\n"
     << "    }\n\n";
  if (hasDiffeRetVal) {
    os << "    Value *dif = diffe(&call, Builder2);\n";
  }

  // TODO: adj_ args
  //  os
  //<< "    Value *adj_" << name << " =
  //lookup(gutils->invertPointerM(call.getArgOperand(arg_" << name << "),
  //Builder2))\n";

  llvm::StringSet handled{}; // We only emit one derivcall per blass call type
  for (auto rule : rules) {
    emit_deriv_fnc(patternMap, rule, handled, os);
  }

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto name = nameVec[i];
    auto typeOfArg = typeMap.lookup(i);
    if (typeOfArg == argType::vincData) {
      os << "    Value *d_" << name << " = active_" << name << "\n"
         << "     ? lookup(gutils->invertPointerM(arg_" << name
         << ", Builder2), Builder2)\n"
         << "     : nullptr;\n";
    } else if (typeOfArg == argType::fp) {
      os << "    Value *d_" << name << " = UndefValue::get(fpType);\n";
    }
  }

  os << "    applyChainRule(\n"
     << "      Builder2,\n"
     << "      [&](";
  bool first = true;
  for (auto arg : activeArgs) {
    auto name = nameVec[arg];
    auto typeOfArg = typeMap.lookup(arg);
    // We don't pass in shaddows of fp values,
    // we just create and struct-return the shaddows
    if (typeOfArg == argType::fp)
      continue;
    os << ((first) ? "" : ", ") << "Value *"
       << "d_" + name;
    first = false;
  }

  if (hasDiffeRetVal) {
    os << ((first) ? "" : ", ") << "Value *dif) {\n"
       << "        if (byRef) {\n"
       << "          Builder2.CreateStore(dif, alloc);\n"
       << "          dif = alloc;\n"
       << "        }\n"
       << "        unsigned int idx = 0;\n";
  } else {
    os << ") {\n"
       << "        unsigned int idx = 0;\n";
  }

  for (Rule rule : rules) {
    size_t actArg = rule.getHandledArgIdx();
    auto ruleDag = rule.getRuleDag();
    auto name = nameVec[actArg];
    auto nameMap = rule.getArgNameMap();
    auto typeOfArg = typeMap.lookup(actArg);
    auto args = llvm::SmallString<40>();
    size_t numArgs = rule_call_args(rule, actArg, args);
    auto valueTypes = ValueType_helper(pattern, actArg);
    auto opName = ruleDag->getOperator()->getAsString();
    auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
    if (Def->isSubClassOf("DiffeRet")) {
      os << "      if (active_" << name << ") {\n"
         << "        Value *toadd = dif;\n"
         << "        addToDiffe(arg_" << name << ", toadd, Builder2, type_"
         << name << ");\n"
         << "      }\n";
    } else if (Def->isSubClassOf("b")) {
      auto actCondition = "active_" + name;
      for (size_t pos = 0; pos < ruleDag->getNumArgs();) {
        auto arg = ruleDag->getArg(pos);
        if (DefInit *DefArg = dyn_cast<DefInit>(arg)) {
          auto Def = DefArg->getDef();
          if (Def->isSubClassOf("adj")) {
            auto name = Def->getValueAsString("name");
            actCondition.append((Twine(" && d_") + name).str());
          }
        }
        pos++;
      }
      auto dfnc_name = Def->getValueAsString("s");
      os << "      if (" << actCondition << ") {\n"
         << "        Value *args1[" << numArgs << "] = {" << args << "};\n"
         << "        auto Defs = gutils->getInvertedBundles(&call, {"
         << valueTypes << "}, Builder2, /* lookup */ true);\n";

      if (typeOfArg == argType::fp) {
        // extra handling, since we will update only a fp scalar as part of the
        // return struct it's presumably done by setting it to the value
        // returned by this call
        os << "        CallInst *cubcall = "
              "cast<CallInst>(Builder2.CreateCall(derivcall_"
           << dfnc_name << ", args1, Defs));\n"
           << "        addToDiffe(arg_" << name
           << ", cubcall, Builder2, fpType);\n"
           << "        idx++;\n"
           << "      }\n";
      } else {
        os << "        Builder2.CreateCall(derivcall_" << dfnc_name
           << ", args1, Defs);\n"
           << "      }\n";
      }
    } else if (Def->isSubClassOf("adj")) {
    } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "noop") {
    } else if (Def->isSubClassOf("Constant")) {
    } else {
      llvm::errs() << Def->getName() << "\n";
      PrintFatalError("Unhandled blas-rev case!");
    }
  }
  os << "    },\n"
     << "    ";

  first = true;
  for (auto arg : activeArgs) {
    auto name = nameVec[arg];
    auto typeOfArg = typeMap.lookup(arg);
    // We don't pass in shaddows of fp values,
    // we just create and struct-return the shaddows
    if (typeOfArg == argType::fp)
      continue;
    os << ((first) ? "" : ", ") << "d_" + name;
    first = false;
  }
  if (hasDiffeRetVal) {
    os << ((first) ? "" : ", ") << "dif);\n"
       << "  setDiffe(\n"
       << "    &call,\n"
       << "    Constant::getNullValue(gutils->getShadowType(call.getType())),\n"
       << "    Builder2);\n"
       << "  }\n";
  } else {
    os << "  );\n"
       << "  }\n";
  }
}

// NEXT TODO: for input args (vectors) being overwritten.
// Cache them and use the cache later

/*
 * We create the following variables:
 */
void emitBlasDerivatives(const RecordKeeper &RK, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &blasPatterns = RK.getAllDerivedDefinitions("CallBlasPattern");

  std::vector<TGPattern> newBlasPatterns{};
  StringMap<TGPattern> patternMap;
  for (auto pattern : blasPatterns) {
    auto parsedPattern = TGPattern(*pattern);
    newBlasPatterns.push_back(TGPattern(*pattern));
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

  for (auto newPattern : newBlasPatterns) {

    llvm::errs() << "\nhandling: " + newPattern.getName() + "\n";

    emit_beginning(newPattern, os);
    emit_helper(newPattern, os);
    emit_castvals(newPattern, os);
    emit_scalar_types(newPattern, os);

    emit_caching(newPattern, os);
    emit_extract_calls(newPattern, os);

    emit_fwd_rewrite_rules(newPattern, os);
    emit_rev_rewrite_rules(patternMap, newPattern, os);

    //// writeEnums(pattern, blas_modes, os);
    emit_free_and_ending(newPattern, os);
  }
}

static bool EnzymeTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case GenDerivatives:
    emitDerivatives(records, os);
    return false;
  case GenBlasDerivatives:
    emitBlasDerivatives(records, os);
    return false;

  default:
    llvm::errs() << "unknown tablegen action!\n";
    llvm_unreachable("unknown tablegen action!");
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &EnzymeTableGenMain);
}

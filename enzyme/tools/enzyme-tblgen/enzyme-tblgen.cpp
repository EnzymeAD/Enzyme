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
     << "bool handleBLAS(llvm::CallInst &call, llvm::Function *called, BlasInfo "
        "blas,\n"
     << "                const std::map<llvm::Argument *, bool> &uncacheable_args) { "
        "\n"
     << "  using llvm::Type;"
     << "                                                                      "
        "\n"
     << "  bool result = true;                                                 "
        "\n"
     << "  if (!gutils->isConstantInstruction(&call)) {                        "
        "\n"
     << "    Type *fpType;                                             \n"
     << "    if (blas.floatType == \"d\") {                                    "
        "\n"
     << "      fpType = Type::getDoubleTy(call.getContext());               \n"
     << "    } else if (blas.floatType == \"s\") {                             "
        "\n"
     << "      fpType = Type::getFloatTy(call.getContext());                \n"
     << "    } else {                                                          "
        "\n"
     << "      assert(false && \"Unreachable\");                               "
        "\n"
     << "    }                                                                 "
        "\n";
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
  os << "    } else {                                                          "
        "\n"
     << "      llvm::errs() << \" fallback?\\n\";                              "
        "\n"
     << "      return false;                                                   "
        "\n"
     << "    }                                                                 "
        "\n"
     << "  }                                                                   "
        "\n"
     << "                                                                      "
        "\n"
     << "  if (Mode == DerivativeMode::ReverseModeGradient) {                  "
        "\n"
     << "    eraseIfUnused(call, /*erase*/ true, /*check*/ false);             "
        "\n"
     << "  } else {                                                            "
        "\n"
     << "    eraseIfUnused(call);                                              "
        "\n"
     << "  }                                                                   "
        "\n"
     << "                                                                      "
        "\n"
     << "  return result;                                                      "
        "\n"
     << "}                                                                     "
        "\n";
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

    // emit_beginning(newPattern, os);
    // emit_helper(newPattern, os);
    // emit_castvals(newPattern, os);
    // emit_scalar_types(newPattern, os);

    // emit_caching(newPattern, os);
    // emit_extract_calls(newPattern, os);

    // emit_fwd_rewrite_rules(newPattern, os);
    // emit_rev_rewrite_rules(patternMap, newPattern, os);

    //// writeEnums(pattern, blas_modes, os);
    // emit_free_and_ending(newPattern, os);
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

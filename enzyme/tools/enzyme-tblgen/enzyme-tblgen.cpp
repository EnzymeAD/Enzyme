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

#include "datastructures.h"

using namespace llvm;

enum ActionType {
  GenDerivatives,
  GenBlasDerivatives,
  UpdateBlasDecl,
  UpdateBlasTA,
  GenBlasDiffUse,
};

static cl::opt<ActionType>
    action(cl::desc("Action to perform:"),
           cl::values(clEnumValN(GenBlasDerivatives, "gen-blas-derivatives",
                                 "Generate BLAS derivatives")),
           cl::values(clEnumValN(UpdateBlasDecl, "update-blas-declarations",
                                 "Update BLAS declarations")),
           cl::values(clEnumValN(UpdateBlasTA, "gen-blas-typeanalysis",
                                 "Update BLAS TypeAnalysis")),
           cl::values(clEnumValN(GenBlasDiffUse, "gen-blas-diffuseanalysis",
                                 "Update BLAS DiffUseAnalysis")),
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

llvm::raw_ostream &operator<<(raw_ostream &os, StringMap<std::string> &C) {
  os << "{";
  bool first = true;
  for (auto &pair : C) {
    if (!first)
      os << ", ";
    os << pair.first() << ":" << pair.second;
    first = false;
  }
  return os << "}";
}

void initializeNames(raw_ostream &os, Init *resultTree) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    for (size_t i = 0; i < resultRoot->arg_size(); i++) {
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
          os << "if (auto AT = dyn_cast<ArrayType>(T)) T = "
                "AT->getElementType();\n";
          os << "else if (auto AT = dyn_cast<VectorType>(T)) T = "
                "AT->getElementType();\n";
          os << "else if (auto AT = dyn_cast<StructType>(T)) T = "
                "AT->getElementType((unsigned)"
             << i << ");\n";
        }
        os << "Value *out = UndefValue::get(gutils->getShadowType(T));\n";

        os << "            for(unsigned int idx=0, W=gutils->getWidth(); "
              "idx<W; idx++) {\n";

        os << " Value *prev = (gutils->getWidth() == 1) ? gutils->extractMeta("
           << builder << ", dif, ArrayRef<unsigned>({";
        bool first = true;
        for (auto ind : retidx) {
          if (!first)
            os << ", ";
          os << ind;
          first = false;
        }
        os << "})) : gutils->extractMeta(" << builder
           << ", dif, ArrayRef<unsigned>({idx";
        for (auto ind : retidx) {
          os << ", ";
          os << ind;
          first = false;
        }
        os << "}));\n";
        os << "              out = (gutils->getWidth() > 1) ? "
              "Builder2.CreateInsertValue(out, prev, idx) : prev;\n";
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
      vectorValued.push_back(handle(os, pattern, std::get<0>(zp), builder,
                                    nameToOrdinal, lookup, retidx));
      os << " ;\n";
      if (std::get<1>(zp)) {
        auto name = std::get<1>(zp)->getAsUnquotedString();
        if (vectorValued.back())
          PrintFatalError(pattern->getLoc(),
                          Twine("Cannot have vector valued saved arg '") +
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
        auto attrName = attrDef->getValueInit("name")->getAsUnquotedString();
#if LLVM_VERSION_MAJOR >= 16
        if (attrName == "ReadNone") {
            os << " cubcall->setOnlyReadsMemory();\n";
            os << " cubcall->setOnlyWritesMemory();\n";
            continue;
        }
        if (attrName == "ReadOnly") {
            os << " cubcall->setOnlyReadsMemory();\n";
            continue;
        }
#endif

        os << "#if LLVM_VERSION_MAJOR >= 14\n"
           << " cubcall->addAttributeAtIndex(AttributeList::FunctionIndex, "
           << "Attribute::"
           << attrName << ");\n";
        os << "#else\n"
           << " cubcall->addAttribute(AttributeList::FunctionIndex, "
           << "Attribute::"
           << attrName << ");\n";
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
          os << "gutils->extractMeta(" << builder << ", args[" << i
             << "], idx)";
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

  for (Record *pattern : patterns) {
    DagInit *tree = pattern->getValueAsDag("PatternToMatch");

    StringMap<std::string> nameToOrdinal;

    if (tree->getNameStr().str().size())
      nameToOrdinal[tree->getNameStr().str()] = "&call";

    std::function<void(DagInit *, std::vector<unsigned>)> insert =
        [&](DagInit *ptree, std::vector<unsigned> prev) {
          unsigned i = 0;
          for (auto tree : ptree->getArgs()) {
            std::vector<unsigned> next = prev;
            next.push_back(i);
            if (auto dg = dyn_cast<DagInit>(tree))
              insert(dg, next);

            if (ptree->getArgNameStr(i).size()) {
              auto op = "call.getOperand(" + std::to_string(next[0]) + ")";
              if (prev.size() > 0) {
                op = "gutils->extractMeta(Builder2, " + op +
                     ", ArrayRef<unsigned>({";
                bool first = true;
                for (unsigned i = 1; i < next.size(); i++) {
                  if (!first)
                    op += ", ";
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
      os << "        if (!gutils->isConstantValue(call.getArgOperand(" << argIdx
         << "))) {\n";
      os << "          Value *dif = diffe(call.getArgOperand(" << argIdx
         << "), Builder2);\n";

      initializeNames(os, argOpEn.value());
      std::function<void(std::vector<unsigned>, Init *)> fwdres =
          [&](std::vector<unsigned> idx, Init *ival) {
            if (DagInit *resultTree = dyn_cast<DagInit>(ival)) {
              if ("ArrayRet" == resultTree->getOperator()->getAsString()) {
                unsigned i = 0;
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
                os << "                Value *out = "
                      "UndefValue::get(gutils->getShadowType(call.getType()));"
                      "\n";
              else
                os << "                Value *out = res ? res : "
                      "Constant::getNullValue(gutils->getShadowType(call."
                      "getType()));\n";
              os << "                for(unsigned int idx=0, "
                    "W=gutils->getWidth(); idx<W; idx++) {\n";

              os << "                  Value *prev = res;\n";

              os << "                  if (prev) prev = gutils->getWidth() == "
                    "1 ? prev : gutils->extractMeta(Builder2, prev, idx);\n";
              if (idx.size() != 0) {
                os << "                  if (prev) prev = "
                      "gutils->extractMeta(Builder2, prev, "
                      "ArrayRef<unsigned>({";
                bool first = true;
                for (auto v : idx) {
                  if (!first)
                    os << ", ";
                  first = true;
                  os << v;
                }
                os << "}));\n";
              }

              os << "                  Value *next = itmp;\n";

              if (vectorValued)
                os << "                  if (gutils->getWidth() != 1) next = "
                      "gutils->extractMeta(Builder2, next, idx);\n";
              os << "                  if (prev) next = "
                    "Builder2.CreateFAdd(prev, next);\n";

              if (idx.size() == 0) {
                os << "                  out = (gutils->getWidth() == 1) ? "
                      "next : Builder2.CreateInsertValue(out, next, {idx});\n";
              } else {
                os << "                  out = (gutils->getWidth() == 1) ? "
                      "Builder2.CreateInsertValue(out, next, "
                      "ArrayRef<unsigned>({";
                bool first = true;
                for (auto v : idx) {
                  if (!first)
                    os << ", ";
                  first = false;
                  os << v;
                }
                os << "})) : Builder2.CreateInsertValue(out, next, "
                      "ArrayRef<unsigned>({idx,";
                for (auto v : idx) {
                  os << v;
                }
                os << "}));\n";
              }
              os << "                }\n";
              os << "                res = out;\n";
              os << "           }\n";
            } else if (ListInit *lst = dyn_cast<ListInit>(ival)) {
              unsigned i = 0;
              for (auto r : *lst) {
                std::vector<unsigned> next = idx;
                next.push_back(i);
                i++;
                fwdres(next, r);
              }
            } else
              PrintFatalError(pattern->getLoc(),
                              Twine("Unknown subinitialization"));
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
      os << "if (!dif && !gutils->isConstantValue(call.getArgOperand(" << argIdx
         << "))) {\n";
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

    std::function<void(size_t, std::vector<unsigned>, Init *)> revres =
        [&](size_t argIdx, std::vector<unsigned> idx, Init *ival) {
          if (DagInit *resultTree = dyn_cast<DagInit>(ival)) {
            if ("ArrayRet" == resultTree->getOperator()->getAsString()) {
              unsigned i = 0;
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

            os << "                Value *out = "
                  "UndefValue::get(gutils->getShadowType(call.getArgOperand("
               << argIdx << ")->getType()));\n";

            os << "            for(unsigned int idx=0, W=gutils->getWidth(); "
                  "idx<W; idx++) {\n";

            os << "              Value *prev = toadd ? (gutils->getWidth() == "
                  "1 ? toadd : gutils->extractMeta(Builder2, toadd, idx)) : "
                  "nullptr;\n";
            os << "              Value *next = tmp;\n";
            if (vectorValued)
              os << "              if (gutils->getWidth() > 1) next = "
                    "gutils->extractMeta(Builder2, next, idx);\n";
            os << "              if (prev) next = Builder2.CreateFAdd(prev, "
                  "next);\n";
            os << "              out = (gutils->getWidth() > 1) ? "
                  "Builder2.CreateInsertValue(out, next, idx) : next;\n";
            os << "            }\n";
            os << "            toadd = out;\n";

            os << "          }\n";

          } else if (ListInit *lst = dyn_cast<ListInit>(ival)) {
            unsigned i = 0;
            for (auto elem : *lst) {
              auto next = idx;
              next.push_back(i);
              revres(argIdx, next, elem);
              i++;
            }
          } else
            assert(0);
        };

    for (auto argOpEn : llvm::enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();

      os << "        if (!gutils->isConstantValue(call.getArgOperand(" << argIdx
         << "))) {\n";
      initializeNames(os, argOpEn.value());
      os << ";\n";

      os << "          Value *toadd = nullptr;\n";
      revres(argIdx, {}, argOpEn.value());

      os << "          if (toadd) addToDiffe(call.getArgOperand(" << argIdx
         << "), toadd";
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
    if (!CalledBlas) llvm::errs() << " opName: " << opName << "\n";
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

// handleBLAS is called in the AdjointGenerator.h 
void emit_handleBLAS(const std::vector<TGPattern> &blasPatterns,
                     raw_ostream &os) {
  os << "bool handleBLAS(llvm::CallInst &call, llvm::Function *called,"
        "BlasInfo blas,const std::vector<bool> &overwritten_args) {         \n"
     << "  using llvm::Type;                                                \n"
     << "  bool result = true;                                              \n"
     << "  if (!gutils->isConstantInstruction(&call)) {                     \n"
     << "    Type *fpType;                                                  \n"
     << "    if (blas.floatType == \"d\") {                                 \n"
     << "      fpType = Type::getDoubleTy(call.getContext());               \n"
     << "    } else if (blas.floatType == \"s\") {                          \n"
     << "      fpType = Type::getFloatTy(call.getContext());                \n"
     << "    } else {                                                       \n"
     << "      assert(false && \"Unreachable\");                            \n"
     << "    }                                                              \n";
  bool first = true;
  for (auto pattern : blasPatterns) {
    bool hasNonInactive = false;
    for (Rule rule : pattern.getRules()) {
        const auto ruleDag = rule.getRuleDag();
        const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
        if (Def->isSubClassOf("MagicInst") && Def->getName() == "inactive")
            continue;
        hasNonInactive = true;
        break;
    }
    if (!hasNonInactive) continue;
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
     << "    const std::vector<bool> &overwritten_args, "
        "llvm::Type *fpType) "
        "{\n"
     << "  \n"
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

void emit_free_and_ending(TGPattern &pattern, raw_ostream &os) {
  os << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
     << "      Mode == DerivativeMode::ReverseModeGradient ||\n"
     << "      Mode == DerivativeMode::ForwardModeSplit) {\n"
     << "    if (shouldFree()) {\n";

  auto nameVec = pattern.getArgNames();
  auto typeMap = pattern.getArgTypeMap();
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty == vincData || ty == mldData) {
      auto name = nameVec[i];
      os << "      if (cache_" << name << ") {\n"
         << "        CreateDealloc(Builder2, free_" << name
         << ");\n"
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
  bool lv23 = pattern.isBLASLevel2or3();

  os << "  const bool byRef = blas.prefix == \"\";\n";
  os << "  Value *cacheval = nullptr;\n\n";
  // lv 2 or 3 functions have an extra arg under the cblas_ abi
  if (lv23) {
    os << "  const int offset = (byRef ? 0 : 1);\n\n";
    auto name = nameVec[0];
    os << "// Next ones shall only be called in the !byRef (thus cblas) case,\n"
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
    os << "  const int pos_" << name << " = " << j << (lv23 ? " + offset" : "")
       << ";\n"
       << "  const auto orig_" << name << " = call.getArgOperand(pos_" << name
       << ");\n"
       << "  auto arg_" << name << " = gutils->getNewFromOriginal(orig_" << name
       << ");\n"
       << "  const auto type_" << name << " = arg_" << name << "->getType();\n"
       << "  const bool overwritten_" << name
       << " = (cacheMode ? overwritten_args[pos_" << name << "] : false);\n";
    if (std::count(actArgs.begin(), actArgs.end(), i)) {
      os << "  const bool active_" << name
         << " = !gutils->isConstantValue(orig_" << name << ");\n";
    }
    os << "\n";
  }

  bool anyActive = false;
  for (size_t i = 0; i < nameVec.size(); i++) {
    argType ty = argTypeMap.lookup(i);
    if (ty == fp) {
      anyActive = true;
    }
  }

  if (anyActive) {
    os << "  int num_active_fp = 0;\n";
    for (size_t i = 0; i < nameVec.size(); i++) {
      argType ty = argTypeMap.lookup(i);
      if (ty == fp) {
        os << "  if (active_" << nameVec[i] << ")\n"
           << "    num_active_fp++;\n";
      }
    }
  }

  for (auto name : llvm::enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto ty = argTypeMap.lookup(name.index());
    if (ty == vincData || ty == mldData) {
      os << "  const bool julia_decl = !type_" << name.value()
         << "->isPointerTy();\n";
      return;
    }
  }
  PrintFatalError("Blas function without vector and matrix?");
}

void emit_scalar_types(TGPattern &pattern, raw_ostream &os) {
  // We only look at the type of the first integer showing up.
  // This allows to learn if we use Fortran abi (byRef) or cabi
  std::string name = "";
  bool foundInt = false;

  auto inputTypes = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();

  for (auto val : inputTypes) {
    if (val.second == len) {
      foundInt = true;
      name = nameVec[val.first];
      break;
    }
  }
  assert(foundInt && "no int type found in blas call");

  os << "  // fpType already given by blas type (s, d, c, z) \n"
     << "  IntegerType *intType = dyn_cast<IntegerType>(type_" << name << ");\n"
     << "  // TODO: add Fortran testcases for Fortran ABI\n"
     << "  if (!intType) {\n"
     << "    const auto PT = cast<PointerType>(type_" << name << ");\n"
     << "    if (blas.suffix.contains(\"64\"))\n"
     << "      intType = IntegerType::get(PT->getContext(), 64);\n"
     << "    else\n"
     << "      intType = IntegerType::get(PT->getContext(), 32);\n"
     << "  }\n\n"
     << "  IntegerType *charType = IntegerType::get(intType->getContext(), "
        "8);\n\n";
  os << "  IntegerType *julia_decl_type = nullptr;\n"
     << "  if (julia_decl)\n"
     << "    julia_decl_type = intType;\n";
}

#include "caching.h"

void extract_scalar(std::string name, std::string elemTy, raw_ostream &os) {
  os << "      if (cache_" << name << ") {\n"
     << "        arg_" << name << " = (cacheTypes.size() == 1)\n"
     << "                    ? cacheval\n"
     << "                    : Builder2.CreateExtractValue(cacheval, "
     << "{cacheidx}, \"tape.ext." << name << "\");\n"
     << "        auto alloc = allocationBuilder.CreateAlloca(" << elemTy
     << ", nullptr, \"byref." << name << "\");\n"
     << "        Builder2.CreateStore(arg_" << name << ", alloc);\n"
     << "        arg_" << name
     << " = Builder2.CreatePointerCast(\n"
     << "            alloc, type_" << name << ", \"cast." << name << "\");\n"
     << "        cacheidx++;\n"
     << "      } else {\n"
     << "        if (Mode != DerivativeMode::ForwardModeSplit)\n"
     << "          arg_" << name << " = lookup(arg_" << name << ", Builder2);\n"
     << "      }\n"
     << "\n";
}

void extract_mat_or_vec(std::string name, raw_ostream &os) {
  os << "      if (cache_" << name << ") {\n"
     << "        arg_" << name << " = (cacheTypes.size() == 1)\n"
     << "                    ? cacheval\n"
     << "                    : Builder2.CreateExtractValue(cacheval, "
        "{cacheidx}, \"tape.ext."
     << name << "\");\n"
     << "        free_" << name << " = arg_" << name << ";\n"
     << "        if (type_" << name << "->isIntegerTy()) {\n"
     << "          arg_" << name << " = Builder2.CreatePtrToInt(arg_" << name
     << ", type_" << name << ");\n"
     << "        } else if (arg_" << name << "->getType() != type_" << name << "){\n"
     << "          arg_" << name << " = Builder2.CreatePointerCast(arg_" << name
     << ", type_" << name << ");\n"
     << "        }\n"
     << "        cacheidx++;\n"
     << "      } else {\n"
     << "        if (Mode != DerivativeMode::ForwardModeSplit)\n"
     << "          arg_" << name << " = lookup(arg_" << name << ", Builder2);\n"
     << "      }\n";
}

void emit_extract_calls(TGPattern &pattern, raw_ostream &os) {
  const auto actArgs = pattern.getActiveArgs();
  const auto typeMap = pattern.getArgTypeMap();
  const auto nameVec = pattern.getArgNames();
  const auto argUsers = pattern.getArgUsers();
  const bool lv23 = pattern.isBLASLevel2or3();

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
    auto ty = typeMap.lookup(i);
    auto name = nameVec[i];
    // this branch used "true_" << name everywhere instead of "arg_" << name
    // before. probably randomly, but check to make sure
    if (ty == len || ty == vincInc || ty == mldLD) {
      extract_scalar(name, "intType", os);
    } else if (ty == fp) {
      extract_scalar(name, "fpType", os);
    } else if (ty == trans) {
      // we are in the byRef branch and trans only exist in lv23.
      // So just unconditionally asume that no layout exist and use i-1
      extract_scalar(name, "charType", os);
    }
  }

  os << "    } else if (Mode != DerivativeMode::ForwardModeSplit) {\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    auto name = nameVec[i];
    if (ty == vincInc || ty == mldLD) {
      os << "      if (cache_" << name << ") {\n"
         << "        true_" << name << " = lookup(true_" << name
         << ", Builder2);\n"
         << "        " << name << " = true_" << name << ";\n"
         << "      }\n";
    } else if (ty == len || ty == fp) {
      os << "      arg_" << name << " = lookup(arg_" << name << ", Builder2);\n"
         << "\n";
    }
  }
  os << "    }\n";

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != mldData)
      continue;
    auto name = nameVec[i];
    extract_mat_or_vec(name, os);
  }

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != vincData)
      continue;
    auto name = nameVec[i];
    const auto vecPosition = i;
    const auto vecUsers = argUsers.lookup(vecPosition);
    const auto incName = nameVec[i + 1];
    extract_mat_or_vec(name, os);
    os << "      if (cache_" << name << ") {\n"
       << "      " << incName << " = ConstantInt::get(intType, 1);\n"
       << "       if (byRef) {\n"
       << "         auto alloc = allocationBuilder.CreateAlloca(intType, "
          "nullptr, \"byref."
       << incName << "\");\n"
       << "         Builder2.CreateStore(" << incName << ", alloc);\n"
       << "         " << incName << " = Builder2.CreatePointerCast(\n"
       << "             alloc, type_" << incName << ", \"cast." << incName
       << "\");\n"
       << "      }\n"
       << " }\n";

    if (vecUsers.size() > 0) {
      os << "   else if (";
      bool first = true;
      // TODO: for higher lv: verify x isn't user from data_x (as only adjoint
      // of x will be used)
      for (auto user : vecUsers) {
        auto userName = nameVec[user];
        if (name == userName)
          continue; // see above
        os << ((first) ? "" : " || ") << "active_" << userName;
        first = false;
      }
      os << ") {\n"
         << "      arg_" << name << " = lookup(arg_" << name
         << ", Builder2);\n"
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
       << "      arg_" << vecName << " = Builder2.CreatePtrToInt(arg_"
       << vecName << ", type_" << vecName << ");\n";
  }

  os << "  }\n";
}

// Will be used by Julia
llvm::SmallString<80> ValueType_helper(TGPattern &pattern, size_t actPos) {
  const auto nameVec = pattern.getArgNames();
  const auto typeMap = pattern.getArgTypeMap();
  llvm::SmallString<80> valueTypes{};

  // start with 1 since layout is only used for cblas (!byRef)
  for (size_t pos = 1; pos < nameVec.size();) {
    auto name = nameVec[pos];
    auto type = typeMap.lookup(pos);

    if (pos > 1) {
      valueTypes.append(", ");
    }

    if (type == len) {
      valueTypes.append("ValueType::Both");
    } else if (type == fp) {
      auto floatName = nameVec[pos];
      if (pos == actPos) {
        valueTypes.append("ValueType::Both");
      } else {
        valueTypes.append((Twine("cache_") + floatName +
                           " ? ValueType::Both : ValueType::Both")
                              .str());
      }
    } else if (type == vincData) {
      const auto nextName = nameVec[pos + 1];
      const auto nextType = typeMap.lookup(pos + 1);
      assert(nextType == vincInc);
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
      // llvm::errs() << "type: " << type << "\n";
      // PrintFatalError("Unhandled type!");
    }
    pos++;
  }
  return valueTypes;
}

// TODO: think about how to handle nested rules which aren't simple calling
// another BLAS fnc.

size_t fwd_call_args(TGPattern &pattern, size_t actArg,
                     llvm::SmallString<40> &result) {
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
    const auto ty = typeMap.lookup(pos);
    if (ty == len) {
      result.append((Twine("arg_") + name).str());
    } else if (ty == fp) {
      if (pos == actArg) {
        result.append((Twine("d_") + name).str());
      } else {
        result.append((Twine("arg_") + name).str());
        // result.append((Twine("fp_") + name).str());
      }
    } else if (ty == vincData) {
      auto nextName = nameVec[pos + 1];
      // get the position of the argument in the primary blas call
      auto nextArgPosition = nameMap.lookup(nextName);
      // and based on that get the fp/int + scalar/vector type
      auto typeOfNextArg = typeMap.lookup(nextArgPosition);
      assert(typeOfNextArg == vincInc);
      if (pos == actArg) {
        result.append((Twine("d_") + name + ", true_" + nextName).str());
      } else {
        result.append((Twine("arg_") + name + ", arg_" + nextName).str());
      }
      pos++; // extra ++ due to also handling vincInc
    } else if (ty == vincInc) {
      // might come without vincData, e.g. after DiffeRet
      result.append(name);
    } else if (ty == mldData) {
      auto nextName = nameVec[pos + 1];
      // get the position of the argument in the primary blas call
      auto nextArgPosition = nameMap.lookup(nextName);
      // and based on that get the fp/int + scalar/vector type
      auto typeOfNextArg = typeMap.lookup(nextArgPosition);
      assert(typeOfNextArg == mldLD);
      if (pos == actArg) {
        result.append((Twine("d_") + name + ", true_" + nextName).str());
      } else {
        result.append((Twine("arg_") + name + ", arg_" + nextName).str());
      }
      pos++; // extra ++ due to also handling mldLD
    } else if (ty == mldLD) {
        // might come without mldData, e.g. after DiffeRet
        // coppied from vincInc, but should verify if actually needed
        result.append((Twine("arg_") + name).str());
    } else if (ty == cblas_layout) {
      // TODO: based on byRef
    } else if (ty == trans || ty == diag || ty == uplo || ty == side){
        result.append((Twine("arg_") + name).str());
    } else {
      // TODO: impl
      // llvm::errs() << "name: " << name << " typename: " << ty << "\n";
      // llvm_unreachable("unimplemented input type!");
    }
    pos++;
  }

  // return the size - 1 due to only using the cblas_layout in the !byRef case
  return nameVec.size() - startArg;
}

void emit_fwd_rewrite_rules(TGPattern &pattern, raw_ostream &os) {
  auto rules = pattern.getRules();
  bool lv23 = pattern.isBLASLevel2or3();
  os << "  /* fwd-rewrite */                                 \n"
     << "  if (Mode == DerivativeMode::ForwardMode ||        \n"
     << "      Mode == DerivativeMode::ForwardModeSplit) {   \n"
     << "                                                    \n"
     << "#if LLVM_VERSION_MAJOR >= 11                        \n"
     << "    auto callval = call.getCalledOperand();         \n"
     << "#else                                               \n"
     << "    auto callval = call.getCalledValue();           \n"
     << "#endif                                            \n\n";

  const auto nameVec = pattern.getArgNames();
  const auto inputTypes = pattern.getArgTypeMap();
  const auto activeArgs = pattern.getActiveArgs();
  for (auto inputType : inputTypes) {
    auto ty = inputType.second;
    if (ty == vincData || ty == mldData) {
      const auto name = nameVec[inputType.first];
      os << "    Value *d_" << name << " = active_" << name << "\n"
         << "     ? gutils->invertPointerM(orig_" << name << ", Builder2)\n"
         << "     : nullptr;\n";
    }
    if (ty == fp) {
      const auto name = nameVec[inputType.first];
      os << "    Value *d_" << name
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
    const auto activeArg = activeArgs[i];
    const auto rule = rules[i];
    const auto actName = nameVec[activeArg];
    auto dcallArgs = llvm::SmallString<40>();
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

void emit_deriv_fnc(StringMap<TGPattern> &patternMap, Rule &rule,
                    llvm::StringSet<> &handled, raw_ostream &os) {
  const auto ruleDag = rule.getRuleDag();
  const auto typeMap = rule.getArgTypeMap();
  const auto opName = ruleDag->getOperator()->getAsString();
  const auto nameMap = rule.getArgNameMap();
  const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
  if (Def->isSubClassOf("b")) {
    const auto dfnc_name = Def->getValueAsString("s");
    if (patternMap.find(dfnc_name.str()) == patternMap.end()) {
      PrintFatalError("calling unknown Blas function");
    }
    TGPattern calledPattern = patternMap.find(dfnc_name.str())->getValue();
    bool derivlv23 = calledPattern.isBLASLevel2or3();
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
    // insert arg types based on .td file
    std::string typeString = "";
    bool first = true;
    for (size_t i = 0; i < ruleDag->getNumArgs(); i++) {
      Init *subArg = ruleDag->getArg(i);
      if (DefInit *def = dyn_cast<DefInit>(subArg)) {
        const auto Def = def->getDef();
        std::string typeToAdd = "";
        if (Def->isSubClassOf("DiffeRet")) {
          typeToAdd = "byRef ? PointerType::getUnqual(call.getType()) : "
                      "call.getType()\n";
        } else if (Def->isSubClassOf("input")) {
          auto argStr = Def->getValueAsString("name");
          //  primary and adj have the same type
          typeToAdd = (Twine("type_") + argStr).str();
        } else if (Def->isSubClassOf("adj")) {
          auto argStr = Def->getValueAsString("name");
          // primary and adj have the same type
          typeToAdd = (Twine("type_") + argStr).str();
        } else if (Def->isSubClassOf("Constant")) {
          typeToAdd = "byRef ? (Type*)PointerType::getUnqual(fpType) : (Type*)fpType";
        } else if (Def->isSubClassOf("Char")) {
          typeToAdd = "byRef ? (Type*)PointerType::getUnqual(charType) : (Type*)charType";
        } else if (Def->isSubClassOf("ConstantInt")) {
          typeToAdd = "byRef ? (Type*)PointerType::getUnqual(intType) : (Type*)intType";
        } else if (Def->isSubClassOf("transpose")) {
          auto argStr = Def->getValueAsString("name");
          // transpose the given trans arg, but type stays
          typeToAdd = (Twine("type_") + argStr).str();
        } else {
          PrintFatalError(Def->getLoc(), "PANIC! Unsupported Definit");
        }
        typeString += ((first) ? "" : ", ") + typeToAdd;
      } else { 
        if (auto Dag = dyn_cast<DagInit>(subArg)) {
          auto Def = cast<DefInit>(Dag->getOperator())->getDef();
          if (Def->isSubClassOf("MagicInst") && Def->getName() == "Rows") {
            if (!first) typeString += ", ";
            typeString += (Twine("type_") + Dag->getArgNameStr(1)).str();
            first = false;
            continue;
            }
        }
        const auto argStr = ruleDag->getArgNameStr(i);
        // skip layout because it is cblas only, 
        // so not relevant for the byRef Fortran abi.
        // Optionally add it later as first arg for byRef.
        if (argStr == "layout")
          continue;
        typeString += (first ? "" : ", ");
        typeString += (Twine("type_") + argStr).str();
      }
      first = false;
    }

    os << "    llvm::FunctionType *FT" << dfnc_name << " = nullptr;\n";
    if (derivlv23) {
      os << "    if(byRef) {\n"
         << "      Type* tys" << dfnc_name << "[] = {" << typeString << "};\n"
         << "      FT" << dfnc_name
         << " = FunctionType::get(Builder2.getVoidTy(), tys" << dfnc_name
         << ", false);\n"
         << "    } else {\n"
         << "      Type* tys" << dfnc_name << "[] = {type_layout, "
         << typeString << "};\n"
         << "      FT" << dfnc_name
         << " = FunctionType::get(Builder2.getVoidTy(), tys" << dfnc_name
         << ", false);\n"
         << "    }\n";
    } else {
      os << "    Type* tys" << dfnc_name << "[] = {" << typeString << "};\n"
         << "    FT" << dfnc_name
         << " = FunctionType::get(Builder2.getVoidTy(), tys" << dfnc_name
         << ", false);\n";
    }

    os << "auto derivcall_" << dfnc_name
       << " = gutils->oldFunc->getParent()->getOrInsertFunction(\n"
       << "  (blas.prefix + blas.floatType + \"" << dfnc_name
       << "\" + blas.suffix).str(), FT" << dfnc_name << ");\n";

    os << "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name
       << ".getCallee()))\n"
       << "    {\n"
       << "      attribute_" << dfnc_name << "(blas, F);\n"
       << "    }\n\n";
  } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "noop") {
    // nothing to prepare
  } else if (Def->isSubClassOf("DiffeRet")) {
    // nothing to prepare
  } else if (Def->isSubClassOf("Inst")) {
    // TODO:
    return;
    PrintFatalError("Unhandled Inst Rule!");
  } else {
    PrintFatalError("Unhandled deriv Rule!");
  }
}

// fill the result string and return the number of added args
size_t rev_call_args(Rule &rule, size_t actArg, llvm::SmallString<40> &result) {

  const auto nameMap = rule.getArgNameMap();
  const auto typeMap = rule.getArgTypeMap();
  const auto ruleDag = rule.getRuleDag();
  const size_t numArgs = ruleDag->getNumArgs();
  const size_t startArg = rule.isBLASLevel2or3() ? 1 : 0;

  // just replace argOps with rule
  for (size_t pos = startArg; pos < numArgs;) {
    if (pos > startArg) {
      result.append(", ");
    }

    auto arg = ruleDag->getArg(pos);
    if (auto Dag = dyn_cast<DagInit>(arg)) {
      auto Def = cast<DefInit>(Dag->getOperator())->getDef();

      if (Def->isSubClassOf("MagicInst") && Def->getName() == "Rows") {
        auto tname = Dag->getArgNameStr(0);
        auto rname = Dag->getArgNameStr(1);
        auto cname = Dag->getArgNameStr(2);
        result.append((Twine("get_blas_row(Builder2, arg_transposed_") + tname + ", arg_" + rname + ", arg_" + cname + ", byRef)").str());
      } else {
        llvm::errs() << Def->getName() << "\n";
        PrintFatalError("Dag/Def that isn't a DiffeRet!");
      }
    } else if (DefInit *DefArg = dyn_cast<DefInit>(arg)) {
      auto Def = DefArg->getDef();
      if (Def->isSubClassOf("DiffeRet")) {
        result.append("dif");
      } else if (Def->isSubClassOf("adj")) {
        auto name = Def->getValueAsString("name");
        result.append((Twine("d_") + name).str());
      } else if (Def->isSubClassOf("input")) {
        auto name = Def->getValueAsString("name");
        result.append((Twine("arg_") + name).str());
        // result.append((Twine("input_") + name).str());
      } else if (Def->isSubClassOf("MagicInst")) {
        llvm::errs() << "MagicInst\n";
      } else if (Def->isSubClassOf("Constant")) {
        auto val = Def->getValueAsString("value");
        result.append((Twine("to_blas_callconv(Builder2, ConstantFP::get(fpType, ") + val + "), byRef, nullptr, allocationBuilder, \"constant.fp." + val + "\")").str());
      } else if (Def->isSubClassOf("Char")) {
        auto val = Def->getValueAsString("value");
        result.append((Twine("to_blas_callconv(Builder2, ConstantInt::get(charType, '") + val + "'), byRef, nullptr, allocationBuilder, \"constant.char." + val + "\")").str());
      } else if (Def->isSubClassOf("ConstantInt")) {
        auto val = Def->getValueAsString("value");
        result.append((Twine("to_blas_callconv(Builder2, ConstantInt::get(intType, ") + val + "), byRef, nullptr, allocationBuilder, \"constant.int." + val + "\")").str());
      } else if (Def->isSubClassOf("transpose")) {
        auto name = Def->getValueAsString("name");
        result.append((Twine("arg_transposed_") + name).str());
      } else {
        llvm::errs() << Def->getName() << "\n";
        PrintFatalError("Def that isn't a DiffeRet!");
      }
    } else {
      auto name = ruleDag->getArgNameStr(pos);
      assert(name != "");
      // get the position of the argument in the primary blas call
      if (nameMap.count(name) != 1) {
        llvm::errs() << "couldn't find name: " << name << "\n";
        PrintFatalError("arg not in nameMap!");
      }
      assert(nameMap.count(name) == 1);
      auto argPosition = nameMap.lookup(name);
      // and based on that get the fp/int + scalar/vector type
      auto ty = typeMap.lookup(argPosition);

      // Now we create the adj call args through concating type and primal name
      if (ty == len) {
        result.append((Twine("arg_") + name).str());
      } else if (ty == fp) {
        if (argPosition == actArg) {
          result.append((Twine("d_") + name).str());
        } else {
          result.append((Twine("arg_") + name).str());
        }
      } else if (ty == vincData) {
        auto nextName = ruleDag->getArgNameStr(pos + 1);
        // get the position of the argument in the primary blas call
        auto nextArgPosition = nameMap.lookup(nextName);
        // and based on that get the fp/int + scalar/vector type
        auto typeOfNextArg = typeMap.lookup(nextArgPosition);
        assert(typeOfNextArg == vincInc);
        if (argPosition == actArg) {
          // shadow d_<X> wasn't overwritten or cached, so use true_inc<X>
          result.append((Twine("d_") + name + ", true_" + nextName).str());
        } else {
          result.append((Twine("arg_") + name).str());
          //result.append((Twine("data_") + name).str());
          // if we cached arg_<X> then don't use inc_<X> but rather 1,
          // since we malloc'd the cache
          // result.append((Twine(", cache_" + name + " ?
          // ConstantInt::get(intType, 1) : arg_") + nextName).str());
          //  orig arg might was cached in which cased we overwrote arg_ to 1
          result.append((Twine(", arg_") + nextName).str());
        }
        pos++; // extra ++ due to also handling vincInc
      } else if (ty == vincInc) {
        // might come without vincData, e.g. after DiffeRet
        // result.append(name);
        result.append((Twine("arg_") + name).str());
      } else if (ty == mldData) {
        auto nextName = ruleDag->getArgNameStr(pos + 1);
        // get the position of the argument in the primary blas call
        auto nextArgPosition = nameMap.lookup(nextName);
        // and based on that get the fp/int + scalar/vector type
        auto typeOfNextArg = typeMap.lookup(nextArgPosition);
        assert(typeOfNextArg == mldLD);
        if (pos == actArg) {
          result.append((Twine("d_") + name + ", true_" + nextName).str());
        } else {
          // TODO: fix ld, which is width of the matrix in the cached case.
          // Similar to the vincData case.
          // However, how to access width, especially under transposition?
          result.append((Twine("arg_") + name + ", arg_" + nextName).str());
        }
        pos++; // extra ++ due to also handling mldLD
      } else if (ty == mldLD) {
        // might come without mldData, e.g. after DiffeRet
        // coppied from vincInc, but should verify if actually needed
        // result.append(name);
        result.append((Twine("arg_") + name).str());
      } else if (ty == trans) {
        result.append((Twine("arg_") + name).str());
        //result.append((Twine("arg_") + name).str());
      } else {
        // TODO
        // llvm::errs() << "name: " << name << " typename: " << ty <<
        // "\n"; llvm_unreachable("unimplemented input type!");
      }
    }
    pos++;
  }

  return numArgs - startArg;
}

void emit_rev_rewrite_rules(StringMap<TGPattern> patternMap, TGPattern &pattern,
                            raw_ostream &os) {

  const auto nameVec = pattern.getArgNames();
  const auto typeMap = pattern.getArgTypeMap();
  const auto rules = pattern.getRules();
  const auto activeArgs = pattern.getActiveArgs();
  const bool lv23 = pattern.isBLASLevel2or3();

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

  os << "  /* rev-rewrite */                                 \n"
     << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
     << "      Mode == DerivativeMode::ReverseModeGradient) {\n"
     << "    Value *alloc = nullptr;\n"
     << "    if (byRef) {\n"
     << "      alloc = allocationBuilder.CreateAlloca(fpType, nullptr, \"ret\");\n"
     << "    }\n\n";
  if (hasDiffeRetVal) {
    os << "    Value *dif = diffe(&call, Builder2);\n";
  }

  // We only emit one derivcall per blass call type.
  // This verifies that we don't end up with multiple declarations.
  llvm::StringSet handled{};
  for (auto rule : rules) {
    emit_deriv_fnc(patternMap, rule, handled, os);
  }

  for (size_t i = 0; i < nameVec.size(); i++) {
    const auto name = nameVec[i];
    const auto ty = typeMap.lookup(i);
    if (ty == vincData || ty == mldData) {
      os << "    Value *d_" << name << " = active_" << name << "\n"
         << "     ? lookup(gutils->invertPointerM(orig_" << name
         << ", Builder2), Builder2)\n"
         << "     : nullptr;\n";
    } else if (ty == fp) {
      os << "    Value *d_" << name << " = UndefValue::get(fpType);\n";
    }
  }

  // now we can use it to transpose our trans arguments if they exist
  for (size_t i = (lv23 ? 1 : 0); i < nameVec.size(); i++) {
    auto name = nameVec[i];
    if (typeMap.lookup(i) == trans) {
      os << "  llvm::Value* arg_transposed_" << name
         << " = transpose(Builder2, arg_" << name
         << ", byRef, charType, allocationBuilder, \"" << name << "\");\n";
    }
  }

  os << "    applyChainRule(\n"
     << "      Builder2,\n"
     << "      [&](";
  bool first = true;
  for (auto arg : activeArgs) {
    const auto name = nameVec[arg];
    const auto ty = typeMap.lookup(arg);
    // We don't pass in shaddows of fp values,
    // we just create and struct-return the shaddows
    if (ty == fp)
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
       << "        }\n";
  } else {
    os << ") {\n";
  }

  for (Rule rule : rules) {
    const size_t actArg = rule.getHandledArgIdx();
    const auto ruleDag = rule.getRuleDag();
    const auto name = nameVec[actArg];
    const auto nameMap = rule.getArgNameMap();
    const auto ty = typeMap.lookup(actArg);
    auto args = llvm::SmallString<40>();
    const size_t numArgs = rev_call_args(rule, actArg, args);
    const auto valueTypes = ValueType_helper(pattern, actArg);
    const auto opName = ruleDag->getOperator()->getAsString();
    const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
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

      const auto dfnc_name = Def->getValueAsString("s");
      os << "      if (" << actCondition << ") {\n"
         << "        Value *args1[" << numArgs << "] = {" << args << "};\n"
         << "        const auto Defs = gutils->getInvertedBundles(&call, {"
         << valueTypes << "}, Builder2, /* lookup */ true);\n";

      if (ty == fp) {
        // extra handling, since we will update only a fp scalar as part of the
        // return struct it's presumably done by setting it to the value
        // returned by this call
        os << "        CallInst *cubcall = "
              "cast<CallInst>(Builder2.CreateCall(derivcall_"
           << dfnc_name << ", args1, Defs));\n"
           << "        if (byRef) {\n"
           << "          ((DiffeGradientUtils *)gutils)"
           << "          ->addToInvertedPtrDiffe(&call, nullptr, fpType, 0,"
           << "(blas.suffix.contains(\"64\") ? 8 : 4), arg_" << name 
           << ", cubcall, Builder2);\n"
           << "        } else {"
           << "        addToDiffe(arg_" << name
           << ", cubcall, Builder2, fpType);\n"
           << "        }"
           << "      }\n";

      } else {
        os << "        Builder2.CreateCall(derivcall_" << dfnc_name
           << ", args1, Defs);\n"
           << "      }\n";
      }
    } else if (Def->isSubClassOf("adj")) {
    } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "noop") {
    } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "inactive") {
      os << "      assert(!active_" << name << ");\n";
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
    const auto name = nameVec[arg];
    const auto ty = typeMap.lookup(arg);
    // We don't pass in shaddows of fp values,
    // we just create and struct-return the shaddows
    if (ty == fp)
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

// Further optimization: re-use / share caches where possible

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
    bool hasNonInactive = false;
    for (Rule rule : newPattern.getRules()) {
        const auto ruleDag = rule.getRuleDag();
        const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
        if (Def->isSubClassOf("MagicInst") && Def->getName() == "inactive")
            continue;
        hasNonInactive = true;
        break;
    }
    if (!hasNonInactive) continue;

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

#include "blasDeclUpdater.h"
#include "blasTAUpdater.h"
#include "blasDiffUseUpdater.h"

static bool EnzymeTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case GenDerivatives:
    emitDerivatives(records, os);
    return false;
  case GenBlasDerivatives:
    emitBlasDerivatives(records, os);
    return false;
  case UpdateBlasDecl:
    emitBlasDeclUpdater(records, os);
    return false;
  case GenBlasDiffUse:
    emitBlasDiffUse(records, os);
    return false;
  case UpdateBlasTA:
    emitBlasTAUpdater(records, os);
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

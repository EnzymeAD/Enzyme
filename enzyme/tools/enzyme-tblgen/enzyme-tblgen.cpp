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

#include "blas-tblgen.h"
#include "caching.h"
#include "datastructures.h"

using namespace llvm;

enum ActionType {
  CallDerivatives,
  InstDerivatives,
  BinopDerivatives,
  IntrDerivatives,
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
           cl::values(clEnumValN(IntrDerivatives, "gen-intr-derivatives",
                                 "Generate intrinsic derivative")),
           cl::values(clEnumValN(BinopDerivatives, "gen-binop-derivatives",
                                 "Generate binaryoperator derivative")),
           cl::values(clEnumValN(InstDerivatives, "gen-inst-derivatives",
                                 "Generate instruction derivative")),
           cl::values(clEnumValN(CallDerivatives, "gen-call-derivatives",
                                 "Generate call derivative")));

void getFunction(const Twine &curIndent, raw_ostream &os, StringRef callval,
                 StringRef FT, StringRef cconv, Init *func,
                 StringRef origName) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(func)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "SameFunc" || Def->isSubClassOf("SameFunc")) {
      os << "#if LLVM_VERSION_MAJOR >= 11\n";
      os << curIndent << "auto " << callval << " = cast<CallInst>(&" << origName
         << ")->getCalledOperand();\n";
      os << "#else\n";
      os << curIndent << "auto " << callval << " = cast<CallInst>(&" << origName
         << ")->getCalledValue();\n";
      os << "#endif\n";
      os << curIndent << "auto " << FT << " = cast<CallInst>(&" << origName
         << ")->getFunctionType();\n";
      os << curIndent << "auto " << cconv << " = cast<CallInst>(&" << origName
         << ")->getCallingConv();\n";
      return;
    }
    if (opName == "SameTypesFunc" || Def->isSubClassOf("SameTypesFunc")) {
      os << curIndent << "auto " << FT << " = cast<CallInst>(&" << origName
         << ")->getFunctionType();\n";
      os << curIndent << "auto " << callval
         << " = gutils->oldFunc->getParent()->getOrInsertFunction(";
      os << Def->getValueInit("name")->getAsString();
      os << ", " << FT << ", called->getAttributes()).getCallee();\n";
      os << curIndent << "auto " << cconv << " = cast<CallInst>(&" << origName
         << ")->getCallingConv();\n";
      return;
    }
    if (opName == "PrependArgTypesFunc" ||
        Def->isSubClassOf("PrependArgTypesFunc")) {
      os << curIndent << "auto " << FT << "_old = cast<CallInst>(&" << origName
         << ")->getFunctionType();\n";
      os << curIndent << "SmallVector<llvm::Type*, 1> " << FT << "_args = {";
      bool seen = false;
      for (auto pre : *Def->getValueAsListInit("pretys")) {
        if (seen)
          os << ", ";
        os << "Type::get" << cast<StringInit>(pre)->getValue()
           << "Ty(gutils->oldFunc->getContext())";
      }
      os << "};\n";
      os << curIndent << FT << "_args.append(" << FT
         << "_old->params().begin(), " << FT << "_old->params().end());\n";
      os << curIndent << "auto " << FT << " = FunctionType::get(" << FT
         << "_old->getReturnType(), " << FT << "_args, " << FT
         << "_old->isVarArg());\n";
      os << curIndent << "auto " << callval
         << " = gutils->oldFunc->getParent()->getOrInsertFunction(";
      os << Def->getValueInit("name")->getAsString();
      os << ", " << FT << ", called->getAttributes()).getCallee();\n";
      os << curIndent << "auto " << cconv << " = cast<CallInst>(&" << origName
         << ")->getCallingConv();\n";
      return;
    }
  }
  assert(0 && "Unhandled function");
}
void getIntrinsic(raw_ostream &os, StringRef intrName, ListInit *typeInit,
                  const Twine &argStr, StringRef origName) {
  os << "Intrinsic::getDeclaration(mod, Intrinsic::" << intrName
     << ", std::vector<Type*>({";
  bool first = true;
  for (auto intrType : *typeInit) {
    if (!first)
      os << ", ";
    auto arg = cast<IntInit>(intrType)->getValue();
    os << argStr << "_" << arg << "->getType()";
    first = false;
  }
  os << "}))";
}

raw_ostream &operator<<(raw_ostream &os, StringMap<std::string> &C) {
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

void initializeNames(const Twine &curIndent, raw_ostream &os, Init *resultTree,
                     const Twine &prefix) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    for (size_t i = 0; i < resultRoot->arg_size(); i++) {
      auto arg = resultRoot->getArg(i);
      auto name = resultRoot->getArgName(i);
      if (isa<UnsetInit>(arg) && name) {
        continue;
      }
      if (name) {
        auto namev = name->getAsUnquotedString();
        os << curIndent << "llvm::Value *" << prefix << "_" + namev
           << " = nullptr;\n";
      }
      initializeNames(curIndent, os, arg, prefix);
    }
  } else if (ListInit *lst = dyn_cast<ListInit>(resultTree)) {
    for (auto elem : *lst)
      initializeNames(curIndent, os, elem, prefix);
  }
}

struct VariableSetting {
  StringMap<std::string> nameToOrdinal;
  StringMap<bool> isVector;

  std::pair<std::string, bool> lookup(StringRef name, Record *pattern,
                                      Init *resultRoot) {
    auto ord = nameToOrdinal.find(name);
    if (ord == nameToOrdinal.end())
      PrintFatalError(pattern->getLoc(), Twine("unknown named operand '") +
                                             name + "'" +
                                             resultRoot->getAsString());
    auto iv = isVector.find(name);
    assert(iv != isVector.end());
    return std::make_pair(ord->getValue(), iv->getValue());
  }

  void insert(StringRef name, StringRef value, bool vec) {
    nameToOrdinal[name] = value;
    isVector[name] = vec;
  }
};

#define INDENT "  "
bool handle(const Twine &curIndent, const Twine &argPattern, raw_ostream &os,
            Record *pattern, Init *resultTree, StringRef builder,
            VariableSetting &nameToOrdinal, bool lookup,
            ArrayRef<unsigned> retidx, StringRef origName,
            bool newFromOriginal);

SmallVector<bool, 1> prepareArgs(const Twine &curIndent, raw_ostream &os,
                                 const Twine &argName, Record *pattern,
                                 DagInit *resultRoot, StringRef builder,
                                 VariableSetting &nameToOrdinal, bool lookup,
                                 ArrayRef<unsigned> retidx, StringRef origName,
                                 bool newFromOriginal) {
  SmallVector<bool, 1> vectorValued;

  size_t idx = 0;
  for (auto &&[args, names] :
       zip(resultRoot->getArgs(), resultRoot->getArgNames())) {
    os << curIndent << "auto " << argName << "_" << idx << " = ";
    idx++;
    if (isa<UnsetInit>(args) && names) {
      auto [ord, vecValue] =
          nameToOrdinal.lookup(names->getValue(), pattern, resultRoot);
      if (!vecValue && !StringRef(ord).startswith("local")) {
        if (lookup)
          os << "lookup(";
        if (newFromOriginal)
          os << "gutils->getNewFromOriginal(";
      }
      os << ord;
      if (!vecValue && !StringRef(ord).startswith("local")) {
        if (newFromOriginal)
          os << ")";
        if (lookup)
          os << ", " << builder << ")";
      }
      os << ";\n";
      vectorValued.push_back(vecValue);
      continue;
    }
    vectorValued.push_back(handle(curIndent, argName + "_" + Twine(idx), os,
                                  pattern, args, builder, nameToOrdinal, lookup,
                                  retidx, origName, newFromOriginal));
    os << ";\n";
    if (names) {
      auto name = names->getAsUnquotedString();
      nameToOrdinal.insert(name, "local_" + name, vectorValued.back());
      os << curIndent << "local_" << name << " = " << argName << "_"
         << (idx - 1) << ";\n";
    }
  }
  return vectorValued;
}

// Returns whether value generated is a vector value or not.
bool handle(const Twine &curIndent, const Twine &argPattern, raw_ostream &os,
            Record *pattern, Init *resultTree, StringRef builder,
            VariableSetting &nameToOrdinal, bool lookup,
            ArrayRef<unsigned> retidx, StringRef origName,
            bool newFromOriginal) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (Def->isSubClassOf("Inst")) {
      opName = Def->getValueAsString("name");
    }
    if (opName == "DiffeRetIndex" || Def->isSubClassOf("DiffeRetIndex")) {

      auto indicesP = dyn_cast<ListInit>(Def->getValueInit("indices"));
      if (!indicesP)
        PrintFatalError(pattern->getLoc(),
                        Twine("list 'indices' not defined in ") +
                            resultTree->getAsString());
      SmallVector<unsigned, 2> retidx_cur;
      if (indicesP->getValues().size() == 1 &&
          cast<IntInit>(indicesP->getValues()[0])->getValue() == -1) {
        retidx_cur = SmallVector<unsigned, 2>(retidx.begin(), retidx.end());
      } else {
        for (auto res : indicesP->getValues()) {
          auto val = cast<IntInit>(res)->getValue();
          assert(val >= 0);
          retidx_cur.push_back((unsigned)val);
        }
      }

      if (retidx_cur.size() == 0) {
        os << "dif";
      } else {
        os << "({\n";
        os << curIndent << INDENT
           << "Value *out = UndefValue::get(gutils->getShadowType(getSubType("
           << origName << ".getType()";
        for (auto ind : retidx_cur) {
          os << ", " << ind;
        }
        os << ")));\n";
        os << curIndent << INDENT
           << "for(unsigned int idx=0, W=gutils->getWidth(); "
              "idx<W; idx++) {\n";

        os << curIndent << INDENT << INDENT
           << "Value *prev = (gutils->getWidth() == 1) ? gutils->extractMeta("
           << builder << ", dif, ArrayRef<unsigned>({";
        bool first = true;
        for (auto ind : retidx_cur) {
          if (!first)
            os << ", ";
          os << ind;
          first = false;
        }
        os << "})) : gutils->extractMeta(" << builder
           << ", dif, ArrayRef<unsigned>({idx";
        for (auto ind : retidx_cur) {
          os << ", ";
          os << ind;
        }
        os << "}));\n";
        os << curIndent << INDENT << INDENT
           << "out = (gutils->getWidth() > 1) ? "
              "Builder2.CreateInsertValue(out, prev, idx) : prev;\n";
        os << curIndent << INDENT << INDENT << "}\n";
        os << curIndent << INDENT << "out; })\n";
      }
      return true;
    } else if (opName == "TypeOf" || Def->isSubClassOf("TypeOf")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op TypeOf supported");

      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultRoot);
        assert(!isVec);
        os << ord;
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in typeof") +
                            resultTree->getAsString());
      os << "->getType()";
      return false;
    } else if (opName == "VectorSize" || Def->isSubClassOf("VectorSize")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(),
                        "only single op VectorSize supported");

      os << "cast<VectorType>(";

      if (isa<UnsetInit>(resultRoot->getArg(0)) && resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultRoot);
        assert(!isVec);
        os << ord;
      } else
        handle(curIndent + INDENT, argPattern + "_vs", os, pattern,
               resultRoot->getArg(0), builder, nameToOrdinal, lookup, retidx,
               origName, newFromOriginal);

      os << ")";
#if LLVM_VERSION_MAJOR >= 11
      os << "->getElementCount()";
#if LLVM_VERSION_MAJOR == 11
      os << ".Min";
#endif
#else
      os << "->getNumElements()";
#endif
      return false;
    } else if (opName == "SelectIfActive" ||
               Def->isSubClassOf("SelectIfActive")) {
      if (resultRoot->getNumArgs() != 3)
        PrintFatalError(pattern->getLoc(),
                        "only three op SelectIfActive supported");

      os << "({\n";
      os << curIndent << INDENT << "// Computing SelectIfActive\n";
      os << curIndent << INDENT << "Value *imVal = nullptr;\n";

      os << curIndent << INDENT << "if (!gutils->isConstantValue(";

      if (isa<UnsetInit>(resultRoot->getArg(0)) && resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultRoot);
        assert(!isVec);
        os << ord;
      } else
        assert("Requires name for arg");

      os << ")) {\n";

      for (size_t i = 1; i < 3; i++) {
        os << curIndent << INDENT << INDENT << "imVal = ";
        bool vector;
        if (isa<UnsetInit>(resultRoot->getArg(i)) &&
            resultRoot->getArgName(i)) {
          auto name = resultRoot->getArgName(i)->getAsUnquotedString();
          auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultRoot);
          vector = isVec;
          os << ord;
        } else
          vector = handle(curIndent + INDENT + INDENT,
                          argPattern + "_sia_" + Twine(i), os, pattern,
                          resultRoot->getArg(i), builder, nameToOrdinal, lookup,
                          retidx, origName, newFromOriginal);
        os << ";\n";

        if (!vector) {
          os << curIndent << INDENT << INDENT
             << "llvm::Value* vec_imVal = gutils->getWidth() == 1 ? imVal : "
                "UndefValue::get(gutils->getShadowType(imVal"
             << "->getType()));\n";
          os << curIndent << INDENT << INDENT
             << "if (gutils->getWidth() != 1)\n";
          os << curIndent << INDENT << INDENT << INDENT
             << "for (size_t i=0; i<gutils->getWidth(); i++)\n";
          os << curIndent << INDENT << INDENT << INDENT << INDENT
             << "vec_imVal = " << builder
             << ".CreateInsertValue(vec_imVal, imVal, "
                "std::vector<unsigned>({(unsigned)i}));\n";
          os << curIndent << INDENT << INDENT << "imVal = vec_imVal;\n";
        }
        if (i == 1)
          os << curIndent << INDENT << "} else {\n";
        else
          os << curIndent << INDENT << "}\n";
      }

      os << curIndent << INDENT << "imVal;\n";
      os << curIndent << "})";
      return true;
    } else if (opName == "ConstantFP" || Def->isSubClassOf("ConstantFP")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(),
                        "only single op constantfp supported");

      auto value = dyn_cast<StringInit>(Def->getValueInit("value"));
      if (!value)
        PrintFatalError(pattern->getLoc(), Twine("'value' not defined in ") +
                                               resultTree->getAsString());

      os << "ConstantFP::get(";
      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);
        os << ord;
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in constantfp") +
                            resultTree->getAsString());
      os << "->getType(), \"" << value->getValue() << "\")";
      return false;
    } else if (opName == "Zero" || Def->isSubClassOf("Zero")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op Zero supported");
      os << "Constant::getNullValue(";
      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);
        os << ord;
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in constantfp") +
                            resultTree->getAsString());
      os << "->getType())";
      return false;
    } else if (opName == "ConstantCFP" || Def->isSubClassOf("ConstantCFP")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(),
                        "only single op constantfp supported");

      auto rvalue = dyn_cast<StringInit>(Def->getValueInit("rvalue"));
      if (!rvalue)
        PrintFatalError(pattern->getLoc(), Twine("'rvalue' not defined in ") +
                                               resultTree->getAsString());

      auto ivalue = dyn_cast<StringInit>(Def->getValueInit("ivalue"));
      if (!ivalue)
        PrintFatalError(pattern->getLoc(), Twine("'ivalue' not defined in ") +
                                               resultTree->getAsString());
      os << "({\n";
      os << curIndent << INDENT << "auto ty = ";
      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);
        os << ord;
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in constantcfp") +
                            resultTree->getAsString());
      os << "->getType();\n";
      os << curIndent << INDENT << "Value *ret = nullptr;\n";
      os << curIndent << INDENT
         << "if (auto ST = dyn_cast<StructType>(ty)) {\n";
      os << curIndent << INDENT << INDENT
         << "ret = ConstantStruct::get(ST, "
            "{(llvm::Constant*)ConstantFP::get(ST->getElementType(0), \""
         << rvalue->getValue()
         << "\"), (llvm::Constant*)ConstantFP::get(ST->getElementType(1), \""
         << ivalue->getValue() << "\")});\n";
      os << curIndent << INDENT << "} else assert(0 && \"unhandled cfp\");\n";
      os << curIndent << INDENT << "ret;\n";
      os << curIndent << "})\n";
      return false;
    } else if (opName == "ConstantInt" || Def->isSubClassOf("ConstantInt")) {

      auto valueP = dyn_cast<IntInit>(Def->getValueInit("value"));
      if (!valueP)
        PrintFatalError(pattern->getLoc(),
                        Twine("int 'value' not defined in ") +
                            resultTree->getAsString());
      auto value = valueP->getValue();

      auto bitwidthP = dyn_cast<IntInit>(Def->getValueInit("bitwidth"));
      if (!bitwidthP)
        PrintFatalError(pattern->getLoc(),
                        Twine("int 'bitwidth' not defined in ") +
                            resultTree->getAsString());
      auto bitwidth = bitwidthP->getValue();

      os << "ConstantInt::getSigned(";

      if (bitwidth == 0) {
        if (resultRoot->getNumArgs() != 1)
          PrintFatalError(
              pattern->getLoc(),
              "only single op constantint supported with unspecified width");

        if (resultRoot->getArgName(0)) {
          auto name = resultRoot->getArgName(0)->getAsUnquotedString();
          auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultTree);
          assert(!isVec);
          os << ord;
        } else
          PrintFatalError(pattern->getLoc(),
                          Twine("unknown named operand in constantint") +
                              resultTree->getAsString());
        os << "->getType()";
      } else {
        if (resultRoot->getNumArgs() != 0)
          PrintFatalError(
              pattern->getLoc(),
              "only zero op constantint supported with specified width");
        os << "Type::getIntNTy(gutils->oldFunc->getContext(), " << bitwidth
           << ")";
      }
      os << ", " << value << ")";
      return false;
    } else if (opName == "GlobalExpr" || Def->isSubClassOf("GlobalExpr")) {
      if (resultRoot->getNumArgs() != 0)
        PrintFatalError(pattern->getLoc(), "only zero op globalexpr supported");

      auto value = dyn_cast<StringInit>(Def->getValueInit("value"));
      if (!value)
        PrintFatalError(pattern->getLoc(),
                        Twine("string 'value' not defined in ") +
                            resultTree->getAsString());

      os << value->getValue();
      return false;
    } else if (opName == "Undef" || Def->isSubClassOf("Undef")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op undef supported");

      os << "UndefValue::get(";
      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);
        os << ord;
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in constantfp") +
                            resultTree->getAsString());
      os << "->getType())";
      return false;
    } else if (opName == "Shadow" || Def->isSubClassOf("Shadow")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op shadow supported");

      if (lookup)
        os << "lookup(";
      os << "gutils->invertPointerM(";

      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec] = nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);
        os << ord;
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in shadow") +
                            resultTree->getAsString());
      os << ", " << builder << ")";
      if (lookup)
        os << ", " << builder << ")";
      return true;
    } else if (Def->isSubClassOf("MultiReturn")) {
      os << "({\n";

      bool useStruct = Def->getValueAsBit("struct");

      SmallVector<bool, 1> vectorValued = prepareArgs(
          curIndent + INDENT, os, argPattern, pattern, resultRoot, builder,
          nameToOrdinal, lookup, retidx, origName, newFromOriginal);
      bool anyVector = false;
      for (auto b : vectorValued)
        anyVector |= b;

      if (!useStruct)
        assert(vectorValued.size());

      os << curIndent << INDENT << "Value *res = UndefValue::get(";
      if (anyVector)
        os << "gutils->getShadowType(";

      if (useStruct)
        os << "StructType::get(gutils->newFunc->getContext(), "
              "std::vector<llvm::Type*>({";
      else
        os << "ArrayType::get(";
      for (size_t i = 0; i < (useStruct ? vectorValued.size() : 1); i++) {
        if (i != 0)
          os << ", ";
        if (!vectorValued[i])
          os << argPattern << "_" << i << "->getType()";
        else
          os << "(gutils->getWidth() == 1) ? " << argPattern << "_" << i
             << "->getType() : getSubType(" << argPattern << "_" << i
             << "->getType(), -1)";
      }
      if (useStruct)
        os << "}))";
      else
        os << ", " << vectorValued.size() << ")";

      if (anyVector)
        os << ")";
      os << ");\n";

      if (anyVector)
        os << curIndent << INDENT
           << "for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";
      else
        os << curIndent << INDENT << "{\n";

      for (size_t i = 0; i < vectorValued.size(); i++) {
        os << curIndent << INDENT << INDENT << "{\n";
        os << curIndent << INDENT << INDENT << INDENT
           << "std::vector<unsigned> idxs;\n";
        if (vectorValued[i])
          os << curIndent << INDENT << INDENT << INDENT
             << "if (gutils->getWidth() != 1) idxs.push_back(idx);\n";
        os << curIndent << INDENT << INDENT << INDENT << "idxs.push_back(" << i
           << ");\n";
        os << curIndent << INDENT << INDENT << INDENT << "res = " << builder
           << ".CreateInsertValue(res, ";
        if (vectorValued[i])
          os << "(gutils->getWidth() == 1) ? " << argPattern << "_" << i
             << " : gutils->extractMeta(" << builder << ", " << argPattern
             << "_" << i << ", idx)";
        else
          os << argPattern << "_" << i << "";
        os << ", idxs);\n";
        os << curIndent << INDENT << INDENT << "}\n";
      }
      os << curIndent << INDENT << "}\n";
      os << curIndent << INDENT << " res;\n";
      os << curIndent << "})";
      return anyVector;
    } else if (Def->isSubClassOf("SubRoutine")) {
      auto npattern = Def->getValueAsDag("PatternToMatch");
      if (!npattern)
        PrintFatalError(pattern->getLoc(),
                        Twine("pattern 'PatternToMatch' not defined in ") +
                            resultTree->getAsString());

      auto insts = Def->getValueAsDag("insts");
      if (!insts)
        PrintFatalError(pattern->getLoc(),
                        Twine("pattern 'insts' not defined in ") +
                            resultTree->getAsString());

      os << "({\n";
      os << curIndent << INDENT << "// Computing subroutine " << opName << "\n";
      SmallVector<bool, 1> vectorValued = prepareArgs(
          curIndent + INDENT, os, argPattern, pattern, resultRoot, builder,
          nameToOrdinal, lookup, retidx, origName, newFromOriginal);
      bool anyVector = false;
      for (auto b : vectorValued)
        anyVector |= b;

      VariableSetting nnameToOrdinal;

      if (npattern->getNumArgs() != resultRoot->getNumArgs()) {
        PrintFatalError(pattern->getLoc(),
                        Twine("Attempting to call subroutine '") + opName +
                            " with " + Twine(resultRoot->getNumArgs()) +
                            " args when expected " +
                            Twine(npattern->getNumArgs()) + " " +
                            resultTree->getAsString());
      }

      std::function<void(DagInit *, ArrayRef<unsigned>)> insert =
          [&](DagInit *ptree, ArrayRef<unsigned> prev) {
            unsigned i = 0;
            for (auto tree : ptree->getArgs()) {
              SmallVector<unsigned, 2> next(prev.begin(), prev.end());
              next.push_back(i);
              if (auto dg = dyn_cast<DagInit>(tree))
                insert(dg, next);

              if (ptree->getArgNameStr(i).size()) {
                auto op = (argPattern + "_" + Twine(next[0])).str();
                if (prev.size() > 0) {
                  os << curIndent << INDENT << "Value* local_"
                     << ptree->getArgNameStr(i) << " = ";
                  if (!vectorValued[next[0]]) {
                    os << "gutils->extractMeta(" << builder << ", " << op
                       << ", ArrayRef<unsigned>({";
                    for (unsigned i = 1; i < next.size(); i++) {
                      if (i != 1)
                        os << ", ";
                      os << next[i];
                    }
                    os << "}), \"" << ptree->getArgNameStr(i) << "\");\n";
                  } else {
                    os << "gutils->getWidth() == 1 ? ";

                    os << "gutils->extractMeta(" << builder << ", " << op
                       << ", ArrayRef<unsigned>({";
                    for (unsigned i = 1; i < next.size(); i++) {
                      if (i != 1)
                        os << ", ";
                      os << next[i];
                    }
                    os << "}), \"" << ptree->getArgNameStr(i) << "\")";

                    os << " : UndefValue::get(gutils->getShadowType(getSubType("
                       << op << "->getType(), 0";
                    for (unsigned i = 1; i < next.size(); i++) {
                      os << ", ";
                      os << next[i];
                    }
                    os << ")));\n";
                    os << curIndent << INDENT
                       << "if (gutils->getWidth() != 1)\n";
                    os << curIndent << INDENT << INDENT
                       << "for (size_t i=0; i<gutils->getWidth(); i++)\n";
                    os << curIndent << INDENT << INDENT << INDENT << "local_"
                       << ptree->getArgNameStr(i) << " = " << builder
                       << ".CreateInsertValue(local_" << ptree->getArgNameStr(i)
                       << ", ";

                    os << "gutils->extractMeta(" << builder << ", " << op
                       << ", ArrayRef<unsigned>({(unsigned)i";
                    for (unsigned i = 1; i < next.size(); i++) {
                      os << ", " << next[i];
                    }
                    os << "}), \"" << ptree->getArgNameStr(i)
                       << ".\"+Twine(i)), "
                          "ArrayRef<unsigned>({(unsigned)i}));\n";
                  }
                  op = ("local_" + ptree->getArgNameStr(i)).str();
                }
                nnameToOrdinal.insert(ptree->getArgNameStr(i), op,
                                      vectorValued[next[0]]);
              }
              i++;
            }
          };

      insert(npattern, {});

      initializeNames(curIndent + INDENT, os, insts, "local");

      ArrayRef<unsigned> nretidx{};

      os << curIndent << INDENT;
      bool anyVector2 =
          handle(curIndent + INDENT, argPattern + "_sr", os, pattern, insts,
                 builder, nnameToOrdinal, /*lookup*/ false, nretidx,
                 "<ILLEGAL>", /*newFromOriginal*/ false);
      assert(anyVector == anyVector2);
      os << ";\n";
      os << curIndent << "})";
      return anyVector;

    } else if (Def->isSubClassOf("Inst")) {

      os << "({\n";
      os << curIndent << INDENT << "// Computing " << opName << "\n";
      SmallVector<bool, 1> vectorValued = prepareArgs(
          curIndent + INDENT, os, argPattern, pattern, resultRoot, builder,
          nameToOrdinal, lookup, retidx, origName, newFromOriginal);
      bool anyVector = false;
      for (auto b : vectorValued)
        anyVector |= b;

      bool isCall = opName == "Call" || Def->isSubClassOf("Call");
      bool isIntr = opName == "Intrinsic" || Def->isSubClassOf("Intrinsic");

      if (isCall) {
        getFunction(curIndent + INDENT, os, "callval", "FT", "cconv",
                    Def->getValueInit("func"), origName);
      }

      if (anyVector) {
        os << curIndent << INDENT << "Value *res = nullptr;\n";
        os << curIndent << INDENT
           << "for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";
      }

      os << curIndent << INDENT;
      if (anyVector)
        os << INDENT;
      if (isCall) {
        os << "CallInst *V = ";
      } else if (anyVector) {
        os << "Value *V = ";
      }

      if (isCall) {
        os << "cast<CallInst>(" << builder
           << ".CreateCall(FT, callval, ArrayRef<Value*>({";
      } else if (isIntr) {
        os << builder << ".CreateCall(";
        auto intrName = Def->getValueAsString("name");
        auto intrTypes = Def->getValueAsListInit("types");
        getIntrinsic(os, intrName, intrTypes, argPattern, origName);
        os << ", ArrayRef<Value*>({";
      } else if (opName == "CheckedMul") {
        os << "checkedMul(" << builder << ", ";
      } else if (opName == "CheckedDiv") {
        os << "checkedDiv(" << builder << ", ";
      } else {
        os << builder << ".Create" << opName << "(";
      }
      for (size_t i = 0; i < vectorValued.size(); i++) {
        if (i > 0)
          os << ", ";
        if (vectorValued[i])
          os << "(gutils->getWidth() == 1) ? " << argPattern << "_" << i
             << " : gutils->extractMeta(" << builder << ", " << argPattern
             << "_" << i << ", idx)";
        else
          os << argPattern << "_" << i << "";
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
      if (isCall) {
        os << ")";
      }
      os << ";\n";

      if (isCall) {
        os << curIndent << INDENT;
        if (anyVector)
          os << INDENT;
        os << "V->setDebugLoc(gutils->getNewFromOriginal(" << origName
           << ".getDebugLoc()));"
              "\n";
        os << curIndent << INDENT;
        if (anyVector)
          os << INDENT;
        os << "V->setCallingConv(cconv);\n";
        for (auto *attr : *cast<ListInit>(Def->getValueAsListInit("fnattrs"))) {
          auto attrDef = cast<DefInit>(attr)->getDef();
          auto attrName = attrDef->getValueAsString("name");
          if (attrName == "ReadNone") {
            os << "#if LLVM_VERSION_MAJOR >= 16\n";
            os << curIndent << INDENT;
            if (anyVector)
              os << INDENT;
            os << "V->setOnlyReadsMemory();\n";
            os << "V->setOnlyWritesMemory();\n";
            os << "#elif LLVM_VERSION_MAJOR >= 14\n";
          } else if (attrName == "ReadOnly") {
            os << "#if LLVM_VERSION_MAJOR >= 16\n";
            os << curIndent << INDENT;
            if (anyVector)
              os << INDENT;
            os << "V->setOnlyReadsMemory();\n";
            os << "#elif LLVM_VERSION_MAJOR >= 14\n";
          } else
            os << "#if LLVM_VERSION_MAJOR >= 14\n";
          os << curIndent << INDENT;
          if (anyVector)
            os << INDENT;
          os << "V->addAttributeAtIndex(AttributeList::FunctionIndex, "
                "Attribute::"
             << attrName << ");\n";
          os << "#else \n";

          os << curIndent << INDENT;
          if (anyVector)
            os << INDENT;
          os << "V->addAttribute(AttributeList::FunctionIndex, "
                "Attribute::"
             << attrName << ");\n";
          os << "#endif \n";
        }
      }
      if (anyVector) {
        os << curIndent << INDENT << INDENT
           << "if (gutils->getWidth() == 1) res = "
              "V;\n";
        os << curIndent << INDENT << INDENT << "else {\n";
        os << curIndent << INDENT << INDENT << INDENT
           << "if (idx == 0) res = "
              "UndefValue::get(ArrayType::get(V->getType(), "
              "gutils->getWidth()));\n";
        os << curIndent << INDENT << INDENT << INDENT << "res = " << builder
           << ".CreateInsertValue(res, V, {idx});\n";
        os << curIndent << INDENT << INDENT << "}\n";
        os << curIndent << INDENT "}\n";
        os << curIndent << INDENT << "res;\n";
      } else if (isCall)
        os << curIndent << INDENT << "V;\n";

      os << curIndent << "})";
      return anyVector;
    }
    errs() << *resultRoot << "\n";
  }
  errs() << *resultTree << "\n";
  PrintFatalError(pattern->getLoc(), Twine("unknown operation"));
}

static void emitDerivatives(const RecordKeeper &recordKeeper, raw_ostream &os,
                            ActionType intrinsic) {
  emitSourceFileHeader("Rewriters", os);
  const char *patternNames;
  switch (intrinsic) {
  case CallDerivatives:
    patternNames = "CallPattern";
    break;
  case InstDerivatives:
    patternNames = "InstPattern";
    break;
  case IntrDerivatives:
    patternNames = "IntrPattern";
    break;
  case BinopDerivatives:
    patternNames = "BinopPattern";
    break;
  default:
    assert(0 && "Illegal pattern type");
  }
  const auto &patterns = recordKeeper.getAllDerivedDefinitions(patternNames);

  for (Record *pattern : patterns) {
    DagInit *tree = pattern->getValueAsDag("PatternToMatch");

    DagInit *duals = pattern->getValueAsDag("ArgDuals");

    // Emit RewritePattern for Pattern.
    ListInit *argOps = pattern->getValueAsListInit("ArgDerivatives");

    if (tree->getNumArgs() != argOps->size()) {
      PrintFatalError(pattern->getLoc(),
                      Twine("Defined rule pattern to have ") +
                          Twine(tree->getNumArgs()) +
                          " args but reverse rule array is a list of size " +
                          Twine(argOps->size()));
    }

    std::string origName;
    switch (intrinsic) {
    case CallDerivatives: {
      os << "  if ((";
      bool prev = false;
      for (auto *nameI :
           *cast<ListInit>(pattern->getValueAsListInit("names"))) {
        if (prev)
          os << " ||\n      ";
        os << "funcName == " << cast<StringInit>(nameI)->getAsString() << "";
        prev = true;
      }
      origName = "call";
#if LLVM_VERSION_MAJOR >= 14
      os << ") && call.arg_size() == " << tree->getNumArgs() << " ){\n";
#else
      os << ") && call.getNumArgOperands() == " << tree->getNumArgs()
         << " ){\n";
#endif
      break;
    }
    case IntrDerivatives: {
      bool anyVersion = false;
      for (auto *nameI :
           *cast<ListInit>(pattern->getValueAsListInit("names"))) {
        auto lst = cast<ListInit>(nameI);
        assert(lst->size() >= 1);
        StringRef name = cast<StringInit>(lst->getValues()[0])->getValue();
        if (lst->size() >= 2) {
          auto min = cast<StringInit>(lst->getValues()[1])->getValue();
          int min_int;
          min.getAsInteger(10, min_int);
          if (min.size() != 0 && LLVM_VERSION_MAJOR < min_int)
            continue;
          if (lst->size() >= 3) {
            auto max = cast<StringInit>(lst->getValues()[2])->getValue();
            int max_int;
            max.getAsInteger(10, max_int);
            if (max.size() != 0 && LLVM_VERSION_MAJOR > max_int)
              continue;
          }
        }
        os << " case Intrinsic::" << name << ":\n";
        anyVersion = true;
      }
      if (!anyVersion)
        continue;
      origName = "I";
      os << " {\n";
      os << "    CallInst *const newCall = "
            "cast<CallInst>(gutils->getNewFromOriginal(&"
         << origName << "));\n";
      os << "    IRBuilder<> BuilderZ(newCall);\n";
      os << "    BuilderZ.setFastMathFlags(getFast());\n";
      break;
    }
    case InstDerivatives: {
      auto minVer = pattern->getValueAsInt("minVer");
      auto maxVer = pattern->getValueAsInt("maxVer");
      auto name = pattern->getValueAsString("name");
      if (minVer != 0) {
        if (LLVM_VERSION_MAJOR < minVer)
          continue;
      }
      if (maxVer != 0) {
        if (LLVM_VERSION_MAJOR > maxVer)
          continue;
      }
      os << " case llvm::Instruction::" << name << ":\n";

      origName = "inst";
      os << " {\n";
      os << "    auto mod = inst.getParent()->getParent()->getParent();\n";
      os << "    auto *const newCall = "
            "cast<llvm::Instruction>(gutils->getNewFromOriginal(&"
         << origName << "));\n";
      os << "    IRBuilder<> BuilderZ(newCall);\n";
      os << "    BuilderZ.setFastMathFlags(getFast());\n";
      break;
    }
    case BinopDerivatives: {
      auto minVer = pattern->getValueAsInt("minVer");
      auto maxVer = pattern->getValueAsInt("maxVer");
      auto name = pattern->getValueAsString("name");
      if (minVer != 0) {
        if (LLVM_VERSION_MAJOR < minVer)
          continue;
      }
      if (maxVer != 0) {
        if (LLVM_VERSION_MAJOR > maxVer)
          continue;
      }

      os << " case llvm::Instruction::" << name << ":\n";

      origName = "BO";
      os << " {\n";
      os << "    auto mod = BO.getParent()->getParent()->getParent();\n";
      os << "    auto *const newCall = "
            "cast<llvm::Instruction>(gutils->getNewFromOriginal(&"
         << origName << "));\n";
      os << "    IRBuilder<> BuilderZ(newCall);\n";
      os << "    BuilderZ.setFastMathFlags(getFast());\n";
      break;
    }
    }

    VariableSetting nameToOrdinal;

    std::function<void(DagInit *, ArrayRef<unsigned>)> insert =
        [&](DagInit *ptree, ArrayRef<unsigned> prev) {
          unsigned i = 0;
          for (auto tree : ptree->getArgs()) {
            SmallVector<unsigned, 2> next(prev.begin(), prev.end());
            next.push_back(i);
            if (auto dg = dyn_cast<DagInit>(tree))
              insert(dg, next);

            if (ptree->getArgNameStr(i).size()) {
              auto op =
                  (origName + ".getOperand(" + Twine(next[0]) + ")").str();
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
              nameToOrdinal.insert(ptree->getArgNameStr(i), op, false);
            }
            i++;
          }
        };

    insert(tree, {});

    if (tree->getNameStr().size())
      nameToOrdinal.insert(tree->getNameStr(),
                           (Twine("(&") + origName + ")").str(), false);

    if (intrinsic != BinopDerivatives && intrinsic != InstDerivatives) {
      os << "    if (gutils->knownRecomputeHeuristic.find(&" << origName
         << ") !=\n";
      os << "        gutils->knownRecomputeHeuristic.end()) {\n";
      os << "        if (!gutils->knownRecomputeHeuristic[&" << origName
         << "]) {\n";
      os << "          gutils->cacheForReverse(BuilderZ, newCall,\n";
      os << "                                  getIndex(&" << origName
         << ", "
            "CacheType::Self));\n";
      os << "        }\n";
      os << "    }\n";
    }
    os << "    eraseIfUnused(" << origName << ");\n";

    os << "    if (gutils->isConstantInstruction(&" << origName << "))\n";
    if (intrinsic == IntrDerivatives)
      os << "      return true;\n";
    else
      os << "      return;\n";

    os << "    switch (Mode) {\n";
    os << "      case DerivativeMode::ForwardModeSplit:\n";
    os << "      case DerivativeMode::ForwardMode:{\n";
    os << "        IRBuilder<> Builder2(&" << origName << ");\n";
    os << "        getForwardBuilder(Builder2);\n";
    // TODO

    if (duals->getOperator()->getAsString() ==
            "ForwardFromSummedReverseInternal" ||
        cast<DefInit>(duals->getOperator())
            ->getDef()
            ->isSubClassOf("ForwardFromSummedReverseInternal")) {
      os << "        Value *res = Constant::getNullValue(gutils->getShadowType("
         << origName
         << "."
            "getType()));\n";

      for (auto argOpEn : enumerate(*argOps)) {
        size_t argIdx = argOpEn.index();

        const char *curIndent = "        ";

        if (DagInit *resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
          auto opName = resultRoot->getOperator()->getAsString();
          auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
          if (Def->isSubClassOf("InactiveArgSpec")) {
            if (Def->getValueAsBit("asserting"))
              os << " assert(gutils->isConstantValue(" << origName
                 << ".getOperand(" << argIdx << ")));\n";
            continue;
          }
        }

        os << curIndent << "if (!gutils->isConstantValue(" << origName
           << ".getOperand(" << argIdx << "))) {\n";
        os << curIndent << INDENT << "Value *dif = diffe(" << origName
           << ".getOperand(" << argIdx << "), Builder2);\n";
        os << curIndent << INDENT
           << "Value *arg_diff_tmp = UndefValue::get(res->getType());\n";

        initializeNames(Twine(curIndent) + INDENT, os, argOpEn.value(),
                        "local");
        std::function<void(ArrayRef<unsigned>, Init *)> fwdres =
            [&](ArrayRef<unsigned> idx, Init *ival) {
              if (DagInit *resultTree = dyn_cast<DagInit>(ival)) {
                auto Def = cast<DefInit>(resultTree->getOperator())->getDef();
                if (Def->isSubClassOf("MultiReturn")) {
                  unsigned i = 0;
                  for (auto r : resultTree->getArgs()) {
                    SmallVector<unsigned, 2> next(idx.begin(), idx.end());
                    next.push_back(i);
                    i++;
                    fwdres(next, r);
                  }
                  return;
                }
                os << curIndent << INDENT << "{\n";
                os << curIndent << INDENT << INDENT << "Value *itmp = ";
                ArrayRef<unsigned> retidx{};
                bool vectorValued = handle(
                    Twine(curIndent) + INDENT + INDENT, "fwdarg", os, pattern,
                    resultTree, "Builder2", nameToOrdinal, /*lookup*/ false,
                    retidx, origName, /*newFromOriginal*/ true);
                os << ";\n";
                assert(vectorValued);
                os << curIndent << INDENT << INDENT
                   << "arg_diff_tmp = GradientUtils::recursiveFAdd(Builder2,";
                os << "res, itmp, {";
                {
                  bool seen = false;
                  for (auto i : idx) {
                    if (seen)
                      os << ", ";
                    os << i;
                    seen = true;
                  }
                }

                os << "}, {}, arg_diff_tmp, gutils->getWidth() != 1);\n";
                os << curIndent << INDENT << "}\n";
              } else if (ListInit *lst = dyn_cast<ListInit>(ival)) {
                unsigned i = 0;
                for (auto r : *lst) {
                  SmallVector<unsigned, 2> next(idx.begin(), idx.end());
                  next.push_back(i);
                  i++;
                  fwdres(next, r);
                }
              } else
                PrintFatalError(pattern->getLoc(),
                                Twine("Unknown subinitialization"));
            };
        fwdres({}, argOpEn.value());
        os << curIndent << INDENT << "res = arg_diff_tmp;\n";
        os << "        }\n";
      }
    } else {

      os << "            Value *res = ";
      ArrayRef<unsigned> retidx{};
      bool vectorValued =
          handle("            ", "fwdnsrarg", os, pattern, duals, "Builder2",
                 nameToOrdinal, /*lookup*/ false, retidx, origName,
                 /*newFromOriginal*/ true);
      assert(vectorValued);
      os << ";\n";
    }
    os << "        assert(res);\n";
    os << "        setDiffe(&" << origName << ", res, Builder2);\n";
    os << "        break;\n";
    os << "      }\n";

    os << "      case DerivativeMode::ReverseModeGradient:\n";
    os << "      case DerivativeMode::ReverseModeCombined:{\n";
    os << "        IRBuilder<> Builder2(&" << origName << ");\n";
    os << "        getReverseBuilder(Builder2);\n";
    // TODO vector

    os << "        Value *dif = nullptr;\n";
    bool seen = false;
    for (auto argOpEn : enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();
      if (DagInit *resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
        auto opName = resultRoot->getOperator()->getAsString();
        auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
        if (opName == "InactiveArgSpec" ||
            Def->isSubClassOf("InactiveArgSpec")) {
          if (Def->getValueAsBit("asserting"))
            os << " assert(gutils->isConstantValue(" << origName
               << ".getOperand(" << argIdx << ")));\n";
          continue;
        }
      }

      os << "        ";
      if (seen)
        os << "} else ";
      seen = true;
      os << "if (!dif && !gutils->isConstantValue(" << origName
         << ".getOperand(" << argIdx << "))) {\n";
      DagInit *resultTree = cast<DagInit>(argOpEn.value());
      if (hasDiffeRet(resultTree)) {
        os << "          dif = diffe(&" << origName << ", Builder2);\n";
        os << "          setDiffe(&" << origName
           << ", "
              "Constant::getNullValue(gutils->getShadowType("
           << origName
           << ".getType())), "
              "Builder2);\n";
      }
    }
    if (seen)
      os << "        }\n";

    std::function<void(size_t, ArrayRef<unsigned>, Init *)> revres =
        [&](size_t argIdx, ArrayRef<unsigned> idx, Init *ival) {
          if (DagInit *resultTree = dyn_cast<DagInit>(ival)) {
            auto Def = cast<DefInit>(resultTree->getOperator())->getDef();
            if (Def->isSubClassOf("MultiReturn")) {
              unsigned i = 0;
              for (auto r : resultTree->getArgs()) {
                SmallVector<unsigned, 1> next(idx.begin(), idx.end());
                next.push_back(i);
                revres(argIdx, next, r);
                i++;
              }
              return;
            }
            const char *curIndent = "          ";
            os << curIndent << "{\n";
            os << curIndent << INDENT << "Value *tmp = ";
            bool vectorValued =
                handle(Twine(curIndent) + INDENT, "revarg", os, pattern,
                       resultTree, "Builder2", nameToOrdinal, /*lookup*/ true,
                       idx, origName, /*newFromOriginal*/ true);
            os << ";\n";

            os << curIndent << INDENT
               << "Value *out = "
                  "UndefValue::get(gutils->getShadowType("
               << origName << ".getOperand(" << argIdx << ")->getType()));\n";

            os << curIndent << INDENT
               << "for(unsigned int idx=0, W=gutils->getWidth(); "
                  "idx<W; idx++) {\n";

            os << curIndent << INDENT << INDENT
               << "Value *prev = toadd ? (gutils->getWidth() == "
                  "1 ? toadd : gutils->extractMeta(Builder2, toadd, idx)) : "
                  "nullptr;\n";
            os << curIndent << INDENT << INDENT << "Value *next = tmp;\n";
            if (vectorValued)
              os << curIndent << INDENT << INDENT
                 << "if (gutils->getWidth() > 1) next = "
                    "gutils->extractMeta(Builder2, next, idx);\n";
            os << curIndent << INDENT << INDENT
               << "if (prev) next = Builder2.CreateFAdd(prev, "
                  "next);\n";
            os << curIndent << INDENT << INDENT
               << "out = (gutils->getWidth() > 1) ? "
                  "Builder2.CreateInsertValue(out, next, idx) : next;\n";
            os << curIndent << INDENT << "}\n";
            os << curIndent << INDENT << "toadd = out;\n";

            os << curIndent << "}\n";

          } else if (ListInit *lst = dyn_cast<ListInit>(ival)) {
            unsigned i = 0;
            for (auto elem : *lst) {
              SmallVector<unsigned, 1> next(idx.begin(), idx.end());
              next.push_back(i);
              revres(argIdx, next, elem);
              i++;
            }
          } else
            assert(0);
        };

    for (auto argOpEn : enumerate(*argOps)) {
      size_t argIdx = argOpEn.index();
      if (DagInit *resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
        auto opName = resultRoot->getOperator()->getAsString();
        auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
        if (opName == "InactiveArgSpec" ||
            Def->isSubClassOf("InactiveArgSpec")) {
          continue;
        }
      }

      const char *curIndent = "        ";
      os << curIndent << "if (!gutils->isConstantValue(" << origName
         << ".getOperand(" << argIdx << "))) {\n";
      initializeNames(Twine(curIndent) + INDENT, os, argOpEn.value(), "local");
      os << curIndent << INDENT << "Value *toadd = nullptr;\n";
      revres(argIdx, {}, argOpEn.value());

      os << curIndent << INDENT << "if (toadd) addToDiffe(" << origName
         << ".getOperand(" << argIdx << "), toadd";
      os << ", Builder2, " << origName << ".getOperand(" << argIdx
         << ")->getType());\n";
      os << curIndent << "}\n";
    }

    os << "        break;\n";
    os << "      }\n";

    os << "      case DerivativeMode::ReverseModePrimal:{\n";
    // TODO
    os << "        break;\n";
    os << "      }\n";
    os << "    }\n";

    if (intrinsic == IntrDerivatives)
      os << "    return true;\n  }\n";
    else
      os << "    return;\n  }\n";
  }
}

#include "blasDeclUpdater.h"
#include "blasDiffUseUpdater.h"
#include "blasTAUpdater.h"

static bool EnzymeTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case CallDerivatives:
  case InstDerivatives:
  case IntrDerivatives:
  case BinopDerivatives:
    emitDerivatives(records, os, action);
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
    errs() << "unknown tablegen action!\n";
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

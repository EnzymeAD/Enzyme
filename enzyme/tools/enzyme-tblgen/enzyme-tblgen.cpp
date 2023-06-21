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
#include "datastructures.h"

using namespace llvm;

enum ActionType {
  GenDerivatives,
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
           cl::values(clEnumValN(GenDerivatives, "gen-derivatives",
                                 "Generate instruction derivative")));

bool hasDiffeRet(Init *resultTree) {
  if (DagInit *resultRoot = dyn_cast<DagInit>(resultTree)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "DiffeRetIndex" || Def->isSubClassOf("DiffeRetIndex")) {
      return true;
    }
    for (auto arg : resultRoot->getArgs()) {
      if (hasDiffeRet(arg))
        return true;
    }
  }
  return false;
}

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
  std::string patternNames;
  switch (intrinsic) {
  case GenDerivatives:
    patternNames = "CallPattern";
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
    case GenDerivatives: {
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
      os << "    auto mod = call.getParent()->getParent()->getParent();\n";
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
      os << "    auto mod = I.getParent()->getParent()->getParent();\n";
      os << "    auto called = cast<CallInst>(&" << origName
         << ")->getCalledFunction();\n";
      os << "    CallInst *const newCall = "
            "cast<CallInst>(gutils->getNewFromOriginal(&"
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

    if (intrinsic != BinopDerivatives) {
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

      os << "    eraseIfUnused(" << origName << ");\n";
    }

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

static void checkBlasCallsInDag(const RecordKeeper &RK,
                                ArrayRef<Record *> blasPatterns,
                                StringRef blasName, const DagInit *toSearch) {

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
    if (!CalledBlas)
      errs() << " opName: " << opName << "\n";
    assert(CalledBlas);
    auto expectedNumArgs =
        CalledBlas->getValueAsDag("PatternToMatch")->getNumArgs();
    if (expectedNumArgs != numArgs) {
      errs() << "failed calling " << opName << " in the derivative of "
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
                           ArrayRef<Record *> blasPatterns) {
  for (auto &&pattern : blasPatterns) {
    ListInit *argOps = pattern->getValueAsListInit("ArgDerivatives");
    // for each possibly active parameter
    for (auto argOp : *argOps) {
      DagInit *resultRoot = cast<DagInit>(argOp);
      checkBlasCallsInDag(RK, blasPatterns, pattern->getName(), resultRoot);
    }
  }
}

// handleBLAS is called in the AdjointGenerator.h
void emit_handleBLAS(ArrayRef<TGPattern> blasPatterns, raw_ostream &os) {
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
  for (auto &&pattern : blasPatterns) {
    bool hasNonInactive = false;
    for (Rule rule : pattern.getRules()) {
      const auto ruleDag = rule.getRuleDag();
      const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
      if (Def->isSubClassOf("MagicInst") && Def->getName() == "inactive")
        continue;
      hasNonInactive = true;
      break;
    }
    if (!hasNonInactive)
      continue;
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

void emit_beginning(const TGPattern &pattern, raw_ostream &os) {
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

void emit_free_and_ending(const TGPattern &pattern, raw_ostream &os) {
  os << "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
     << "      Mode == DerivativeMode::ReverseModeGradient ||\n"
     << "      Mode == DerivativeMode::ForwardModeSplit) {\n"
     << "    if (shouldFree()) {\n";

  auto nameVec = pattern.getArgNames();
  auto typeMap = pattern.getArgTypeMap();
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty == ArgType::vincData || ty == ArgType::mldData) {
      auto name = nameVec[i];
      os << "      if (cache_" << name << ") {\n"
         << "        CreateDealloc(Builder2, free_" << name << ");\n"
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

void emit_helper(const TGPattern &pattern, raw_ostream &os) {
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
    ArgType ty = argTypeMap.lookup(i);
    if (ty == ArgType::trans) {
      //      os << "  assert(is_normal(BuilderZ, arg_" << name << "));\n";
    }
    if (std::count(actArgs.begin(), actArgs.end(), i)) {
      os << "  const bool active_" << name
         << " = !gutils->isConstantValue(orig_" << name << ");\n";
    }
    os << "\n";
  }

  bool anyActive = false;
  for (size_t i = 0; i < nameVec.size(); i++) {
    ArgType ty = argTypeMap.lookup(i);
    if (ty == ArgType::fp) {
      anyActive = true;
    }
  }

  if (anyActive) {
    os << "  int num_active_fp = 0;\n";
    for (size_t i = 0; i < nameVec.size(); i++) {
      ArgType ty = argTypeMap.lookup(i);
      if (ty == ArgType::fp) {
        os << "  if (active_" << nameVec[i] << ")\n"
           << "    num_active_fp++;\n";
      }
    }
  }

  for (auto name : enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto ty = argTypeMap.lookup(name.index());
    if (ty == ArgType::vincData || ty == ArgType::mldData) {
      os << "  const bool julia_decl = !type_" << name.value()
         << "->isPointerTy();\n";
      return;
    }
  }
  PrintFatalError("Blas function without vector and matrix?");
}

void emit_scalar_types(const TGPattern &pattern, raw_ostream &os) {
  // We only look at the type of the first integer showing up.
  // This allows to learn if we use Fortran abi (byRef) or cabi
  std::string name = "";
  bool foundInt = false;

  auto inputTypes = pattern.getArgTypeMap();
  auto nameVec = pattern.getArgNames();

  for (auto val : inputTypes) {
    if (val.second == ArgType::len) {
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

void extract_scalar(StringRef name, StringRef elemTy, raw_ostream &os) {
  os << "      if (cache_" << name << ") {\n"
     << "        arg_" << name << " = (cacheTypes.size() == 1)\n"
     << "                    ? cacheval\n"
     << "                    : Builder2.CreateExtractValue(cacheval, "
     << "{cacheidx}, \"tape.ext." << name << "\");\n"
     << "        auto alloc = allocationBuilder.CreateAlloca(" << elemTy
     << ", nullptr, \"byref." << name << "\");\n"
     << "        Builder2.CreateStore(arg_" << name << ", alloc);\n"
     << "        arg_" << name << " = Builder2.CreatePointerCast(\n"
     << "            alloc, type_" << name << ", \"cast." << name << "\");\n"
     << "        cacheidx++;\n"
     << "      }\n"
     << "\n";
}

void extract_mat_or_vec(StringRef name, raw_ostream &os) {
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
     << "        } else if (arg_" << name << "->getType() != type_" << name
     << "){\n"
     << "          arg_" << name << " = Builder2.CreatePointerCast(arg_" << name
     << ", type_" << name << ");\n"
     << "        }\n"
     << "        cacheidx++;\n"
     << "      }\n";
}

void emit_extract_calls(const TGPattern &pattern, raw_ostream &os) {
  const auto typeMap = pattern.getArgTypeMap();
  const auto nameVec = pattern.getArgNames();
  const auto argUsers = pattern.getArgUsers();

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
    if (ty == ArgType::len || ty == ArgType::vincInc || ty == ArgType::mldLD) {
      extract_scalar(name, "intType", os);
    } else if (ty == ArgType::fp) {
      extract_scalar(name, "fpType", os);
    } else if (ty == ArgType::trans) {
      // we are in the byRef branch and trans only exist in lv23.
      // So just unconditionally asume that no layout exist and use i-1
      extract_scalar(name, "charType", os);
    }
  }
  os << "    }\n";

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != ArgType::mldData)
      continue;
    auto name = nameVec[i];
    extract_mat_or_vec(name, os);
  }

  // If we cached matrix or vector X, then we did that in a dense form.
  // Therefore, we overwrite the related inc_X to be 1 and ld_X to be = m
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != ArgType::vincData)
      continue;
    auto name = nameVec[i];
    const auto vecPosition = i;
    const auto vecUsers = argUsers.lookup(vecPosition);
    const auto incName = nameVec[i + 1];
    extract_mat_or_vec(name, os);
    os << "      if (cache_" << name << ") {\n"
       << "        arg_" << incName << " = ConstantInt::get(intType, 1);\n"
       << "       if (byRef) {\n"
       << "         auto alloc = allocationBuilder.CreateAlloca(intType, "
          "nullptr, \"byref."
       << incName << "\");\n"
       << "         Builder2.CreateStore(arg_" << incName << ", alloc);\n"
       << "         arg_" << incName << " = Builder2.CreatePointerCast(\n"
       << "             alloc, type_" << incName << ", \"cast." << incName
       << "\");\n"
       << "      }\n"
       << " }\n";
  }

  os << "  } else {\n"
     << "\n";

  for (size_t i = 0; i < nameVec.size(); i++) {
    if (typeMap.lookup(i) != ArgType::vincData)
      continue;
    auto vecName = nameVec[i];
    os << "    if (type_" << vecName << "->isIntegerTy())\n"
       << "      arg_" << vecName << " = Builder2.CreatePtrToInt(arg_"
       << vecName << ", type_" << vecName << ");\n";
  }

  os << "  }\n";
}

// Will be used by Julia
SmallString<80> ValueType_helper(const TGPattern &pattern, size_t actPos) {
  const auto nameVec = pattern.getArgNames();
  const auto typeMap = pattern.getArgTypeMap();
  SmallString<80> valueTypes{};

  // start with 1 since layout is only used for cblas (!byRef)
  for (size_t pos = 1; pos < nameVec.size();) {
    auto name = nameVec[pos];
    auto ty = typeMap.lookup(pos);

    if (pos > 1) {
      valueTypes.append(", ");
    }

    if (ty == ArgType::len) {
      valueTypes.append("ValueType::Both");
    } else if (ty == ArgType::fp) {
      auto floatName = nameVec[pos];
      if (pos == actPos) {
        valueTypes.append("ValueType::Both");
      } else {
        valueTypes.append((Twine("cache_") + floatName +
                           " ? ValueType::Both : ValueType::Both")
                              .str());
      }
    } else if (ty == ArgType::vincData) {
      const auto nextName = nameVec[pos + 1];
      const auto nextTy = typeMap.lookup(pos + 1);
      assert(nextTy == ArgType::vincInc);
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
      // llvm::errs() << "type: " << ty << "\n";
      // PrintFatalError("Unhandled type!");
    }
    pos++;
  }
  return valueTypes;
}

// TODO: think about how to handle nested rules which aren't simple calling
// another BLAS fnc.

size_t fwd_call_args(const TGPattern &pattern, size_t actArg,
                     SmallString<40> &result) {
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
    auto ty = typeMap.lookup(pos);
    if (ty == ArgType::len) {
      result.append((Twine("arg_") + name).str());
    } else if (ty == ArgType::fp) {
      if (pos == actArg) {
        result.append((Twine("d_") + name).str());
      } else {
        result.append((Twine("arg_") + name).str());
        // result.append((Twine("fp_") + name).str());
      }
    } else if (ty == ArgType::vincData) {
      auto nextName = nameVec[pos + 1];
      // get the position of the argument in the primary blas call
      auto nextArgPosition = nameMap.lookup(nextName);
      // and based on that get the fp/int + scalar/vector type
      auto typeOfNextArg = typeMap.lookup(nextArgPosition);
      assert(typeOfNextArg == ArgType::vincInc);
      if (pos == actArg) {
        result.append((Twine("d_") + name + ", true_" + nextName).str());
      } else {
        result.append((Twine("arg_") + name + ", arg_" + nextName).str());
      }
      pos++; // extra ++ due to also handling vincInc
    } else if (ty == ArgType::vincInc) {
      // might come without vincData, e.g. after DiffeRet
      result.append(name);
    } else if (ty == ArgType::mldData) {
      auto nextName = nameVec[pos + 1];
      // get the position of the argument in the primary blas call
      auto nextArgPosition = nameMap.lookup(nextName);
      // and based on that get the fp/int + scalar/vector type
      auto nextTy = typeMap.lookup(nextArgPosition);
      assert(nextTy == ArgType::mldLD);
      if (pos == actArg) {
        result.append((Twine("d_") + name + ", true_" + nextName).str());
      } else {
        result.append((Twine("arg_") + name + ", arg_" + nextName).str());
      }
      pos++; // extra ++ due to also handling mldLD
    } else if (ty == ArgType::mldLD) {
      // might come without mldData, e.g. after DiffeRet
      // coppied from vincInc, but should verify if actually needed
      result.append((Twine("arg_") + name).str());
    } else if (ty == ArgType::cblas_layout) {
      // layout is only allowed as first type (which we skipped)
      errs() << "name: " << name << " typename: " << ty
             << " only allowed as first arg!\n";
      llvm_unreachable("layout only allowed as first type!\n");
    } else if (ty == ArgType::trans || ty == ArgType::diag ||
               ty == ArgType::uplo || ty == ArgType::side) {
      result.append((Twine("arg_") + name).str());
    } else {
      errs() << "name: " << name << " typename: " << ty << "\n";
      llvm_unreachable("unimplemented input type!\n");
    }
    pos++;
  }

  // return the size - 1 due to only using the cblas_layout in the !byRef case
  return nameVec.size() - startArg;
}

void emit_fwd_rewrite_rules(const TGPattern &pattern, raw_ostream &os) {
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
    if (ty == ArgType::vincData || ty == ArgType::mldData) {
      const auto name = nameVec[inputType.first];
      os << "    Value *d_" << name << " = active_" << name << "\n"
         << "     ? gutils->invertPointerM(orig_" << name << ", Builder2)\n"
         << "     : nullptr;\n";
    }
    if (ty == ArgType::fp) {
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
    auto dcallArgs = SmallString<40>();
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

void emit_deriv_fnc(const StringMap<TGPattern> &patternMap, Rule &rule,
                    StringSet<> &handled, raw_ostream &os) {
  const auto ruleDag = rule.getRuleDag();
  const auto typeMap = rule.getArgTypeMap();
  const auto opName = ruleDag->getOperator()->getAsString();
  const auto nameMap = rule.getArgNameMap();
  const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
  if (Def->isSubClassOf("b")) {
    const auto dfnc_name = Def->getValueAsString("s");
    if (patternMap.find(dfnc_name) == patternMap.end()) {
      PrintFatalError("calling unknown Blas function");
    }
    TGPattern calledPattern = patternMap.find(dfnc_name)->getValue();
    bool derivlv23 = calledPattern.isBLASLevel2or3();
    DenseSet<size_t> mutableArgs = calledPattern.getMutableArgs();

    if (handled.find(dfnc_name) != handled.end())
      return;
    else
      handled.insert(dfnc_name);

    // insert arg types based on .td file
    std::string typeString = "";
    bool first = true;
    for (size_t i = 0; i < ruleDag->getNumArgs(); i++) {
      Init *subArg = ruleDag->getArg(i);
      if (DefInit *def = dyn_cast<DefInit>(subArg)) {
        const auto Def = def->getDef();
        std::string typeToAdd = "";
        if (Def->isSubClassOf("DiffeRetIndex")) {
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
          typeToAdd =
              "byRef ? (Type*)PointerType::getUnqual(fpType) : (Type*)fpType";
        } else if (Def->isSubClassOf("Char")) {
          typeToAdd = "byRef ? (Type*)PointerType::getUnqual(charType) : "
                      "(Type*)charType";
        } else if (Def->isSubClassOf("ConstantInt")) {
          typeToAdd =
              "byRef ? (Type*)PointerType::getUnqual(intType) : (Type*)intType";
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
            if (!first)
              typeString += ", ";
            typeString += (Twine("type_") + Dag->getArgNameStr(1)).str();
            first = false;
            continue;
          } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "ld") {
            if (!first)
              typeString += ", ";
            //(ld $A, $transa, $lda, $m, $k)
            // Either of 2,3,4 would work
            typeString += (Twine("type_") + Dag->getArgNameStr(2)).str();
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
        typeString +=
            (Twine(first ? "" : ", ") + Twine("type_") + argStr).str();
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
  } else if (Def->isSubClassOf("DiffeRetIndex")) {
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
void rev_call_args(StringRef argName, Rule &rule, size_t actArg,
                   raw_ostream &os) {

  const auto nameMap = rule.getArgNameMap();
  const auto typeMap = rule.getArgTypeMap();
  const auto ruleDag = rule.getRuleDag();
  const size_t numArgs = ruleDag->getNumArgs();
  const size_t startArg = rule.isBLASLevel2or3() ? 1 : 0;

  os << "        Value *" << argName << "[" << (numArgs - startArg) << "] = {";

  // just replace argOps with rule
  for (size_t pos = startArg; pos < numArgs;) {
    if (pos > startArg) {
      os << ", ";
    }

    auto arg = ruleDag->getArg(pos);
    if (auto Dag = dyn_cast<DagInit>(arg)) {
      auto Def = cast<DefInit>(Dag->getOperator())->getDef();

      if (Def->isSubClassOf("MagicInst") && Def->getName() == "Rows") {
        auto tname = Dag->getArgNameStr(0);
        auto rname = Dag->getArgNameStr(1);
        auto cname = Dag->getArgNameStr(2);
        os << "get_blas_row(Builder2, arg_transposed_" << tname << ", arg_"
           << rname << ", arg_" << cname << ", byRef)";
      } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "ld") {
        assert(Dag->getNumArgs() == 5);
        //(ld $A, $transa, $lda, $m, $k)
        const auto transName = Dag->getArgNameStr(1);
        const auto ldName = Dag->getArgNameStr(2);
        const auto dim1Name = Dag->getArgNameStr(3);
        const auto dim2Name = Dag->getArgNameStr(4);
        const auto matName = Dag->getArgNameStr(0);
        os << "get_cached_mat_width(Builder2, "
           << "arg_" << transName << ", arg_" << ldName << ", arg_" << dim1Name
           << ", arg_" << dim2Name << ", cache_" << matName << ", byRef)";
      } else {
        errs() << Def->getName() << "\n";
        PrintFatalError("Dag/Def that isn't a DiffeRet!");
      }
    } else if (DefInit *DefArg = dyn_cast<DefInit>(arg)) {
      auto Def = DefArg->getDef();
      if (Def->isSubClassOf("DiffeRetIndex")) {
        os << "dif";
      } else if (Def->isSubClassOf("adj")) {
        auto name = Def->getValueAsString("name");
        os << "d_" << name;
      } else if (Def->isSubClassOf("input")) {
        auto name = Def->getValueAsString("name");
        os << "arg_" << name;
      } else if (Def->isSubClassOf("MagicInst")) {
        errs() << "MagicInst\n";
      } else if (Def->isSubClassOf("Constant")) {
        auto val = Def->getValueAsString("value");
        os << "to_blas_callconv(Builder2, ConstantFP::get(fpType, " << val
           << "), byRef, nullptr, allocationBuilder, \"constant.fp." << val
           << "\")";
      } else if (Def->isSubClassOf("Char")) {
        auto val = Def->getValueAsString("value");
        os << "to_blas_callconv(Builder2, ConstantInt::get(charType, '" << val
           << "'), byRef, nullptr, allocationBuilder, \"constant.char." << val
           << "\")";
      } else if (Def->isSubClassOf("ConstantInt")) {
        auto val = Def->getValueAsInt("value");
        os << "to_blas_callconv(Builder2, ConstantInt::get(intType, " << val
           << "), byRef, nullptr, allocationBuilder, \"constant.int." << val
           << "\")";
      } else if (Def->isSubClassOf("transpose")) {
        auto name = Def->getValueAsString("name");
        os << "arg_transposed_" << name;
      } else {
        errs() << Def->getName() << "\n";
        PrintFatalError("Def that isn't a DiffeRet!");
      }
    } else {
      auto name = ruleDag->getArgNameStr(pos);
      assert(name != "");
      // get the position of the argument in the primary blas call
      if (nameMap.count(name) != 1) {
        errs() << "couldn't find name: " << name << "\n";
        PrintFatalError("arg not in nameMap!");
      }
      assert(nameMap.count(name) == 1);
      auto argPosition = nameMap.lookup(name);
      // and based on that get the fp/int + scalar/vector type
      auto ty = typeMap.lookup(argPosition);

      // Now we create the adj call args through concating type and primal name
      if (ty == ArgType::len) {
        os << "arg_" << name;
      } else if (ty == ArgType::fp) {
        if (argPosition == actArg) {
          os << "d_" << name;
        } else {
          os << "arg_" << name;
        }
      } else if (ty == ArgType::vincData) {
        auto nextName = ruleDag->getArgNameStr(pos + 1);
        // get the position of the argument in the primary blas call
        auto nextArgPosition = nameMap.lookup(nextName);
        // and based on that get the fp/int + scalar/vector type
        auto typeOfNextArg = typeMap.lookup(nextArgPosition);
        assert(typeOfNextArg == ArgType::vincInc);
        if (argPosition == actArg) {
          // shadow d_<X> wasn't overwritten or cached, so use true_inc<X>
          // since arg_inc<X> was set to 1 if arg_<X> was cached
          os << "d_" << name << ", true_" << nextName;
        } else {
          os << "arg_" << name << ", arg_" << nextName;
        }
        pos++; // extra ++ due to also handling vincInc
      } else if (ty == ArgType::vincInc) {
        // might come without vincData, e.g. after DiffeRet
        os << "arg_" << name;
      } else if (ty == ArgType::mldData) {
        auto nextName = ruleDag->getArgNameStr(pos + 1);
        // get the position of the argument in the primary blas call
        auto nextArgPosition = nameMap.lookup(nextName);
        // and based on that get the fp/int + scalar/vector type
        auto nextTy = typeMap.lookup(nextArgPosition);
        if (pos == actArg) {
          assert(nextTy == ArgType::mldLD);
          os << "d_" << name << ", true_" << nextName;
          pos++; // extra ++ due to also handling mldLD
        } else {
          // if this matrix got cached, we need more complex logic
          // to determine the next arg. Thus handle it once we reach it
          os << "arg_" << name;
        }
      } else if (ty == ArgType::mldLD) {
        auto prevArg = ruleDag->getArg(pos - 1);
        if (DefInit *DefArg = dyn_cast<DefInit>(prevArg)) {
          if (DefArg->getDef()->isSubClassOf("adj")) {
            // all ok, single LD after shadow of mat
            // use original ld, since shadow is never cached
            os << "arg_" << name;
          } else {
            errs() << rule.to_string() << "\n";
            PrintFatalError("sholdn't be hit??\n");
          }
        } else {
          errs() << rule.to_string() << "\n";
          PrintFatalError("sholdn't be hit??\n");
        }
      } else if (ty == ArgType::trans) {
        os << "arg_" << name;
      } else {
        errs() << "name: " << name << " typename: " << ty << "\n";
        llvm_unreachable("unimplemented input type!\n");
      }
    }
    pos++;
  }
  os << "};\n";
}

void emit_rev_rewrite_rules(const StringMap<TGPattern> &patternMap,
                            TGPattern &pattern, raw_ostream &os) {

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
        if (Def->isSubClassOf("DiffeRetIndex")) {
          hasDiffeRetVal = true;
        }
      }
    }
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "DiffeRetIndex" || Def->isSubClassOf("DiffeRetIndex")) {
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
     << "      alloc = allocationBuilder.CreateAlloca(fpType, nullptr, "
        "\"ret\");\n"
     << "    }\n\n";
  if (hasDiffeRetVal) {
    os << "    Value *dif = diffe(&call, Builder2);\n";
  }

  // We only emit one derivcall per blass call type.
  // This verifies that we don't end up with multiple declarations.
  StringSet handled{};
  for (auto rule : rules) {
    emit_deriv_fnc(patternMap, rule, handled, os);
  }

  for (size_t i = 0; i < nameVec.size(); i++) {
    const auto name = nameVec[i];
    const auto ty = typeMap.lookup(i);
    if (ty == ArgType::vincData || ty == ArgType::mldData) {
      os << "    Value *d_" << name << " = active_" << name << "\n"
         << "     ? lookup(gutils->invertPointerM(orig_" << name
         << ", Builder2), Builder2)\n"
         << "     : nullptr;\n";
    } else if (ty == ArgType::fp) {
      os << "    Value *d_" << name << " = UndefValue::get(fpType);\n";
    }
  }

  // We need to lookup all args which we haven't cached or overwritten and which
  // are required.
  for (size_t i = 0; i < nameVec.size(); i++) {
    const auto name = nameVec[i];
    const auto ty = typeMap.lookup(i);
    if (ty == ArgType::len || ty == ArgType::fp || ty == ArgType::vincData ||
        ty == ArgType::mldData || ty == ArgType::trans || ty == ArgType::uplo ||
        ty == ArgType::diag) {
      os << "    if (!cache_" << name << " && need_" << name << ")\n"
         << "      arg_" << name << " = lookup(arg_" << name
         << ", Builder2);\n";
    } else if (ty == ArgType::vincInc) {
      // extra handling, because if we cache a vec we overwrite the inc
      const auto prevTy = typeMap.lookup(i - 1);
      assert(prevTy == ArgType::vincData);
      const auto vecName = nameVec[i - 1];
      os << "    if (!(cache_" << name << " || cache_" << vecName
         << ") && need_" << name << ")\n"
         << "      arg_" << name << " = lookup(arg_" << name
         << ", Builder2);\n";
    } else if (ty == ArgType::mldLD) {
      // extra handling, because if we cache a mat we overwrite the ld
      const auto prevTy = typeMap.lookup(i - 1);
      assert(prevTy == ArgType::mldData);
      const auto matName = nameVec[i - 1];
      os << "    if (!(cache_" << name << " || cache_" << matName
         << ") && need_" << name << ")\n"
         << "      arg_" << name << " = lookup(arg_" << name
         << ", Builder2);\n";
    }
  }

  // now we can use it to transpose our trans arguments if they exist
  for (size_t i = (lv23 ? 1 : 0); i < nameVec.size(); i++) {
    auto name = nameVec[i];
    if (typeMap.lookup(i) == ArgType::trans) {
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
    if (ty == ArgType::fp)
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
    const auto valueTypes = ValueType_helper(pattern, actArg);
    const auto opName = ruleDag->getOperator()->getAsString();
    const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
    if (Def->isSubClassOf("DiffeRetIndex")) {
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
      os << "      if (" << actCondition << ") {\n";
      rev_call_args("args1", rule, actArg, os);
      os << "        const auto Defs = gutils->getInvertedBundles(&call, {"
         << valueTypes << "}, Builder2, /* lookup */ true);\n";

      if (ty == ArgType::fp) {
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
      errs() << Def->getName() << "\n";
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
    if (ty == ArgType::fp)
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

  SmallVector<TGPattern, 8> newBlasPatterns;
  StringMap<TGPattern> patternMap;
  for (auto &&pattern : blasPatterns) {
    auto parsedPattern = TGPattern(pattern);
    newBlasPatterns.push_back(TGPattern(pattern));
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

  for (auto &&newPattern : newBlasPatterns) {
    bool hasNonInactive = false;
    for (Rule rule : newPattern.getRules()) {
      const auto ruleDag = rule.getRuleDag();
      const auto Def = cast<DefInit>(ruleDag->getOperator())->getDef();
      if (Def->isSubClassOf("MagicInst") && Def->getName() == "inactive")
        continue;
      hasNonInactive = true;
      break;
    }
    if (!hasNonInactive)
      continue;

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
#include "blasDiffUseUpdater.h"
#include "blasTAUpdater.h"

static bool EnzymeTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case GenDerivatives:
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

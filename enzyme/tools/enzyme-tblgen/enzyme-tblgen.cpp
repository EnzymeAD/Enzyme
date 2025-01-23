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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "blas-tblgen.h"
#include "caching.h"
#include "datastructures.h"
#include "enzyme-tblgen.h"

using namespace llvm;

static inline bool startsWith(llvm::StringRef string, llvm::StringRef prefix) {
#if LLVM_VERSION_MAJOR >= 18
  return string.starts_with(prefix);
#else
  return string.startswith(prefix);
#endif // LLVM_VERSION_MAJOR
}

static inline bool endsWith(llvm::StringRef string, llvm::StringRef suffix) {
#if LLVM_VERSION_MAJOR >= 18
  return string.ends_with(suffix);
#else
  return string.endswith(suffix);
#endif // LLVM_VERSION_MAJOR
}

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
           cl::values(clEnumValN(MLIRDerivatives, "gen-mlir-derivatives",
                                 "Generate MLIR derivative")),
           cl::values(clEnumValN(CallDerivatives, "gen-call-derivatives",
                                 "Generate call derivative")));

void getFunction(const Twine &curIndent, raw_ostream &os, StringRef callval,
                 StringRef FT, StringRef cconv, const Init *func,
                 StringRef origName) {
  if (auto resultRoot = dyn_cast<DagInit>(func)) {
    auto opName = resultRoot->getOperator()->getAsString();
    auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
    if (opName == "SameFunc" || Def->isSubClassOf("SameFunc")) {
      os << curIndent << "auto " << callval << " = cast<CallInst>(&" << origName
         << ")->getCalledOperand();\n";
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
      os << ", " << FT
         << ", called->getAttributes().removeFnAttribute(called->getContext(), "
            "\"enzymejl_needs_restoration\")).getCallee();\n";
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
      os << ", " << FT
         << ", called->getAttributes().removeFnAttribute(called->getContext(), "
            "\"enzymejl_needs_restoration\")).getCallee();\n";
      os << curIndent << "auto " << cconv << " = cast<CallInst>(&" << origName
         << ")->getCallingConv();\n";
      return;
    }
    if (opName == "ArgAsRetTypesFunc" ||
        Def->isSubClassOf("ArgAsRetTypesFunc")) {
      os << curIndent << "auto " << FT << "_old = cast<CallInst>(&" << origName
         << ")->getFunctionType();\n";
      os << curIndent << "auto " << FT << " = FunctionType::get(" << FT
         << "_old->params()[0], " << FT << "_old->params(), " << FT
         << "_old->isVarArg());\n";
      os << curIndent << "auto " << callval
         << " = gutils->oldFunc->getParent()->getOrInsertFunction(";
      os << Def->getValueInit("name")->getAsString();
      os << ", " << FT
         << ", called->getAttributes().removeFnAttribute(called->getContext(), "
            "\"enzymejl_needs_restoration\")).getCallee();\n";
      os << curIndent << "auto " << cconv << " = cast<CallInst>(&" << origName
         << ")->getCallingConv();\n";
      return;
    }
  }
  assert(0 && "Unhandled function");
}
void getIntrinsic(raw_ostream &os, StringRef intrName, const ListInit *typeInit,
                  const Twine &argStr, StringRef origName) {
  os << "getIntrinsicDeclaration(mod, Intrinsic::" << intrName
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

void initializeNames(const Twine &curIndent, raw_ostream &os,
                     const Init *resultTree, const Twine &prefix,
                     ActionType intrinsic) {
  if (auto resultRoot = dyn_cast<DagInit>(resultTree)) {
    for (size_t i = 0; i < resultRoot->arg_size(); i++) {
      auto arg = resultRoot->getArg(i);
      auto name = resultRoot->getArgName(i);
      if (isa<UnsetInit>(arg) && name) {
        continue;
      }
      if (name) {
        auto namev = name->getAsUnquotedString();
        if (intrinsic == MLIRDerivatives)
          os << curIndent << "mlir::Value " << prefix << "_" + namev
             << " = nullptr;\n";
        else
          os << curIndent << "llvm::Value *" << prefix << "_" + namev
             << " = nullptr;\n";
      }
      initializeNames(curIndent, os, arg, prefix, intrinsic);
    }
  } else if (auto lst = dyn_cast<ListInit>(resultTree)) {
    for (auto elem : *lst)
      initializeNames(curIndent, os, elem, prefix, intrinsic);
  }
}

struct VariableSetting {
  StringMap<std::string> nameToOrdinal;
  StringMap<bool> isVector;
  StringMap<std::vector<int>> extractions;

  std::tuple<std::string, bool, std::vector<int>>
  lookup(StringRef name, const Record *pattern, const Init *resultRoot) const {
    auto ord = nameToOrdinal.find(name);
    if (ord == nameToOrdinal.end())
      PrintFatalError(pattern->getLoc(), Twine("unknown named operand '") +
                                             name + "'" +
                                             resultRoot->getAsString());
    auto iv = isVector.find(name);
    assert(iv != isVector.end());

    auto ext = extractions.find(name);
    assert(ext != extractions.end());
    return std::make_tuple(ord->getValue(), iv->getValue(), ext->getValue());
  }

  void insert(StringRef name, StringRef value, bool vec, std::vector<int> ext) {
    nameToOrdinal[name] = value;
    isVector[name] = vec;
    extractions[name] = ext;
  }
};

#define INDENT "  "
bool handle(const Twine &curIndent, const Twine &argPattern, raw_ostream &os,
            const Record *pattern, const Init *resultTree, StringRef builder,
            VariableSetting &nameToOrdinal, bool lookup,
            ArrayRef<unsigned> retidx, StringRef origName, bool newFromOriginal,
            ActionType intrinsic);

SmallVector<bool, 1> prepareArgs(const Twine &curIndent, raw_ostream &os,
                                 const Twine &argName, const Record *pattern,
                                 const DagInit *resultRoot, StringRef builder,
                                 VariableSetting &nameToOrdinal, bool lookup,
                                 ArrayRef<unsigned> retidx, StringRef origName,
                                 bool newFromOriginal, ActionType intrinsic) {
  SmallVector<bool, 1> vectorValued;

  size_t idx = 0;
  for (auto &&[args, names] :
       zip(resultRoot->getArgs(), resultRoot->getArgNames())) {
    os << curIndent << "auto " << argName << "_" << idx << " = ";
    idx++;
    if (isa<UnsetInit>(args) && names) {
      auto [ord, vecValue, ext] =
          nameToOrdinal.lookup(names->getValue(), pattern, resultRoot);
      if (!vecValue && !startsWith(ord, "local")) {

        if (ext.size()) {
          if (!lookup)
            os << "gutils->extractMeta(" << builder << ", ";
          else
            os << builder << ".CreateExtractValue(";
        }

        if (lookup && intrinsic != MLIRDerivatives)
          os << "lookup(";

        if (newFromOriginal && (!lookup || intrinsic != MLIRDerivatives))
          os << "gutils->getNewFromOriginal(";
      }
      if (lookup && !vecValue && !startsWith(ord, "local") &&
          intrinsic == MLIRDerivatives) {
        auto start = ord.find('(') + 1;
        auto end = ord.find(')');
        os << "operands[" << ord.substr(start, end - start) << "]";
      } else {
        os << ord;
      }
      if (!vecValue && !startsWith(ord, "local")) {
        if (newFromOriginal && (!lookup || intrinsic != MLIRDerivatives)) {
          os << ")";
        }
        if (intrinsic == MLIRDerivatives) {
          os << ";\n";
          os << curIndent << "if (gutils->width != 1) {\n"
             << curIndent << " " << argName << "_" << (idx - 1)
             << " = builder.create<enzyme::BroadcastOp>(\n"
             << curIndent << "   op.getLoc(),\n"
             << curIndent << "   " << argName << "_" << (idx - 1) << ",\n"
             << curIndent
             << "   llvm::SmallVector<int64_t>({gutils->width}));\n"
             << curIndent << "}";
        }

        if (lookup && intrinsic != MLIRDerivatives)
          os << ", " << builder << ")";

        if (ext.size()) {
          os << ", ArrayRef<unsigned>({";
          for (unsigned i = 0; i < ext.size(); i++) {
            if (i != 0)
              os << ", ";
            os << std::to_string(ext[i]);
          }
          os << "}))";
        }
      }
      os << ";\n";
      vectorValued.push_back(vecValue);
      continue;
    }
    vectorValued.push_back(handle(
        curIndent, argName + "_" + Twine(idx), os, pattern, args, builder,
        nameToOrdinal, lookup, retidx, origName, newFromOriginal, intrinsic));
    os << ";\n";
    if (names) {
      auto name = names->getAsUnquotedString();
      nameToOrdinal.insert(name, "local_" + name, vectorValued.back(), {});
      os << curIndent << "local_" << name << " = " << argName << "_"
         << (idx - 1) << ";\n";
    }
  }
  return vectorValued;
}

// Returns whether value generated is a vector value or not.
bool handle(const Twine &curIndent, const Twine &argPattern, raw_ostream &os,
            const Record *pattern, const Init *resultTree, StringRef builder,
            VariableSetting &nameToOrdinal, bool lookup,
            ArrayRef<unsigned> retidx, StringRef origName, bool newFromOriginal,
            ActionType intrinsic) {
  if (auto resultRoot = dyn_cast<DagInit>(resultTree)) {
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
        auto [ord, isVec, ext] =
            nameToOrdinal.lookup(name, pattern, resultRoot);
        assert(!isVec);
        assert(ext.size() == 0);
        os << ord;
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in typeof") +
                            resultTree->getAsString());
      if (intrinsic == MLIRDerivatives)
        os << ".getType()";
      else
        os << "->getType()";
      return false;
    } else if (opName == "VectorSize" || Def->isSubClassOf("VectorSize")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(),
                        "only single op VectorSize supported");

      os << "cast<VectorType>(";

      if (isa<UnsetInit>(resultRoot->getArg(0)) && resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec, ext] =
            nameToOrdinal.lookup(name, pattern, resultRoot);
        assert(!isVec);
        assert(!ext.size());
        os << ord;
      } else
        handle(curIndent + INDENT, argPattern + "_vs", os, pattern,
               resultRoot->getArg(0), builder, nameToOrdinal, lookup, retidx,
               origName, newFromOriginal, intrinsic);

      os << ")";
      os << "->getElementCount()";
#if LLVM_VERSION_MAJOR == 11
      os << ".Min";
#endif
      return false;
    } else if (Def->isSubClassOf("StaticSelect")) {
      auto numArgs = resultRoot->getNumArgs();

      if (numArgs != 2 && numArgs != 3)
        PrintFatalError(pattern->getLoc(),
                        "only two/three op StaticSelect supported");

      os << "({\n";
      os << curIndent << INDENT << "// Computing " << opName << "\n";
      if (intrinsic == MLIRDerivatives)
        os << curIndent << INDENT << "mlir::Value imVal = ";
      else
        os << curIndent << INDENT << "llvm::Value *imVal = ";

      int index = numArgs == 3;

      // First one is a name, set imVal to it
      if (numArgs == 3) {
        if (isa<UnsetInit>(resultRoot->getArg(0)) &&
            resultRoot->getArgName(0)) {
          auto name = resultRoot->getArgName(0)->getAsUnquotedString();
          auto [ord, isVec, ext] =
              nameToOrdinal.lookup(name, pattern, resultRoot);
          assert(!isVec);
          os << ord << ";\n";
        } else
          assert("Requires name for arg");
      } else {
        os << "nullptr;\n";
      }

      os << curIndent << INDENT << "bool condition = ";

      auto condition = dyn_cast<StringInit>(Def->getValueInit("condition"));
      if (!condition)
        PrintFatalError(pattern->getLoc(),
                        Twine("string 'condition' not defined in ") +
                            resultTree->getAsString());
      auto conditionStr = condition->getValue();

      if (conditionStr.contains("imVal") && numArgs == 2)
        PrintFatalError(pattern->getLoc(), "need a name as first argument");

      bool complexExpr = conditionStr.contains(';');
      if (complexExpr)
        os << "({\n";
      os << conditionStr;
      if (complexExpr)
        os << "\n" << curIndent << INDENT << "})";

      os << ";\n";

      os << curIndent << INDENT << "bool vectorized = false;\n";

      os << curIndent << INDENT << "if (condition) {\n";

      bool any_vector = false;
      bool all_vector = true;
      for (size_t i = index; i < numArgs; ++i) {
        os << curIndent << INDENT << INDENT << "imVal = ";

        bool vector;
        if (isa<UnsetInit>(resultRoot->getArg(i)) &&
            resultRoot->getArgName(i)) {
          auto name = resultRoot->getArgName(i)->getAsUnquotedString();
          auto [ord, isVec, ext] =
              nameToOrdinal.lookup(name, pattern, resultRoot);
          assert(!ext.size());
          vector = isVec;
          os << ord;
        } else {
          vector =
              handle(curIndent + INDENT + INDENT, argPattern + "_cic", os,
                     pattern, resultRoot->getArg(i), builder, nameToOrdinal,
                     lookup, retidx, origName, newFromOriginal, intrinsic);
        }
        os << ";\n";
        if (vector) {
          any_vector = true;
          os << curIndent << INDENT << INDENT << "vectorized = true;\n";
        } else {
          all_vector = false;
        }

        if (i == numArgs - 1) {
          os << curIndent << INDENT << "}\n";
        } else {
          os << curIndent << INDENT << "} else {\n";
        }
      }

      if (any_vector && !all_vector) {
        os << curIndent << INDENT << "if (!vectorized) {\n";
        if (intrinsic != MLIRDerivatives) {
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
        } else {
          os << curIndent << INDENT << "if (gutils->width != 1)\n"
             << curIndent << INDENT << INDENT
             << "imVal = builder.create<enzyme::BroadcastOp>(imVal.getLoc(), "
                "imVal, SmallVector<int64_t>({gutils->width}));\n";
        }
        os << curIndent << INDENT << "}\n";
      }

      os << curIndent << INDENT << "imVal;\n";
      os << curIndent << INDENT << "})";

      return any_vector;
    } else if (opName == "ConstantFP" || Def->isSubClassOf("ConstantFP")) {
      auto value = dyn_cast<StringInit>(Def->getValueInit("value"));
      if (!value)
        PrintFatalError(pattern->getLoc(), Twine("'value' not defined in ") +
                                               resultTree->getAsString());

      if (intrinsic == MLIRDerivatives) {
        if (resultRoot->getNumArgs() > 1)
          PrintFatalError(pattern->getLoc(),
                          "only zero or single op constantfp supported");
        os << builder << ".create<"
           << cast<StringInit>(Def->getValueInit("dialect"))->getValue()
           << "::" << cast<StringInit>(Def->getValueInit("opName"))->getValue()
           << ">(op.getLoc(), ";
        std::string ord;
        if (resultRoot->getNumArgs() == 0) {
          ord = "op->getResult(0)";
        } else {
          auto name = resultRoot->getArgName(0)->getAsUnquotedString();
          auto [ord1, isVec, ext] =
              nameToOrdinal.lookup(name, pattern, resultTree);
          assert(!isVec);
          assert(!ext.size());
          ord = ord1;
        }
        os << ord << ".getType(), ";
        auto typeCast =
            dyn_cast<StringInit>(Def->getValueInit("type"))->getValue();
        if (typeCast != "")
          os << "(" << typeCast << ")";
        os << "mlir::enzyme::getConstantAttr(" << ord << ".getType(), ";
        os << "\"" << value->getValue() << "\"))";
      } else {
        if (resultRoot->getNumArgs() != 1)
          PrintFatalError(pattern->getLoc(),
                          "only single op constantfp supported");

        os << "ConstantFP::get(";
        if (resultRoot->getArgName(0)) {
          auto name = resultRoot->getArgName(0)->getAsUnquotedString();
          auto [ord, isVec, ext] =
              nameToOrdinal.lookup(name, pattern, resultTree);
          assert(!isVec);
          if (ext.size())
            os << "gutils->extractMeta(";
          os << ord << "->getType()";
          if (ext.size()) {
            os << ", ArrayRef<unsigned>({";
            for (unsigned i = 0; i < ext.size(); i++) {
              if (i != 0)
                os << ", ";
              os << std::to_string(ext[i]);
            }
            os << "}))";
          }
        } else
          PrintFatalError(pattern->getLoc(),
                          Twine("unknown named operand in constantfp") +
                              resultTree->getAsString());
        os << ", \"" << value->getValue() << "\")";
      }
      return false;
    } else if (opName == "Zero" || Def->isSubClassOf("Zero")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op Zero supported");
      os << "Constant::getNullValue(";
      std::vector<int> exto;
      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec, ext] =
            nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);
        exto = std::move(ext);
        if (exto.size())
          os << "gutils->extractMeta(";
        os << ord;
      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in constantfp") +
                            resultTree->getAsString());
      os << "->getType()";
      if (exto.size()) {
        os << ", ArrayRef<unsigned>({";
        for (unsigned i = 0; i < exto.size(); i++) {
          if (i != 0)
            os << ", ";
          os << std::to_string(exto[i]);
        }
        os << "}))";
      }
      os << ")";
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
        auto [ord, isVec, ext] =
            nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);
        assert(!ext.size());
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
         << ivalue->getValue() << "\")});\n"
         << "} else if (auto AT = dyn_cast<ArrayType>(ty)) {\n"
         << curIndent << INDENT << INDENT
         << "ret = ConstantArray::get(AT, "
            "{(llvm::Constant*)ConstantFP::get(AT->getElementType(), \""
         << rvalue->getValue()
         << "\"), (llvm::Constant*)ConstantFP::get(AT->getElementType(), \""
         << ivalue->getValue() << "\")});\n";
      os << curIndent << INDENT << "} else {\n";
      os << curIndent << INDENT << "  llvm::errs() << *ty << \"\\n\";\n";
      os << curIndent << INDENT << "  assert(0 && \"unhandled cfp\");\n";
      os << curIndent << INDENT << "}\n";
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
          auto [ord, isVec, ext] =
              nameToOrdinal.lookup(name, pattern, resultTree);
          assert(!isVec);
          assert(!ext.size());
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
      if (value->getValue().contains(';'))
        os << "({ ";
      os << value->getValue();
      if (value->getValue().contains(';'))
        os << " })";
      return false;
    } else if (opName == "Undef" || Def->isSubClassOf("Undef")) {
      if (resultRoot->getNumArgs() != 1)
        PrintFatalError(pattern->getLoc(), "only single op undef supported");

      os << "UndefValue::get(";
      if (resultRoot->getArgName(0)) {
        auto name = resultRoot->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec, ext] =
            nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);
        assert(!ext.size());
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
        auto [ord, isVec, ext] =
            nameToOrdinal.lookup(name, pattern, resultTree);
        assert(!isVec);

        if (ext.size())
          os << "gutils->extractMeta(" << builder << ",";
        os << ord;

        if (ext.size()) {
          os << ", ArrayRef<unsigned>({";
          for (unsigned i = 0; i < ext.size(); i++) {
            if (i != 0)
              os << ", ";
            os << std::to_string(ext[i]);
          }
          os << "}))";
        }

      } else
        PrintFatalError(pattern->getLoc(),
                        Twine("unknown named operand in shadow") +
                            resultTree->getAsString());
      os << ", " << builder;
      if (intrinsic != MLIRDerivatives)
        os << ", /*nullShadow*/true";
      os << ")";
      if (lookup)
        os << ", " << builder << ")";
      return true;
    } else if (Def->isSubClassOf("MultiReturn")) {
      os << "({\n";

      bool useStruct = Def->getValueAsBit("struct");
      bool useRetType = Def->getValueAsBit("useRetType");

      SmallVector<bool, 1> vectorValued = prepareArgs(
          curIndent + INDENT, os, argPattern, pattern, resultRoot, builder,
          nameToOrdinal, lookup, retidx, origName, newFromOriginal, intrinsic);
      bool anyVector = false;
      for (auto b : vectorValued)
        anyVector |= b;

      if (!useStruct)
        assert(vectorValued.size());

      os << curIndent << INDENT << "Value *res = UndefValue::get(";
      if (anyVector)
        os << "gutils->getShadowType(";

      if (useRetType) {
        os << (origName == "<ILLEGAL>" ? "call" : origName) << ".getType()";
      } else {
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
      }

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
          nameToOrdinal, lookup, retidx, origName, newFromOriginal, intrinsic);
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

      std::function<void(const DagInit *, ArrayRef<unsigned>)> insert =
          [&](const DagInit *ptree, ArrayRef<unsigned> prev) {
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
                    os << builder << ".CreateExtractValue(" << op
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
                                      vectorValued[next[0]], {});
              }
              i++;
            }
          };

      insert(npattern, {});

      initializeNames(curIndent + INDENT, os, insts, "local", intrinsic);

      ArrayRef<unsigned> nretidx{};

      os << curIndent << INDENT;
      bool anyVector2 =
          handle(curIndent + INDENT, argPattern + "_sr", os, pattern, insts,
                 builder, nnameToOrdinal, /*lookup*/ false, nretidx,
                 "<ILLEGAL>", /*newFromOriginal*/ false, intrinsic);
      (void)anyVector2;
      assert(anyVector == anyVector2);
      os << ";\n";
      os << curIndent << "})";
      return anyVector;
    } else if (Def->isSubClassOf("Inst")) {

      os << "({\n";
      os << curIndent << INDENT << "// Computing " << opName << "\n";
      SmallVector<bool, 1> vectorValued = prepareArgs(
          curIndent + INDENT, os, argPattern, pattern, resultRoot, builder,
          nameToOrdinal, lookup, retidx, origName, newFromOriginal, intrinsic);
      bool anyVector = false;
      for (auto b : vectorValued)
        anyVector |= b;

      bool isCall = opName == "Call" || Def->isSubClassOf("Call");
      bool isIntr = opName == "Intrinsic" || Def->isSubClassOf("Intrinsic");

      if (isCall) {
        getFunction(curIndent + INDENT, os, "callval", "FT", "cconv",
                    Def->getValueInit("func"), origName);
      }

      if (anyVector && intrinsic != MLIRDerivatives) {
        os << curIndent << INDENT << "Value *res = nullptr;\n";
        os << curIndent << INDENT
           << "for(unsigned int idx=0, W=gutils->getWidth(); idx<W; idx++) {\n";
      }

      os << curIndent << INDENT;
      if (anyVector && intrinsic != MLIRDerivatives)
        os << INDENT;
      if (isCall) {
        os << "CallInst *V = ";
      } else if (anyVector && intrinsic != MLIRDerivatives) {
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
      } else if (intrinsic == MLIRDerivatives) {
        auto dialect = Def->getValueAsString("dialect");
        os << builder << ".create<" << dialect << "::" << opName
           << ">(op.getLoc(), ";
      } else {
        os << builder << ".Create" << opName << "(";
      }
      for (size_t i = 0; i < vectorValued.size(); i++) {
        if (i > 0)
          os << ", ";
        if (vectorValued[i] && intrinsic != MLIRDerivatives)
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
      if (intrinsic == MLIRDerivatives) {
        auto postop = Def->getValueAsString("postop");
        os << postop;
      }
      if (isCall) {
        os << ")";
      }
      os << ";\n";

      if (isCall) {
        os << curIndent << INDENT;
        if (anyVector && intrinsic != MLIRDerivatives)
          os << INDENT;
        if (intrinsic != MLIRDerivatives) {
          os << "V->setDebugLoc(gutils->getNewFromOriginal(" << origName
             << ".getDebugLoc()));"
                "\n";
          os << curIndent << INDENT;
          if (anyVector)
            os << INDENT;
          os << "V->setCallingConv(cconv);\n";
          for (auto *attr :
               *cast<ListInit>(Def->getValueAsListInit("fnattrs"))) {
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
      }
      if (anyVector && intrinsic != MLIRDerivatives) {
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

std::string ReplaceAll(std::string str, const std::string &from,
                       const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos +=
        to.length(); // Handles case where 'to' is a substring of 'from'
  }
  return str;
}

void handleUse(
    const DagInit *root, const DagInit *resultTree, std::string &foundPrimalUse,
    std::string &foundShadowUse, bool &foundDiffRet, std::string precondition,
    const DagInit *tree,
    StringMap<std::tuple<std::string, std::string, bool>> &varNameToCondition,
    const VariableSetting &nameToOrdinal);

void handleUseArgument(
    StringRef name, const Init *arg, bool usesPrimal, bool usesShadow,
    const DagInit *root, const DagInit *resultTree, std::string &foundPrimalUse,
    std::string &foundShadowUse, bool &foundDiffRet, std::string precondition,
    const DagInit *tree,
    StringMap<std::tuple<std::string, std::string, bool>> &varNameToCondition,
    const VariableSetting &nameToOrdinal) {

  auto arg2 = dyn_cast<DagInit>(arg);

  if (arg2) {
    // Recursive use of shadow is unhandled
    assert(!usesShadow);

    std::string foundPrimalUse2 = "";
    std::string foundShadowUse2 = "";

    bool foundDiffRet2 = false;
    // We set precondition to be false (aka "") if we do not need the
    // primal, since we are now only recurring to set variables
    // correctly.
    if (name.size() || usesPrimal)
      handleUse(root, arg2, name.size() ? foundPrimalUse2 : foundPrimalUse,
                name.size() ? foundShadowUse2 : foundShadowUse,
                name.size() ? foundDiffRet2 : foundDiffRet,
                usesPrimal ? precondition : "", tree, varNameToCondition,
                nameToOrdinal);

    if (name.size()) {
      if (foundPrimalUse2.size() &&
          !(startsWith(foundPrimalUse, foundPrimalUse2) ||
            endsWith(foundPrimalUse, foundPrimalUse2))) {
        if (foundPrimalUse.size() == 0)
          foundPrimalUse = foundPrimalUse2;
        else
          foundPrimalUse += " || " + foundPrimalUse2;
      }
      if (foundShadowUse2.size() &&
          !(startsWith(foundShadowUse, foundShadowUse2) ||
            endsWith(foundShadowUse, foundShadowUse2))) {
        if (foundShadowUse.size() == 0)
          foundShadowUse = foundShadowUse2;
        else
          foundShadowUse += " || " + foundShadowUse2;
      }
      foundDiffRet |= foundDiffRet2;

      varNameToCondition[name] =
          std::make_tuple(foundPrimalUse2, foundShadowUse2, foundDiffRet2);
    }
  } else {
    assert(name.size());

    if (name.size()) {
      auto found = varNameToCondition.find(name);
      if (found == varNameToCondition.end()) {
        llvm::errs() << "tree scope: " << *tree << "\n";
        llvm::errs() << "root scope: " << *root << "\n";
        llvm::errs() << "could not find var name: " << name << "\n";
      }
      assert(found != varNameToCondition.end());
    }

    if (precondition.size()) {
      auto [foundPrimalUse2, foundShadowUse2, foundDiffRet2] =
          varNameToCondition[name];
      if (precondition != "true") {
        if (foundPrimalUse2.size()) {
          foundPrimalUse2 =
              "((" + foundPrimalUse2 + ")&&(" + precondition + "))";
        }
        if (foundShadowUse2.size()) {
          foundShadowUse2 =
              "((" + foundShadowUse2 + ")&&(" + precondition + "))";
        }
      }
      if (usesPrimal) {
        if (foundPrimalUse2.size() &&
            !(startsWith(foundPrimalUse, foundPrimalUse2) ||
              endsWith(foundPrimalUse, foundPrimalUse2))) {
          if (foundPrimalUse.size() == 0)
            foundPrimalUse = foundPrimalUse2;
          else
            foundPrimalUse += " || " + foundPrimalUse2;
        }
        if (foundShadowUse2.size() &&
            !(startsWith(foundShadowUse, foundShadowUse2) ||
              endsWith(foundShadowUse, foundShadowUse2))) {
          if (foundShadowUse.size() == 0)
            foundShadowUse = foundShadowUse2;
          else
            foundShadowUse += " || " + foundShadowUse2;
        }
        foundDiffRet |= foundDiffRet2;
      }
      if (usesShadow) {
        if (foundPrimalUse2.size() &&
            !(startsWith(foundShadowUse, foundPrimalUse2) ||
              endsWith(foundShadowUse, foundPrimalUse2))) {
          if (foundShadowUse.size() == 0)
            foundShadowUse = foundPrimalUse2;
          else
            foundShadowUse += " || " + foundPrimalUse2;
        }
        assert(!foundDiffRet2);
        assert(foundShadowUse2 == "");
      }
    }
  }
}
void handleUse(
    const DagInit *root, const DagInit *resultTree, std::string &foundPrimalUse,
    std::string &foundShadowUse, bool &foundDiffRet, std::string precondition,
    const DagInit *tree,
    StringMap<std::tuple<std::string, std::string, bool>> &varNameToCondition,
    const VariableSetting &nameToOrdinal) {
  auto opName = resultTree->getOperator()->getAsString();
  auto Def = cast<DefInit>(resultTree->getOperator())->getDef();
  if (opName == "DiffeRetIndex" || Def->isSubClassOf("DiffeRetIndex")) {
    foundDiffRet = true;
    return;
  }
  if (opName == "InactiveArgSpec" || Def->isSubClassOf("InactiveArgSpec")) {
    return;
  }
  if (!Def->isSubClassOf("Operation")) {
    errs() << *resultTree << "\n";
    errs() << opName << " " << *Def << "\n";
  }
  assert(Def->isSubClassOf("Operation"));
  bool usesPrimal = Def->getValueAsBit("usesPrimal");
  bool usesShadow = Def->getValueAsBit("usesShadow");
  bool usesCustom = Def->getValueAsBit("usesCustom");

  if (Def->isSubClassOf("StaticSelect")) {
    auto numArgs = resultTree->getNumArgs();

    assert(numArgs == 2 || numArgs == 3);
    auto condition = dyn_cast<StringInit>(Def->getValueInit("condition"));
    assert(condition);
    std::string conditionStr = condition->getValue().str();

    assert(!(StringRef(conditionStr).contains("imVal") && numArgs == 2));

    // First one is a name, set imVal to it
    if (numArgs == 3) {
      if (isa<UnsetInit>(resultTree->getArg(0)) && resultTree->getArgName(0)) {
        auto name = resultTree->getArgName(0)->getAsUnquotedString();
        auto [ord, isVec, ext] = nameToOrdinal.lookup(name, nullptr, nullptr);
        assert(!isVec);
        conditionStr = ReplaceAll(conditionStr, "imVal", ord);
      } else
        assert("Requires name for arg");
    }

    bool complexExpr = StringRef(conditionStr).contains(';');
    if (complexExpr) {
      conditionStr = " ({ " + conditionStr + " }) ";
    }

    for (size_t i = numArgs == 3; i < numArgs; ++i) {
      std::string conditionStr2 =
          (i == numArgs - 1) ? (" !( " + conditionStr + " ) ") : conditionStr;
      std::string precondition2;
      if (precondition == "true")
        precondition2 = conditionStr2;
      else
        precondition2 = "((" + precondition + ")&&(" + conditionStr2 + "))";

      auto name = resultTree->getArgNameStr(i);
      auto arg = resultTree->getArg(i);
      handleUseArgument(name, arg, true, false, root, resultTree,
                        foundPrimalUse, foundShadowUse, foundDiffRet,
                        precondition2, tree, varNameToCondition, nameToOrdinal);
    }

    return;
  }

  (void)usesCustom;
  assert(!usesCustom);

  for (auto argEn : llvm::enumerate(resultTree->getArgs())) {
    auto name = resultTree->getArgNameStr(argEn.index());
    handleUseArgument(name, argEn.value(), usesPrimal, usesShadow, root,
                      resultTree, foundPrimalUse, foundShadowUse, foundDiffRet,
                      precondition, tree, varNameToCondition, nameToOrdinal);
  }
}

static VariableSetting parseVariables(const DagInit *tree, ActionType intrinsic,
                                      StringRef origName) {
  VariableSetting nameToOrdinal;
  std::function<void(const DagInit *, ArrayRef<unsigned>)> insert =
      [&](const DagInit *ptree, ArrayRef<unsigned> prev) {
        unsigned i = 0;
        for (auto tree : ptree->getArgs()) {
          SmallVector<unsigned, 2> next(prev.begin(), prev.end());
          next.push_back(i);
          if (auto dg = dyn_cast<DagInit>(tree))
            insert(dg, next);

          if (ptree->getArgNameStr(i).size()) {
            std::string op;
            if (intrinsic != MLIRDerivatives)
              op = (origName + ".getOperand(" + Twine(next[0]) + ")").str();
            else
              op = (origName + "->getOperand(" + Twine(next[0]) + ")").str();
            std::vector<int> extractions;
            if (prev.size() > 0) {
              for (unsigned i = 1; i < next.size(); i++) {
                extractions.push_back(next[i]);
              }
            }
            nameToOrdinal.insert(ptree->getArgNameStr(i), op, false,
                                 extractions);
          }
          i++;
        }
      };

  insert(tree, {});

  if (tree->getNameStr().size())
    nameToOrdinal.insert(tree->getNameStr(),
                         (Twine("(&") + origName + ")").str(), false, {});
  return nameToOrdinal;
}

void printDiffUse(
    raw_ostream &os, Twine prefix, const ListInit *argOps, StringRef origName,
    ActionType intrinsic, const DagInit *tree,
    StringMap<std::tuple<std::string, std::string, bool>> &varNameToCondition) {
  os << prefix << "  // Rule " << *tree << "\n";

  VariableSetting nameToOrdinal = parseVariables(tree, intrinsic, origName);

  for (auto argOpEn : enumerate(*argOps)) {
    size_t argIdx = argOpEn.index();
    if (auto resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
      auto opName = resultRoot->getOperator()->getAsString();
      auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
      if (opName == "InactiveArgSpec" || Def->isSubClassOf("InactiveArgSpec")) {
        continue;
      }
    }

    // The condition necessary to require the use of the arg
    std::string foundPrimalUse = "";
    std::string foundShadowUse = "";
    bool foundDiffRet = false;

    auto resultTree = cast<DagInit>(argOpEn.value());

    if (intrinsic != MLIRDerivatives) {
      os << prefix
         << "  if (gutils->mode == DerivativeMode::ForwardModeError) {\n";
      os << prefix
         << "    if (!gutils->isConstantValue(const_cast<Value*>(val))) {\n";
      os << prefix
         << "      if (EnzymePrintDiffUse) llvm::errs() << \"Need primal of "
            "all active operands in error propagation\\n\";\n";
      os << prefix << "      return true;\n";
      os << prefix << "    }\n";
      os << prefix << "  }\n";
    }

    // hasDiffeRet(resultTree)
    handleUse(resultTree, resultTree, foundPrimalUse, foundShadowUse,
              foundDiffRet, /*precondition*/ "true", tree, varNameToCondition,
              nameToOrdinal);

    os << prefix << "  // Arg " << argIdx << " : " << *resultTree << "\n";

    if (foundPrimalUse != "") {
      if (intrinsic == MLIRDerivatives)
        os << prefix << "  if (!gutils->isConstantValue(" << origName
           << "->getOperand(" << argIdx << "))";
      else
        os << prefix
           << "  if (!shadow && !gutils->isConstantValue(const_cast<Value*>("
           << origName << "->getOperand(" << argIdx << ")))";

      if (foundDiffRet) {
        if (intrinsic == MLIRDerivatives)
          os << " && !gutils->isConstantValue(" << origName
             << "->getResult(0))";
        else
          os << " && !gutils->isConstantValue(const_cast<Value*>((const Value*)"
             << origName << "))";
      } else {
        if (intrinsic == MLIRDerivatives)
          os << " && !gutils->isConstantInstruction(" << origName << ")";
        else
          os << " && !gutils->isConstantInstruction(const_cast<Instruction*>( "
             << origName << "))";
      }

      os << ") {\n";
      os << prefix << "    if (" << foundPrimalUse << ") {\n";
      if (intrinsic == MLIRDerivatives)
        os << prefix << "      used = true;\n";
      else {
        os << prefix << "      if (EnzymePrintDiffUse)\n";
        os << prefix
           << "         llvm::errs() << \"Need direct primal of \" << *val << ";
        os << "\"in reverse from \" << *user << \" from condition "
           << foundPrimalUse;
        os << "\";\n";
        os << prefix << "      return true;\n";
      }
      os << prefix << "    }\n";

      os << prefix << "  }\n";
    }

    if (intrinsic != MLIRDerivatives) {
      os << prefix << "  if (shadow && !gutils->isConstantValue(" << origName
         << "->getOperand(" << argIdx << "))";

      if (foundDiffRet) {
        os << " && !gutils->isConstantValue(const_cast<Value*>((const Value*)"
           << origName << "))";
      } else {
        os << " && !gutils->isConstantInstruction(const_cast<Instruction*>( "
           << origName << "))";
      }

      os << ") {\n";

      os << prefix
         << "    if (qtype == QueryType::Shadow && (mode == "
            "DerivativeMode::ForwardMode || mode == "
            "DerivativeMode::ForwardModeSplit)) {\n";
      os << prefix
         << "      if (EnzymePrintDiffUse) llvm::errs() << \"Need forward "
            "shadow of \" << *val << \" from condition \" << *user << "
            "\"\\n\";\n";
      os << prefix << "        return true;\n";
      os << prefix << "      }\n";

      if (foundShadowUse != "") {
        os << prefix << "    if (" << foundShadowUse << ") {\n";
        os << prefix << "      if (EnzymePrintDiffUse)\n";
        os << "           llvm::errs() << \"Need direct shadow of \" << *val "
              "<< ";
        os << "\"in reverse from \" << *user << \" from condition "
           << foundShadowUse;
        os << "\";\n";
        os << prefix << "      return true;\n";
        os << prefix << "    }\n";
      }

      os << prefix << "  }\n";
    }
  }

  if (intrinsic != MLIRDerivatives) {
    os << prefix << "  return false;\n";
    os << prefix << "}\n";
  }
}

static void emitMLIRReverse(raw_ostream &os, const Record *pattern,
                            const DagInit *tree, ActionType intrinsic,
                            StringRef origName, const ListInit *argOps) {
  auto opName = pattern->getValueAsString("opName");
  auto dialect = pattern->getValueAsString("dialect");
  os << "struct " << opName << "RevDerivative : \n";
  os << "			public "
        "ReverseAutoDiffOpInterface::ExternalModel<"
     << opName << "RevDerivative, " << dialect << "::" << opName << "> {\n";
  os << "       SmallVector<bool> cachedArguments(Operation *op,\n";
  os << "                                 MGradientUtilsReverse *gutils) "
        "const {\n";
  os << "         SmallVector<bool> toret(op->getNumOperands(), false);\n";
  StringMap<std::tuple<std::string, std::string, bool>> varNameToCondition;

  std::function<void(const DagInit *, ArrayRef<unsigned>)> insert =
      [&](const DagInit *ptree, ArrayRef<unsigned> prev) {
        for (auto treeEn : llvm::enumerate(ptree->getArgs())) {
          auto tree = treeEn.value();
          auto name = ptree->getArgNameStr(treeEn.index());
          SmallVector<unsigned, 2> next(prev.begin(), prev.end());
          next.push_back(treeEn.index());
          if (auto dg = dyn_cast<DagInit>(tree))
            insert(dg, next);

          if (name.size()) {
            varNameToCondition[name] = std::make_tuple(
                "idx == " + std::to_string(treeEn.index()), "", false);
          }
        }
      };

  insert(tree, {});

  if (tree->getNameStr().size())
    varNameToCondition[tree->getNameStr()] =
        std::make_tuple("ILLEGAL", "ILLEGAL", false);

  os << "         for (size_t idx=0; idx<op->getNumOperands(); idx++) {\n";
  os << "            bool used = false;\n";
  printDiffUse(os, "          ", argOps, origName, intrinsic, tree,
               varNameToCondition);
  os << "            toret[idx] = used;\n";
  os << "         }\n";
  os << "         return toret;\n";
  os << "       }\n";

  os << "       SmallVector<Value> cacheValues(Operation *op,\n";
  os << "                                 MGradientUtilsReverse *gutils) "
        "const {\n";
  os << "          if (gutils->isConstantInstruction(op) || "
        "gutils->isConstantValue(op->getResult(0))) return {};\n";
  os << "          auto neededArgs = cachedArguments(op, gutils);\n";
  os << "          SmallVector<Value> toret;\n";
  os << "          OpBuilder builder(gutils->getNewFromOriginal(op));\n";
  os << "          for (auto en : llvm::enumerate(neededArgs))\n";
  os << "            if (en.value()) {\n";
  os << "              Value cache = "
        "gutils->initAndPushCache(gutils->getNewFromOriginal(op->"
        "getOperand(en.index())), builder);\n";
  os << "              toret.push_back(cache);\n";
  os << "            }\n";
  os << "          return toret;\n";
  os << "       }\n";
  os << "\n";
  os << "  void createShadowValues(Operation *op, OpBuilder &builder,\n";
  os << "                          MGradientUtilsReverse *gutils) const "
        "{}\n";

  os << "     LogicalResult createReverseModeAdjoint(Operation *op0, OpBuilder "
        "&builder,\n";
  os << "                            MGradientUtilsReverse *gutils,\n";
  os << "                            SmallVector<Value> caches) const {\n";
  os << "    auto op = cast<" << dialect << "::" << opName << ">(op0);\n";
  os << "        mlir::Value dif = nullptr;\n";
}

static void emitReverseCommon(raw_ostream &os, const Record *pattern,
                              const DagInit *tree, ActionType intrinsic,
                              StringRef origName, const ListInit *argOps) {
  auto nameToOrdinal = parseVariables(tree, intrinsic, origName);

  bool seen = false;
  for (auto argOpEn : enumerate(*argOps)) {
    size_t argIdx = argOpEn.index();
    if (auto resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
      auto opName = resultRoot->getOperator()->getAsString();
      auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
      if (opName == "InactiveArgSpec" || Def->isSubClassOf("InactiveArgSpec")) {
        if (Def->getValueAsBit("asserting")) {
          if (intrinsic == MLIRDerivatives) {
            os << " if (!gutils->isConstantValue(" << origName
               << "->getOperand(" << argIdx << "))) {\n";
            os << "    " << origName
               << "->emitError() << \"Unimplemented derivative for argument "
               << argIdx << " in reverse mode for op \" << *" << origName
               << " << \"\\n\";\n";
            os << "  return failure();\n";
            os << "  }\n";
          } else {
            os << " assert(gutils->isConstantValue(" << origName
               << ".getOperand(" << argIdx << ")));\n";
          }
        }
        continue;
      }
    }

    os << "        ";
    if (seen)
      os << "} else ";
    seen = true;
    if (intrinsic == MLIRDerivatives) {
      os << "if (!dif && !gutils->isConstantValue(" << origName
         << "->getOperand(" << argIdx << "))) {\n";
    } else {
      os << "if (!dif && !gutils->isConstantValue(" << origName
         << ".getOperand(" << argIdx << ")) && !isa<PointerType>(" << origName
         << ".getOperand(" << argIdx << ")->getType()) ) {\n";
    }
    auto resultTree = cast<DagInit>(argOpEn.value());
    if (hasDiffeRet(resultTree)) {
      if (intrinsic == MLIRDerivatives) {
        os << "          dif = gutils->diffe(" << origName << ", builder);\n";
        os << "          dif = "
              "cast<AutoDiffTypeInterface>(dif.getType()).createConjOp(builder,"
              " dif.getLoc(), dif);\n";
        os << "          gutils->zeroDiffe(" << origName << ", builder);\n";
      } else {
        os << "          dif = diffe(&" << origName << ", Builder2);\n";
        os << "          setDiffe(&" << origName
           << ", "
              "Constant::getNullValue(gutils->getShadowType("
           << origName
           << ".getType())), "
              "Builder2);\n";
      }
    }
  }
  if (seen)
    os << "        }\n";

  if (intrinsic == MLIRDerivatives) {
    os << "   SmallVector<Value> operands(op->getNumOperands(), nullptr);\n";
    os << "          auto neededArgs = cachedArguments(op, gutils);\n";
    os << "          size_t count = 0;\n";
    os << "          for (auto en : llvm::enumerate(neededArgs))\n";
    os << "            if (en.value()) {\n";
    os << "              operands[en.index()] = "
          "gutils->popCache(caches[count], builder);\n";
    os << "              count++;\n";
    os << "            }\n";
  }

  std::function<void(size_t, ArrayRef<unsigned>, const Init *)> revres =
      [&](size_t argIdx, ArrayRef<unsigned> idx, const Init *ival) {
        if (auto resultTree = dyn_cast<DagInit>(ival)) {
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
          if (Def->isSubClassOf("InactiveArgSpec")) {
            return;
          }
          const char *curIndent = "          ";
          os << curIndent << "{\n";
          if (intrinsic == MLIRDerivatives)
            os << curIndent << INDENT << "mlir::Value tmp = ";
          else
            os << curIndent << INDENT << "Value *tmp = ";
          bool vectorValued = handle(
              Twine(curIndent) + INDENT, "revarg", os, pattern, resultTree,
              (intrinsic == MLIRDerivatives) ? "builder" : "Builder2",
              nameToOrdinal, /*lookup*/ true, idx, origName,
              /*newFromOriginal*/ true, intrinsic);
          os << ";\n";

          if (intrinsic == MLIRDerivatives) {
            os << curIndent << INDENT
               << "tmp = "
                  "tmp.getType().cast<AutoDiffTypeInterface>().createConjOp("
                  "builder, op.getLoc(), tmp);\n";
          }

          if (intrinsic == MLIRDerivatives) {
            os << "assert(toadd == nullptr); toadd = tmp;\n";
          } else {
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
          }
          os << curIndent << "}\n";
        } else if (auto lst = dyn_cast<ListInit>(ival)) {
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
    if (auto resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
      auto opName = resultRoot->getOperator()->getAsString();
      auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
      if (opName == "InactiveArgSpec" || Def->isSubClassOf("InactiveArgSpec")) {
        continue;
      }
    }

    const char *curIndent = "        ";
    if (intrinsic == MLIRDerivatives)
      os << curIndent << "if (!gutils->isConstantValue(" << origName
         << "->getOperand(" << argIdx << "))) {\n";
    else
      os << curIndent << "if (!gutils->isConstantValue(" << origName
         << ".getOperand(" << argIdx << ")) && !isa<PointerType>(" << origName
         << ".getOperand(" << argIdx << ")->getType()) ) {\n";

    initializeNames(Twine(curIndent) + INDENT, os, argOpEn.value(), "local",
                    intrinsic);
    if (intrinsic == MLIRDerivatives)
      os << curIndent << INDENT << "mlir::Value toadd = nullptr;\n";
    else
      os << curIndent << INDENT << "Value *toadd = nullptr;\n";
    revres(argIdx, {}, argOpEn.value());

    if (intrinsic == MLIRDerivatives) {
      os << curIndent << INDENT << "if (toadd) gutils->addToDiffe(" << origName
         << "->getOperand(" << argIdx << "), toadd, builder);\n";
    } else {
      os << curIndent << INDENT << "if (toadd) addToDiffe(" << origName
         << ".getOperand(" << argIdx << "), toadd";
      os << ", Builder2, " << origName << ".getOperand(" << argIdx
         << ")->getType());\n";
    }
    os << curIndent << "}\n";
  }
}
static void emitDerivatives(const RecordKeeper &recordKeeper, raw_ostream &os,
                            ActionType intrinsic) {
  emitSourceFileHeader("Rewriters", os);
  const char *patternNames = "";
  switch (intrinsic) {
  case MLIRDerivatives:
    patternNames = "MLIRDerivative";
    break;
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
  case GenHeaderVariables:
  case GenBlasDerivatives:
  case UpdateBlasDecl:
  case UpdateBlasTA:
  case GenBlasDiffUse:
    llvm_unreachable("Cannot use blas updaters inside emitDerivatives");
  }
  const auto &patterns = recordKeeper.getAllDerivedDefinitions(patternNames);

  for (const Record *pattern : patterns) {
    auto tree = pattern->getValueAsDag("PatternToMatch");

    auto duals = pattern->getValueAsDag("ArgDuals");
    assert(duals);

    // Emit RewritePattern for Pattern.
    auto argOps = pattern->getValueAsListInit("ArgDerivatives");

    if (tree->getNumArgs() != argOps->size()) {
      PrintFatalError(pattern->getLoc(),
                      Twine("Defined rule pattern to have ") +
                          Twine(tree->getNumArgs()) +
                          " args but reverse rule array is a list of size " +
                          Twine(argOps->size()));
    }

    std::string origName;
    switch (intrinsic) {
    case GenBlasDerivatives:
    case UpdateBlasDecl:
    case UpdateBlasTA:
    case GenBlasDiffUse:
    case GenHeaderVariables:
      llvm_unreachable("Cannot use blas updaters inside emitDerivatives");
    case MLIRDerivatives: {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "struct " << opName << "FwdDerivative : \n";
      os << "			public AutoDiffOpInterface::ExternalModel<"
         << opName << "FwdDerivative, " << dialect << "::" << opName << "> {\n";
      os << "  LogicalResult createForwardModeTangent(Operation *op0, "
            "OpBuilder &builder, MGradientUtils *gutils) const {\n";
      os << "    auto op = cast<" << dialect << "::" << opName << ">(op0);\n";
      origName = "op";
      break;
    }
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
          int min_int = 100000;
          min.getAsInteger(10, min_int);
          if (min.size() != 0 && LLVM_VERSION_MAJOR < min_int)
            continue;
          if (lst->size() >= 3) {
            auto max = cast<StringInit>(lst->getValues()[2])->getValue();
            int max_int = 0;
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
      os << "#ifdef __clang__\n"
         << "#pragma clang diagnostic push\n"
         << "#pragma clang diagnostic ignored \"-Wunused-variable\"\n"
         << "#pragma clang diagnostic ignored \"-Wunused-but-set-variable\"\n"
         << "#else\n"
         << "#pragma GCC diagnostic push\n"
         << "#pragma GCC diagnostic ignored \"-Wunused-variable\"\n"
         << "#pragma GCC diagnostic ignored \"-Wunused-but-set-variable\"\n"
         << "#endif\n";
      os << "    auto mod = inst.getParent()->getParent()->getParent();\n";
      os << "    auto *const newCall = "
            "cast<llvm::Instruction>(gutils->getNewFromOriginal(&"
         << origName << "));\n";
      os << "#ifdef __clang__\n"
         << "#pragma clang diagnostic pop\n"
         << "#else\n"
         << "#pragma GCC diagnostic pop\n"
         << "#endif\n";
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
      os << "#ifdef __clang__\n"
         << "#pragma clang diagnostic push\n"
         << "#pragma clang diagnostic ignored \"-Wunused-variable\"\n"
         << "#pragma clang diagnostic ignored \"-Wunused-but-set-variable\"\n"
         << "#else\n"
         << "#pragma GCC diagnostic push\n"
         << "#pragma GCC diagnostic ignored \"-Wunused-variable\"\n"
         << "#pragma GCC diagnostic ignored \"-Wunused-but-set-variable\"\n"
         << "#endif\n";
      os << "    auto mod = BO.getParent()->getParent()->getParent();\n";
      os << "    auto *const newCall = "
            "cast<llvm::Instruction>(gutils->getNewFromOriginal(&"
         << origName << "));\n";
      os << "#ifdef __clang__\n"
         << "#pragma clang diagnostic pop\n"
         << "#else\n"
         << "#pragma GCC diagnostic pop\n"
         << "#endif\n";
      os << "    IRBuilder<> BuilderZ(newCall);\n";
      os << "    BuilderZ.setFastMathFlags(getFast());\n";
      break;
    }
    }

    VariableSetting nameToOrdinal = parseVariables(tree, intrinsic, origName);

    if (intrinsic != BinopDerivatives && intrinsic != InstDerivatives &&
        intrinsic != MLIRDerivatives) {
      os << "    if (gutils->knownRecomputeHeuristic.find(&" << origName
         << ") !=\n";
      os << "        gutils->knownRecomputeHeuristic.end()) {\n";
      os << "        if (!gutils->knownRecomputeHeuristic[&" << origName
         << "]) {\n";
      os << "          gutils->cacheForReverse(BuilderZ, newCall,\n";
      os << "                                  getIndex(&" << origName
         << ", "
            "CacheType::Self, BuilderZ));\n";
      os << "        }\n";
      os << "    }\n";
    }

    if (intrinsic != MLIRDerivatives)
      os << "    eraseIfUnused(" << origName << ");\n";
    else
      os << "    gutils->eraseIfUnused(" << origName << ");\n";

    if (intrinsic != MLIRDerivatives) {
      os << "#ifdef ENZYME_ENABLE_FPOPT\n";
      os << "    if (auto *logFunc = getFPOptLogger(" << origName
         << ".getModule(), \"enzymeLogValue\")) {\n"
         << "      IRBuilder<> BuilderZ(&" << origName << ");\n"
         << "      getForwardBuilder(BuilderZ);\n"
         << "      std::string idStr = getLogIdentifier(" << origName << ");\n"
         << "      Value *idValue = "
            "BuilderZ.CreateGlobalStringPtr(idStr);\n"
         << "      Value *origValue = "
            "BuilderZ.CreateFPExt(gutils->getNewFromOriginal(&"
         << origName << "), Type::getDoubleTy(" << origName
         << ".getContext()));\n"
         << "      unsigned numOperands = isa<CallInst>(" << origName
         << ") ? cast<CallInst>(" << origName << ").arg_size() : " << origName
         << ".getNumOperands();\n"
         << "      Value *numOperandsValue = "
            "ConstantInt::get(Type::getInt32Ty("
         << origName << ".getContext()), numOperands);\n"
         << "      auto operands = isa<CallInst>(" << origName
         << ") ? cast<CallInst>(" << origName << ").args() : " << origName
         << ".operands();\n"
         << "      ArrayType *operandArrayType = "
            "ArrayType::get(Type::getDoubleTy("
         << origName << ".getContext()), numOperands);\n"
         << "      Value *operandArrayValue = "
            "IRBuilder<>(gutils->inversionAllocs).CreateAlloca("
            "operandArrayType);\n"
         << "      for (auto operand : enumerate(operands)) {\n"
         << "        Value *operandValue = "
            "BuilderZ.CreateFPExt(gutils->getNewFromOriginal(operand.value()), "
            "Type::getDoubleTy("
         << origName << ".getContext()));\n"
         << "        Value *ptr = "
            "BuilderZ.CreateGEP(operandArrayType, operandArrayValue, "
            "{ConstantInt::get(Type::getInt32Ty("
         << origName << ".getContext()), 0), ConstantInt::get(Type::getInt32Ty("
         << origName << ".getContext()), operand.index())});\n"
         << "        BuilderZ.CreateStore(operandValue, ptr);\n"
         << "      }\n"
         << "      Value *operandPtrValue = "
            "BuilderZ.CreateGEP(operandArrayType, operandArrayValue, "
            "{ConstantInt::get(Type::getInt32Ty("
         << origName << ".getContext()), 0), ConstantInt::get(Type::getInt32Ty("
         << origName << ".getContext()), 0)});\n"
         << "      CallInst *logCallInst = BuilderZ.CreateCall(logFunc, "
         << "{idValue, origValue, numOperandsValue, operandPtrValue});\n"
         << "      logCallInst->setDebugLoc(gutils->getNewFromOriginal("
         << origName << ".getDebugLoc()));\n"
         << "    }\n";
      os << "#endif\n";
    }

    if (intrinsic == MLIRDerivatives) {
      os << "    if (gutils->isConstantInstruction(op))\n";
      os << "      return success();\n";
    } else {
      os << "    if (gutils->isConstantInstruction(&" << origName
         << ") && gutils->isConstantValue(&" << origName << "))\n";
      if (intrinsic == IntrDerivatives || intrinsic == CallDerivatives)
        os << "      return true;\n";
      else
        os << "      return;\n";

      os << "    switch (Mode) {\n";
      os << "      case DerivativeMode::ForwardModeSplit:\n";
      os << "      case DerivativeMode::ForwardMode:{\n";
      os << "        IRBuilder<> Builder2(&" << origName << ");\n";
      os << "        getForwardBuilder(Builder2);\n";
    }
    // TODO

    if (duals->getOperator()->getAsString() ==
            "ForwardFromSummedReverseInternal" ||
        cast<DefInit>(duals->getOperator())
            ->getDef()
            ->isSubClassOf("ForwardFromSummedReverseInternal")) {

      if (intrinsic == MLIRDerivatives) {
        os << "     mlir::Value res = nullptr;\n";
      } else {
        os << "        Value *res = "
              "Constant::getNullValue(gutils->getShadowType("
           << origName
           << "."
              "getType()));\n";
      }

      for (auto argOpEn : enumerate(*argOps)) {
        size_t argIdx = argOpEn.index();

        const char *curIndent = "        ";

        if (auto resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
          auto opName = resultRoot->getOperator()->getAsString();
          auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
          if (Def->isSubClassOf("InactiveArgSpec")) {
            if (Def->getValueAsBit("asserting"))
              os << " assert(gutils->isConstantValue(" << origName
                 << ".getOperand(" << argIdx << ")));\n";
            continue;
          }
        }

        if (intrinsic == MLIRDerivatives) {
          os << curIndent << "if (!gutils->isConstantValue(" << origName
             << "->getOperand(" << argIdx << "))) {\n";
          os << curIndent << INDENT << "auto dif = gutils->invertPointerM("
             << origName << "->getOperand(" << argIdx << "), builder);\n";
        } else {
          os << curIndent << "if (!gutils->isConstantValue(" << origName
             << ".getOperand(" << argIdx << "))) {\n";
          os << curIndent << INDENT << "Value *dif = diffe(" << origName
             << ".getOperand(" << argIdx << "), Builder2);\n";
          os << curIndent << INDENT
             << "Value *arg_diff_tmp = UndefValue::get(res->getType());\n";
        }

        initializeNames(Twine(curIndent) + INDENT, os, argOpEn.value(), "local",
                        intrinsic);
        std::function<void(ArrayRef<unsigned>, const Init *)> fwdres =
            [&](ArrayRef<unsigned> idx, const Init *ival) {
              if (auto resultTree = dyn_cast<DagInit>(ival)) {
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
                if (Def->isSubClassOf("InactiveArgSpec")) {
                  return;
                }
                os << curIndent << INDENT << "{\n";
                if (intrinsic == MLIRDerivatives)
                  os << curIndent << INDENT << INDENT << "mlir::Value itmp = ";
                else
                  os << curIndent << INDENT << INDENT << "Value *itmp = ";
                ArrayRef<unsigned> retidx{};
                bool vectorValued = handle(
                    Twine(curIndent) + INDENT + INDENT, "fwdarg", os, pattern,
                    resultTree,
                    (intrinsic == MLIRDerivatives) ? "builder" : "Builder2",
                    nameToOrdinal, /*lookup*/ false, retidx, origName,
                    /*newFromOriginal*/ true, intrinsic);
                os << ";\n";
                (void)vectorValued;
                assert(vectorValued);
                if (intrinsic == MLIRDerivatives) {
                  os << curIndent << INDENT << INDENT
                     << "if (!res) res = itmp;\n";
                  os << curIndent << INDENT << INDENT << "else {\n";
                  os << curIndent << INDENT << INDENT << INDENT
                     << "auto operandType = "
                        "cast<AutoDiffTypeInterface>(res.getType());\n";
                  os << curIndent << INDENT << INDENT << INDENT
                     << "res = operandType.createAddOp(builder, op.getLoc(), "
                        "res, itmp);\n";
                  os << curIndent << INDENT << INDENT << "}\n";
                } else {
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
                }
                os << curIndent << INDENT << "}\n";
              } else if (auto lst = dyn_cast<ListInit>(ival)) {
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
        if (intrinsic != MLIRDerivatives) {
          os << curIndent << INDENT << "res = arg_diff_tmp;\n";
        }
        os << "        }\n";
      }
    } else {

      if (intrinsic == MLIRDerivatives) {
        os << "            mlir::Value res = ";
      } else {
        os << "            Value *res = ";
      }
      ArrayRef<unsigned> retidx{};
      bool vectorValued =
          handle("            ", "fwdnsrarg", os, pattern, duals,
                 (intrinsic == MLIRDerivatives) ? "builder" : "Builder2",
                 nameToOrdinal, /*lookup*/ false, retidx, origName,
                 /*newFromOriginal*/ true, intrinsic);
      (void)vectorValued;
      assert(vectorValued);
      os << ";\n";
    }
    os << "        assert(res);\n";
    if (intrinsic == MLIRDerivatives) {
      os << "        gutils->setDiffe(" << origName
         << "->getResult(0), res, builder);\n";
      os << "        return success();\n";
    } else {
      os << "        setDiffe(&" << origName << ", res, Builder2);\n";
      os << "        break;\n";
    }
    os << "      }\n";

    // forward error TODO: `ForwardFromSummedReverse` behavior
    // also for custom derivatives.
    if (intrinsic != MLIRDerivatives) {
      os << "      case DerivativeMode::ForwardModeError: {\n";
      os << "        IRBuilder<> Builder2(&" << origName << ");\n";
      os << "        getForwardBuilder(Builder2);\n";
      os << "        Value *res = "
         << "Constant::getNullValue(gutils->getShadowType(" << origName
         << "."
            "getType()));\n";
      for (auto argOpEn : enumerate(*argOps)) {
        size_t argIdx = argOpEn.index();

        const char *curIndent = "        ";

        if (auto resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
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
        // error from https://dl.acm.org/doi/10.1145/3371128
        // error(f(x, y)) = max(ulp(f(x, y)), abs(x / f(x, y) * df/dx *
        // error(x)) + abs(y / f(x, y) * df/dy * error(y)))

        os << curIndent << INDENT
           << "dif = Builder2.CreateFDiv(Builder2.CreateFMul(dif, "
              "gutils->getNewFromOriginal("
           << origName << ".getOperand(" << argIdx
           << "))), gutils->getNewFromOriginal(&" << origName << "));\n";

        os << curIndent << INDENT
           << "Value *arg_diff_tmp = UndefValue::get(res->getType());\n";

        initializeNames(Twine(curIndent) + INDENT, os, argOpEn.value(), "local",
                        intrinsic);
        std::function<void(ArrayRef<unsigned>, const Init *)> fwdres =
            [&](ArrayRef<unsigned> idx, const Init *ival) {
              if (auto resultTree = dyn_cast<DagInit>(ival)) {
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
                if (Def->isSubClassOf("InactiveArgSpec")) {
                  return;
                }
                os << curIndent << INDENT << "{\n";
                os << curIndent << INDENT << INDENT << "Value *itmp = ";
                ArrayRef<unsigned> retidx{};
                bool vectorValued =
                    handle(Twine(curIndent) + INDENT + INDENT, "fwdarg", os,
                           pattern, resultTree, "Builder2", nameToOrdinal,
                           /*lookup*/ false, retidx, origName,
                           /*newFromOriginal*/ true, intrinsic);
                os << ";\n";
                (void)vectorValued;
                assert(vectorValued);

                // Add the sum of the abs of errors due to each argument.

                os << curIndent << INDENT << INDENT
                   << "itmp = Builder2.CreateIntrinsic(Intrinsic::fabs, "
                      "ArrayRef<Type*>(itmp->getType()), "
                      "ArrayRef<Value*>(itmp));\n";

                os << curIndent << INDENT << INDENT
                   << "arg_diff_tmp = "
                      "GradientUtils::recursiveFAdd(Builder2,";
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
              } else if (auto lst = dyn_cast<ListInit>(ival)) {
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
        os << curIndent << "}\n";
      }

      // Perform the max with 1 ulp
      // error TODO
      os << "        res = Builder2.CreateMaxNum(get1ULP(Builder2, "
            "gutils->getNewFromOriginal(&"
         << origName << ")), res);\n";

      os << "        assert(res);\n";

      // Insert logging function call (optional)
      os << "#ifdef ENZYME_ENABLE_FPOPT\n";
      os << "        if (auto *logFunc = getFPOptLogger(" << origName
         << ".getModule(), \"enzymeLogError\")) {\n"
         << "          std::string idStr = getLogIdentifier(" << origName
         << ");\n"
         << "          Value *idValue = "
            "BuilderZ.CreateGlobalStringPtr(idStr);\n"
         << "            Value *errValue = Builder2.CreateFPExt(res, "
            "Type::getDoubleTy("
         << origName << ".getContext()));\n"
         << "            CallInst *logCallInst = Builder2.CreateCall(logFunc, "
            "{idValue, errValue});\n"
         << "            logCallInst->setDebugLoc(gutils->getNewFromOriginal("
         << origName << ".getDebugLoc()));\n"
         << "        }\n";
      os << "#endif\n";

      os << "        setDiffe(&" << origName << ", res, Builder2);\n";
      os << "        break;\n";
      os << "      }\n";
    }

    if (intrinsic != MLIRDerivatives) {
      os << "      case DerivativeMode::ReverseModeGradient:\n";
      os << "      case DerivativeMode::ReverseModeCombined:{\n";
      os << "        IRBuilder<> Builder2(&" << origName << ");\n";
      os << "        getReverseBuilder(Builder2);\n";
      os << "        Value *dif = nullptr;\n";
    } else {
      os << "};\n";
      emitMLIRReverse(os, pattern, tree, intrinsic, origName, argOps);
    }

    emitReverseCommon(os, pattern, tree, intrinsic, origName, argOps);

    if (intrinsic != MLIRDerivatives) {
      os << "#ifdef ENZYME_ENABLE_FPOPT\n";
      os << "        if (auto *logFunc = getFPOptLogger(" << origName
         << ".getModule(), \"enzymeLogGrad\")) {\n"
         << "          std::string idStr = getLogIdentifier(" << origName
         << ");\n"
         << "          Value *idValue = "
            "BuilderZ.CreateGlobalStringPtr(idStr);\n"
         << "            Value *diffValue = Builder2.CreateFPExt(dif, "
            "Type::getDoubleTy("
         << origName << ".getContext()));\n"
         << "            CallInst *logCallInst = Builder2.CreateCall(logFunc, "
            "{idValue, diffValue});\n"
         << "            logCallInst->setDebugLoc(gutils->getNewFromOriginal("
         << origName << ".getDebugLoc()));\n"
         << "        }\n";
      os << "#endif\n";

      os << "        auto found = gutils->invertedPointers.find(&(" << origName
         << "));\n";
      os << "        if (found != gutils->invertedPointers.end() && "
            "!isa<PointerType>("
         << origName << ".getType())) {\n";
      os << "          PHINode* PN = dyn_cast<PHINode>(&*found->second);\n";
      os << "          if (!PN) {\n";
      os << "            std::string str;\n";
      os << "            raw_string_ostream ss(str);\n";
      os << "            ss << \"Shadow of instruction is not phi:\\n\";\n";
      os << "            ss << *gutils->oldFunc << \"\\n\";\n";
      os << "            ss << *gutils->newFunc << \"\\n\";\n";
      os << "            ss << \"orig: \" << " << origName << " << \"\\n\";\n";
      os << "            ss << \"found: \" << *found->second << \"\\n\";\n";
      os << "            if (CustomErrorHandler) {\n";
      os << "              CustomErrorHandler(str.c_str(), wrap(&(" << origName
         << ")), ErrorType::InternalError,\n";
      os << "                                 nullptr, nullptr, nullptr);\n";
      os << "            } else {\n";
      os << "              EmitFailure(\"PHIError\", (" << origName
         << ").getDebugLoc(), &(" << origName << "), ss.str());\n";
      os << "            }\n";
      os << "          }\n";
      os << "          assert(PN);\n";
      os << "          gutils->invertedPointers.erase(found);\n";
      os << "          gutils->erase(PN);\n";
      os << "        }\n";
      os << "        break;\n";
      os << "      }\n";

      os << "      case DerivativeMode::ReverseModePrimal:{\n";
      os << "        auto found = gutils->invertedPointers.find(&(" << origName
         << "));\n";
      os << "        if (found != gutils->invertedPointers.end() && "
            "!isa<PointerType>("
         << origName << ".getType())) {\n";
      os << "          PHINode* PN = dyn_cast<PHINode>(&*found->second);\n";
      os << "          if (!PN) {\n";
      os << "            std::string str;\n";
      os << "            raw_string_ostream ss(str);\n";
      os << "            ss << \"Shadow of instruction is not phi:\\n\";\n";
      os << "            ss << *gutils->oldFunc << \"\\n\";\n";
      os << "            ss << *gutils->newFunc << \"\\n\";\n";
      os << "            ss << \"orig: \" << " << origName << " << \"\\n\";\n";
      os << "            ss << \"found: \" << *found->second << \"\\n\";\n";
      os << "            if (CustomErrorHandler) {\n";
      os << "              CustomErrorHandler(str.c_str(), wrap(&(" << origName
         << ")), ErrorType::InternalError,\n";
      os << "                                 nullptr, nullptr, nullptr);\n";
      os << "            } else {\n";
      os << "              EmitFailure(\"PHIError\", (" << origName
         << ").getDebugLoc(), &(" << origName << "), ss.str());\n";
      os << "            }\n";
      os << "          }\n";
      os << "          assert(PN);\n";
      os << "          gutils->invertedPointers.erase(found);\n";
      os << "          gutils->erase(PN);\n";
      os << "        }\n";
      // TODO
      os << "        break;\n";
      os << "      }\n";
      os << "    }\n";
    }

    if (intrinsic == IntrDerivatives || intrinsic == CallDerivatives)
      os << "    return true;\n  }\n";
    else if (intrinsic == MLIRDerivatives)
      os << "    return success();\n  }\n";
    else
      os << "    return;\n  }\n";
    if (intrinsic == MLIRDerivatives)
      os << "};\n\n";
  }

  if (intrinsic == MLIRDerivatives) {
    const auto &actpatterns =
        recordKeeper.getAllDerivedDefinitions("InactiveOp");
    for (auto &pattern : actpatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "struct " << opName << "Activity : \n";
      os << "			public ActivityOpInterface::ExternalModel<"
         << opName << "Activity, " << dialect << "::" << opName << "> {\n";
      os << "  bool isInactive(mlir::Operation*) const { return true; }\n";
      os << "  bool isArgInactive(mlir::Operation*, size_t) const { "
            "return true; }\n";
      os << "};\n";
    }
    const auto &cfpatterns =
        recordKeeper.getAllDerivedDefinitions("ControlFlowOp");

    const auto &mempatterns =
        recordKeeper.getAllDerivedDefinitions("MemoryIdentityOp");

    for (auto &pattern : cfpatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      auto impl = pattern->getValueAsString("impl");
      os << "struct " << opName << "CF : \n";
      os << "			public "
            "ControlFlowAutoDiffOpInterface::ExternalModel<"
         << opName << "CF, " << dialect << "::" << opName << "> {\n";
      os << impl << "\n";
      os << "};\n";
    }

    for (auto &pattern : mempatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      auto diffargs = pattern->getValueAsListOfInts("ptrargs");
      auto storedargs = pattern->getValueAsListOfInts("storedargs");
      os << "struct " << opName << "MemActivity : \n";
      os << "     public ActivityOpInterface::ExternalModel<" << opName
         << "MemActivity, " << dialect << "::" << opName << "> {\n";
      os << "  bool isInactive(mlir::Operation* op) const {\n";
      os << "    for (size_t i=0, len=op->getNumOperands(); i<len; i++)\n";
      os << "      if (!isArgInactive(op, i)) return false;\n";
      os << "    return true;\n";
      os << "  };\n";
      os << "  bool isArgInactive(mlir::Operation*, size_t idx) const {\n";
      for (auto diffarg : diffargs) {
        if (diffarg == -1) {
          os << "    return false;\n";
          break;
        }
        os << "    if (idx == " << diffarg << ") return false;\n";
      }
      for (auto diffarg : storedargs) {
        if (diffarg == -1) {
          os << "    return false;\n";
          break;
        }
        os << "    if (idx == " << diffarg << ") return false;\n";
      }
      os << "    return true;\n  }\n";
      os << "};\n";

      auto tree = pattern->getValueAsDag("PatternToMatch");

      if (tree->getOperator()->getAsString() != "Unimplemented") {
        auto argOps = pattern->getValueAsListInit("reverse");
        auto origName = "op";
        emitMLIRReverse(os, pattern, tree, intrinsic, origName, argOps);
        emitReverseCommon(os, pattern, tree, intrinsic, origName, argOps);
        os << "     return success();\n";
        os << "   }\n";
        os << " };\n";
      }
    }

    const auto &brpatterns = recordKeeper.getAllDerivedDefinitions("BranchOp");

    const auto &retpatterns = recordKeeper.getAllDerivedDefinitions("ReturnOp");

    const auto &regtpatterns =
        recordKeeper.getAllDerivedDefinitions("RegionTerminatorOp");

    const auto &allocpatterns =
        recordKeeper.getAllDerivedDefinitions("AllocationOp");

    os << "void registerInterfaces(MLIRContext* context) {\n";
    for (const Record *pattern : patterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "  " << dialect << "::" << opName << "::attachInterface<" << opName
         << "FwdDerivative>(*context);\n";
      os << "  " << dialect << "::" << opName << "::attachInterface<" << opName
         << "RevDerivative>(*context);\n";
    }
    for (const Record *pattern : actpatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "  " << dialect << "::" << opName << "::attachInterface<" << opName
         << "Activity>(*context);\n";
    }
    for (const Record *pattern : cfpatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "  " << dialect << "::" << opName << "::attachInterface<" << opName
         << "CF>(*context);\n";
      os << "  registerAutoDiffUsingControlFlowInterface<" << dialect
         << "::" << opName << ">(*context);\n";
    }
    for (const Record *pattern : mempatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "  " << dialect << "::" << opName << "::attachInterface<" << opName
         << "MemActivity>(*context);\n";
      os << "  registerAutoDiffUsingMemoryIdentityInterface<" << dialect
         << "::" << opName;
      for (auto storedarg : pattern->getValueAsListOfInts("storedargs"))
        os << ", " << storedarg;
      os << ">(*context);\n";
      auto tree = pattern->getValueAsDag("PatternToMatch");
      if (tree->getOperator()->getAsString() != "Unimplemented") {
        os << "  " << dialect << "::" << opName << "::attachInterface<"
           << opName << "RevDerivative>(*context);\n";
      }
    }
    for (const Record *pattern : brpatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "  registerAutoDiffUsingBranchInterface<" << dialect
         << "::" << opName << ">(*context);\n";
    }
    for (const Record *pattern : regtpatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "  registerAutoDiffUsingRegionTerminatorInterface<" << dialect
         << "::" << opName << ">(*context);\n";
    }
    for (const Record *pattern : retpatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "  registerAutoDiffUsingReturnInterface<" << dialect
         << "::" << opName << ">(*context);\n";
    }
    for (const Record *pattern : allocpatterns) {
      auto opName = pattern->getValueAsString("opName");
      auto dialect = pattern->getValueAsString("dialect");
      os << "  registerAutoDiffUsingAllocationInterface<" << dialect
         << "::" << opName << ">(*context);\n";
    }
    os << "}\n";
  }
}

void emitDiffUse(const RecordKeeper &recordKeeper, raw_ostream &os,
                 ActionType intrinsic) {
  const char *patternNames;
  switch (intrinsic) {
  case MLIRDerivatives:
  case GenBlasDerivatives:
  case UpdateBlasDecl:
  case UpdateBlasTA:
  case GenBlasDiffUse:
  case GenHeaderVariables:
    llvm_unreachable("Cannot use blas updaters inside emitDiffUse");
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
  }
  const auto &patterns = recordKeeper.getAllDerivedDefinitions(patternNames);

  for (const Record *pattern : patterns) {
    auto tree = pattern->getValueAsDag("PatternToMatch");

    // Emit RewritePattern for Pattern.
    auto argOps = pattern->getValueAsListInit("ArgDerivatives");

    if (tree->getNumArgs() != argOps->size()) {
      PrintFatalError(pattern->getLoc(),
                      Twine("Defined rule pattern to have ") +
                          Twine(tree->getNumArgs()) +
                          " args but reverse rule array is a list of size " +
                          Twine(argOps->size()));
    }

    std::string origName;
    std::string prefix;
    switch (intrinsic) {
    case MLIRDerivatives:
    case GenBlasDerivatives:
    case UpdateBlasDecl:
    case UpdateBlasTA:
    case GenBlasDiffUse:
    case GenHeaderVariables:
      llvm_unreachable("Cannot use blas updaters inside emitDerivatives");
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
      origName = "CI";
#if LLVM_VERSION_MAJOR >= 14
      os << ") && CI->arg_size() == " << tree->getNumArgs() << " ){\n";
#else
      os << ") && CI->getNumArgOperands() == " << tree->getNumArgs() << " ){\n";
#endif
      prefix = "  ";
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
          int min_int = 0;
          if (min.size() != 0 && min.getAsInteger(10, min_int)) {
            PrintFatalError(pattern->getLoc(),
                            "Could not parse min llvm version as int");
          }
          if (min.size() != 0 && LLVM_VERSION_MAJOR < min_int)
            continue;
          if (lst->size() >= 3) {
            auto max = cast<StringInit>(lst->getValues()[2])->getValue();
            int max_int = 0;
            if (max.size() != 0 && max.getAsInteger(10, max_int)) {
              PrintFatalError(pattern->getLoc(),
                              "Could not parse max llvm version as int");
            }
            if (max.size() != 0 && LLVM_VERSION_MAJOR > max_int)
              continue;
          }
        }
        os << "    case Intrinsic::" << name << ":\n";
        anyVersion = true;
      }
      if (!anyVersion)
        continue;
      origName = "CI";
      prefix = "    ";
      os << prefix << "{\n";
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
      os << "  case llvm::Instruction::" << name << ":{\n";

      origName = "user";
      prefix = "  ";
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

      os << "  case llvm::Instruction::" << name << ":{\n";
      origName = "BO";
      prefix = "  ";
      break;
    }
    }

    using StringTy = std::string;

    StringMap<std::tuple<StringTy, StringTy, bool>> varNameToCondition;

    std::function<void(const DagInit *, ArrayRef<unsigned>)> insert =
        [&](const DagInit *ptree, ArrayRef<unsigned> prev) {
          for (auto treeEn : llvm::enumerate(ptree->getArgs())) {
            auto tree = treeEn.value();
            auto name = ptree->getArgNameStr(treeEn.index());
            SmallVector<unsigned, 2> next(prev.begin(), prev.end());
            next.push_back(treeEn.index());
            if (auto dg = dyn_cast<DagInit>(tree))
              insert(dg, next);

            if (name.size()) {
              auto op = (Twine(origName) + "->getOperand(" + Twine(next[0]) +
                         ") == val")
                            .str();
              varNameToCondition[name] = std::make_tuple(op, "", false);
            }
          }
        };

    insert(tree, {});

    if (tree->getNameStr().size())
      varNameToCondition[tree->getNameStr()] =
          std::make_tuple("ILLEGAL", "ILLEGAL", false);

    printDiffUse(os, prefix, argOps, origName, intrinsic, tree,
                 varNameToCondition);
  }
}

#include "blasDeclUpdater.h"
#include "blasDiffUseUpdater.h"
#include "blasTAUpdater.h"

void emitMLIRDerivatives(RecordKeeper &records, raw_ostream &os);

#if LLVM_VERSION_MAJOR >= 20
static bool EnzymeTableGenMain(raw_ostream &os, const RecordKeeper &records)
#else
static bool EnzymeTableGenMain(raw_ostream &os, RecordKeeper &records)
#endif
{
  switch (action) {
  case MLIRDerivatives:
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

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

#include "caching.h"
#include "general.h"

// clang-format off
//


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

void emit_castvals(TGPattern &pattern, raw_ostream &os) {
  auto activeArgs = pattern.getActiveArgs();
  auto nameVec = pattern.getArgNames();
  os 
<< "  /* beginning castvalls */\n"
<< "  Type *castvals[" << activeArgs.size() << "];\n";

  for (size_t i = 0; i < activeArgs.size(); i++) {
    size_t argIdx = activeArgs[i];
    auto name = nameVec[argIdx];
    os 
<< "  if (auto PT = dyn_cast<PointerType>(type_" << name << "))\n"
<< "    castvals[" << i << "] = PT;\n"
<< "  else\n"
<< "    castvals[" << i << "] = PointerType::getUnqual(fpType);\n";
  }
  os 
<< "  Value *cacheval;\n\n"
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


void emit_beginning(TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  os
<< "\nbool handle_" << name
<< "(BlasInfo blas, llvm::CallInst &call, Function *called,\n"
<< "    const std::map<Argument *, bool> &uncacheable_args, Type *fpType) {\n"
<< "  \n"
<< "  CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));\n"
<< "  IRBuilder<> BuilderZ(newCall);\n"
<< "  BuilderZ.setFastMathFlags(getFast());\n"
<< "  IRBuilder<> allocationBuilder(gutils->inversionAllocs);\n"
<< "  allocationBuilder.setFastMathFlags(getFast());\n";
  // not yet needed for lv-1
//<< "  auto &DL = gutils->oldFunc->getParent()->getDataLayout();\n";
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
  // auto name = pattern->getName();
  DagInit *tree = pattern->getValueAsDag("PatternToMatch");
  int lenDagArgs = tree->getNumArgs();
  assert(numTypes == lenDagArgs);
  return activeArgs;
}

// only for testing
#include "llvm/IR/Type.h"


void emit_free_and_ending(TGPattern &pattern, raw_ostream &os) {
    os
<< "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "      Mode == DerivativeMode::ReverseModeGradient ||\n"
<< "      Mode == DerivativeMode::ForwardModeSplit) {\n"
<< "    if (shouldFree()) {\n";

    auto nameVec = pattern.getArgNames();
    auto typeMap = pattern.getArgTypeMap();
    for (size_t i = 0; i < nameVec.size(); i++) {
      if (typeMap.lookup(i) == argType::vincData) {
        auto name = nameVec[i];
        os
<< "      if (cache_" << name << ") {\n"
<< "        CreateDealloc(Builder2, data_ptr_" << name << ");\n"
<< "      }\n";
      }
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
        // TODO: verify x isn't user from data_x (as only adjoint of x will be used)
        for (auto user: vecUsers) {
          auto name = argOps->getArgNameStr(user);
          if (vecName == name)
            continue; // see above
          os 
<< ((first) ? "" : " || ") << "active_" << name;
          first = false;
        }
        os 
<< ") {\n"
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
  // TODO: verify StringRef here is no ub.
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

#include <sstream>


void emit_helper(TGPattern &pattern, raw_ostream &os) {
  PrintNote("function: " + pattern.getName());

  std::vector<size_t> fp_pos{};
  auto nameVec = pattern.getArgNames();
  assert(nameVec.size() > 0);
  auto argTypeMap = pattern.getArgTypeMap();

  auto actArgs = pattern.getActiveArgs();
  os 
<< "  auto calledArg = called->arg_begin();\n\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto name = nameVec[i];
    os 
<< "  auto arg_" << name << " = call.getArgOperand(" << i << ");\n"
<< "  auto type_" << name << " = arg_" << name << "->getType();\n"
<< "  bool uncacheable_" << name << " = uncacheable_args.find(calledArg)->second;\n"
<< "  calledArg++;\n";
    if (std::count(actArgs.begin(), actArgs.end(), i)) {
      os 
<< "  bool active_" << name << " = !gutils->isConstantValue(arg_"
<< name << ");\n";
    }
    os 
<< "\n";
  }
  

  os 
<< "  int num_active_fp = 0;\n";
  for (size_t i = 0; i < nameVec.size(); i++) {
    argType type = argTypeMap.lookup(i);
    if (type == argType::fp) {
      os
<< "  if (active_" << nameVec[i] << ")\n"
<< "    num_active_fp++;\n";
    }
  }

  for (auto name : llvm::enumerate(nameVec)) {
    assert(argTypeMap.count(name.index()) == 1);
    auto type = argTypeMap.lookup(name.index());
    if (type == argType::vincData) {
      os 
<< "  bool julia_decl = !type_" << name.value() << "->isPointerTy();\n";
      return;
    }
  }
 
  PrintFatalError("Blas function without vector?");
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
        valueTypes.append((Twine("cache_") + floatName + " ? ValueType::None : ValueType::Primal").str());
      }
    } else if (type == argType::vincData) {
      auto nextName = nameVec[pos + 1];
      auto nextType = typeMap.lookup(pos + 1);
      assert(nextType == argType::vincInc);
      auto vecName = nameVec[pos];
      if (pos == actPos) {
        valueTypes.append("ValueType::Shadow, ValueType::None");
      } else {
        valueTypes.append((Twine("cache_") + vecName + " ? ValueType::None : ValueType::Primal, ValueType::None").str());
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


// TODO: think about how to handle nested rules which aren't simple calling another BLAS fnc.

size_t pattern_call_args(TGPattern &pattern, size_t actArg, llvm::SmallString<40> &result) {
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
      PrintNote("call_arg_helper_len. Pos: " + std::to_string(pos) + ", name: " + name + ", argPosition: "+ std::to_string(pos));
      auto out = (Twine("len_") + name).str();
      result.append((Twine("len_") + name).str());
    } else if (typeOfArg == argType::fp) {
      PrintNote("call_arg_helper_fp. Pos: " + std::to_string(pos) + ", name: " + name + ", argPosition: "+ std::to_string(pos));
      if (pos == actArg) {
        result.append((Twine("d_") + name).str());
      } else {
        result.append((Twine("fp_") + name).str());
      }
    } else if (typeOfArg == argType::vincData) {
      auto nextName = nameVec[pos+1];
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
      PrintNote("call_arg_helper_vincInc. Pos: " + std::to_string(pos) + ", name: " + name + ", argPosition: "+ std::to_string(pos));
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
size_t rule_call_args(Rule &rule, size_t actArg, llvm::SmallString<40> &result) {

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
    PrintNote("call_arg_helper: " + ruleDag->getArgNameStr(pos));
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
        //result.append((Twine("input_") + name).str());
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
        PrintNote("call_arg_helper_len. Pos: " + std::to_string(pos) + ", name: " + name + ", argPosition: "+ std::to_string(argPosition));
        auto out = (Twine("len_") + name).str();
        result.append((Twine("len_") + name).str());
      } else if (typeOfArg == argType::fp) {
        PrintNote("call_arg_helper_fp. Pos: " + std::to_string(pos) + ", name: " + name + ", argPosition: "+ std::to_string(argPosition));
        if (argPosition == actArg) {
          result.append((Twine("d_") + name).str());
        } else {
          result.append((Twine("fp_") + name).str());
        }
      } else if (typeOfArg == argType::vincData) {
        auto nextName = ruleDag->getArgNameStr(pos+1);
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
        PrintNote("call_arg_helper_vincInc. Pos: " + std::to_string(pos) + ", name: " + name + ", argPosition: "+ std::to_string(argPosition));
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

//void emit_deriv_fnc(DagInit *resultTree, const llvm::DenseMap<StringRef, StringRef> typeOfArgName,
//    llvm::StringSet<> &handled, const llvm::StringMap<llvm::SmallSet<size_t, 3>> mutables, raw_ostream &os) {
void emit_deriv_fnc(StringMap<TGPattern> &patternMap, Rule &rule, llvm::StringSet<> &handled, raw_ostream &os) {
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
    os 
<< "    auto derivcall_" << dfnc_name << " = gutils->oldFunc->getParent()->getOrInsertFunction(\n"
<< "      (blas.prefix + blas.floatType + \"" << dfnc_name << "\" + blas.suffix).str(), " << retTy << ",\n";
      // insert arg types based on .td file 
      bool first = true;
      std::vector<StringRef> usedArgStrs{};
      for (size_t i = 0; i < ruleDag->getNumArgs(); i++) {
        Init* subArg = ruleDag->getArg(i);
        if (DefInit *def = dyn_cast<DefInit>(subArg)) {
          auto Def = def->getDef();
          usedArgStrs.push_back(""); // no need to process later
          std::string typeToAdd = "";                           
          if (Def->isSubClassOf("DiffeRet")) {
            typeToAdd = "byRef ? PointerType::getUnqual(call.getType()) : call.getType()\n";
          } else if (Def->isSubClassOf("input")) {
            auto argStr = Def->getValueAsString("name");
            size_t argPos = nameMap.lookup(argStr);
            llvm::errs() << argStr << " : argStr: " << mutableArgs.count(argPos) << "\n";
            //assert(mutableArgs.count(i) == 1);
            // primary and adj have the same type
            typeToAdd = (Twine("type_") + argStr).str();
            usedArgStrs.push_back((Twine("input_") + argStr).str());
          } else if (Def->isSubClassOf("adj")) {
            auto argStr = Def->getValueAsString("name");
            // primary and adj have the same type
            typeToAdd = (Twine("type_") + argStr).str();
            //assert(mutables.count(argStr) == 1);
            usedArgStrs.push_back((Twine("adj_") + argStr).str());
          } else if (Def->isSubClassOf("Constant")) {
            typeToAdd = "fpType";
          } else {
            PrintFatalError(Def->getLoc(), "PANIC! Unsupported Definit");
          }
          os
<< ((first) ? "" : ", ") << typeToAdd;
        } else {
          auto argStr = ruleDag->getArgNameStr(i);
          os 
<< ((first) ? "" : ", ") << "type_" << argStr; 
          usedArgStrs.push_back(argStr);
        }
        first = false;
        }
      os 
<< ");\n";
    if (dfnc_name == "dot") {
      os 
<< "    assert(derivcall_dot.getFunctionType()->getReturnType() == fpType);\n";
    }
    os
<< "#if LLVM_VERSION_MAJOR >= 9\n"
<< "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name << ".getCallee()))\n"
<< "#else\n"
<< "    if (auto F = dyn_cast<Function>(derivcall_" << dfnc_name << "))\n"
<< "#endif\n"
<< "    {\n"
<< "      F->addFnAttr(Attribute::ArgMemOnly);\n"
<< "      if (byRef) {\n";
  for (size_t argPos = 0; argPos < usedArgStrs.size(); argPos++) {
    auto typeOfArg = typeMap.lookup(argPos);
    if (typeOfArg == argType::len || typeOfArg == argType::vincInc) {
      os 
<< "        F->addParamAttr(" << argPos << ", Attribute::ReadOnly);\n"
<< "        F->addParamAttr(" << argPos << ", Attribute::NoCapture);\n";
    }
  }
  os    
<< "      }\n"
<< "      // Julia declares double* pointers as Int64,\n"
<< "      //  so LLVM won't let us add these Attributes.\n"
<< "      if (!julia_decl) {\n";
  for (size_t argPos = 0; argPos < usedArgStrs.size(); argPos++) {
    auto typeOfArg = typeMap.lookup(argPos);
    if (typeOfArg == argType::vincData) {
      os 
<< "        F->addParamAttr(" << argPos << ", Attribute::NoCapture);\n";
        if (mutableArgs.count(argPos) == 0) {
          // Only emit ReadOnly if the arg isn't mutable
          os 
<< "        F->addParamAttr(" << argPos << ", Attribute::ReadOnly);\n";
      }
    }
  }
  os
<< "      }\n"
<< "    }\n\n";
  } else if (Def->isSubClassOf("MagicInst") && Def->getName() == "noop") {
    // nothing to prepare
  } else if (Def->isSubClassOf("DiffeRet")) {
    // nothing to prepare
  } else if (Def->isSubClassOf("Inst")) {
//TODO:
    PrintFatalError("Unhandled Inst Rule!");
  } else {
    PrintFatalError("Unhandled deriv Rule!");
  }
}

void emit_rev_rewrite_rules(StringMap<TGPattern> patternMap, TGPattern &pattern, raw_ostream &os) {

  auto nameVec = pattern.getArgNames();
  auto typeMap = pattern.getArgTypeMap();
  auto rules   = pattern.getRules();
  auto activeArgs = pattern.getActiveArgs();

  //ListInit *derivOps = pattern->getValueAsListInit("ArgDerivatives"); // correct
  //DagInit *argOps = pattern->getValueAsDag("PatternToMatch");
  //std::vector<Record *> inputTypes = pattern->getValueAsListOfDefs("inputTypes");
  
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
      //DagInit *dagArg = cast<DagInit>(arg);
      //llvm::errs() << "argName: " << dagArg->getName() << "\n";
      //hasDiffeRetVal |= hasDiffeRet(dagArg);
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
  llvm::errs() << "\n\n" << pattern.getName() << hasDiffeRetVal << "\n\n";

  os 
<< "  /* rev-rewrite */                                 \n"
<< "  if (Mode == DerivativeMode::ReverseModeCombined ||\n"
<< "      Mode == DerivativeMode::ReverseModeGradient) {\n"
<< "    Value *alloc = nullptr;\n"
<< "    if (byRef) {\n"
<< "      alloc = allocationBuilder.CreateAlloca(fpType);\n"
<< "    }\n\n";
  if (hasDiffeRetVal) {
    os
<< "    Value *dif = diffe(&call, Builder2);\n";
  }

  // TODO: adj_ args
//  os
//<< "    Value *adj_" << name << " = lookup(gutils->invertPointerM(call.getArgOperand(arg_" << name << "), Builder2))\n";

  llvm::StringSet handled{}; // We only emit one derivcall per blass call type
  for (auto rule : rules) {
    emit_deriv_fnc(patternMap, rule, handled, os);
  }
 
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto name = nameVec[i];
    auto typeOfArg = typeMap.lookup(i);
    if (typeOfArg == argType::vincData) {
  os
<< "    Value *d_" << name << " = active_" << name << "\n"
<< "     ? lookup(gutils->invertPointerM(arg_" << name << ", Builder2), Builder2)\n"
<< "     : nullptr;\n";
    }
    else if (typeOfArg == argType::fp) {
  os
<< "    Value *d_" << name << " = UndefValue::get(fpType);\n";
    }
  }

  os 
<< "    applyChainRule(\n"
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
    os 
<< ((first) ? "" : ", ") << "Value *" << "d_" + name;
    first = false;
  }

  if (hasDiffeRetVal) {
  os 
<< ((first) ? "" : ", ") << "Value *dif) {\n"
<< "        if (byRef) {\n"
<< "          Builder2.CreateStore(dif, alloc);\n"
<< "          dif = alloc;\n"
<< "        }\n"
<< "        unsigned int idx = 0;\n";
  } else {
  os 
<< ") {\n"
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
      os
<< "      if (active_" << name << ") {\n"
<< "        Value *toadd = dif;\n"
<< "        addToDiffe(arg_" << name <<", toadd, Builder2, type_" << name << ");\n"
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
      os
<< "      if (" << actCondition << ") {\n"
<< "        Value *args1[" << numArgs << "] = {" << args << "};\n"
<< "        auto Defs = gutils->getInvertedBundles(&call, {" << valueTypes << "}, Builder2, /* lookup */ true);\n";

      if (typeOfArg == argType::fp) {
        // extra handling, since we will update only a fp scalar as part of the return struct
        // it's presumably done by setting it to the value returned by this call
        os
<< "        CallInst *cubcall = cast<CallInst>(Builder2.CreateCall(derivcall_" << dfnc_name << ", args1, Defs));\n"
<< "        addToDiffe(arg_" << name << ", cubcall, Builder2, fpType);\n"
<< "        idx++;\n"
<< "      }\n";
      } else {
      os
<< "        Builder2.CreateCall(derivcall_" << dfnc_name << ", args1, Defs);\n"
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
  os 
<< "    },\n"
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
  os 
<< ((first) ? "" : ", ") << "dif);\n"
<< "  setDiffe(\n"
<< "    &call,\n"
<< "    Constant::getNullValue(gutils->getShadowType(call.getType())),\n"
<< "    Builder2);\n"
<< "  }\n";
  } else {
  os
<< "  );\n"
<< "  }\n";
  }
}

void emit_fwd_rewrite_rules(TGPattern &pattern, raw_ostream &os) {
  auto rules = pattern.getRules();
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

  auto nameVec = pattern.getArgNames();
  auto inputTypes = pattern.getArgTypeMap();
  auto activeArgs = pattern.getActiveArgs();
  for (auto inputType : inputTypes) {
    if (inputType.second == argType::vincData) {
      auto name = nameVec[inputType.first];
  os
<< "    Value *d_" << name << " = active_" << name << "\n"
<< "     ? gutils->invertPointerM(arg_" << name << ", Builder2)\n"
<< "     : nullptr;\n";
    }
    if (inputType.second == argType::fp) {
      auto name = nameVec[inputType.first];
  os
    // Done: revert Undef to ConstantFP
//<< "    Value *d_" << name << " = UndefValue::get(fpType);\n";
<< "    Value *d_" << name << " = llvm::ConstantFP::get(fpType, 0.0);\n";
    }
  }

  os
<< "    Value *dres = applyChainRule(\n"
<< "        call.getType(), Builder2,\n"
<< "        [&](";
  bool first = true;
  for (auto activeArg : activeArgs) {
    auto name = nameVec[activeArg];
    os 
<< ((first) ? "" : ", ") << "Value *d_" << name; 
    first = false;
  }
  os
<< "  ) {\n"
<< "      Value *dres = nullptr;\n";

 
  for (size_t i = 0; i < activeArgs.size(); i++) {
    auto activeArg = activeArgs[i];
    auto rule = rules[i];
    auto actName = nameVec[activeArg];
    auto dcallArgs = llvm::SmallString<40>();
    size_t numArgs = pattern_call_args(pattern, activeArg, dcallArgs);
    auto valueTypes = ValueType_helper(pattern, activeArg);
    os
<< "      if(active_" << actName << ") {\n"
<< "        Value *args1[" << numArgs << "] = {" << dcallArgs << "};\n\n"
<< "        auto Defs = gutils->getInvertedBundles(\n"
<< "          &call, {" << valueTypes << "}, Builder2, /* lookup */ false);\n";
  if (i == 0) {
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
  }
  os 
<< "      return dres;\n"
<< "    },\n"
<< "    ";

  first = true;
  for (auto activeArg : activeArgs) {
    os 
<< ((first) ? "" : ", ") << "d_" + nameVec[activeArg];
    first = false;
  }
  os 
<< ");\n"
<< "    setDiffe(&call, dres, Builder2);\n"
<< "  }\n";
}



void emit_handleBLAS(const std::vector<TGPattern> &blasPatterns, raw_ostream &os) {
  std::string handledBlasFunctions = "";
  bool first = true;
  for (auto blasPattern : blasPatterns) {
    auto newName = Twine((first) ? "" : ", ") + "\"" + blasPattern.getName() + "\"";
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
    auto name = pattern.getName();
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
      llvm::errs() << pattern->getName() << "\n";
      checkBlasCallsInDag(RK, blasPatterns, pattern->getName(), resultRoot);
    }
  }
}

static void checkBlasCalls2(std::vector<TGPattern> blasPatterns) {
  for (auto pattern : blasPatterns) {
  }
}


llvm::StringMap<llvm::SmallSet<size_t, 3>> getMutableArgs(const std::vector<Record *> blasPatterns) {
  llvm::StringMap<llvm::SmallSet<size_t, 3>> res{};
  for (auto pattern : blasPatterns) {
    auto name = pattern->getName();
    auto args = pattern->getValueAsDag("PatternToMatch");
    llvm::SmallSet<size_t, 3> mutArgs{};
    auto mutableArgs = pattern->getValueAsListOfStrings("mutable");
    // We must replace their names by their position
    for (auto mutableArg : mutableArgs) {
      size_t pos = 0;
      while (args->getArgNameStr(pos) != mutableArg) {
        pos++;
        if (pos == args->getNumArgs()) {
            PrintFatalError("mutable arg isn't an input Arg!");
        }
      }
      mutArgs.insert(pos);
    }

    res.insert(std::pair<std::string, llvm::SmallSet<size_t, 3>>(name.str(), mutArgs));
  }
  return res;
}

// NEXT TODO: for input args (vectors) being overwritten.
// Cache them and use the cache later

/*
 * We create the following variables:
 */
void emitBlasDerivatives(const RecordKeeper &RK, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = RK.getAllDerivedDefinitions("CallPattern");
  const auto &blasPatterns = RK.getAllDerivedDefinitions("CallBlasPattern");
  const auto &blas_modes = RK.getAllDerivedDefinitions("blas_modes");
  Record *attrClass = RK.getClass("Attr");


  // NEW //////////////////// 
  std::vector<TGPattern> newBlasPatterns{};
  StringMap<TGPattern> patternMap;
  for (auto pattern : blasPatterns) {
    auto parsedPattern = TGPattern(*pattern);
    newBlasPatterns.push_back(TGPattern(*pattern));
    auto newEntry = std::pair<std::string, TGPattern>(parsedPattern.getName(), parsedPattern);
    patternMap.insert(newEntry);
  }


  // Make sure that we only call blass function b for calculating the derivative
  // of a iff we have defined b and pass the right amount of parameters.
  // TODO: type check params, as far as possible
  checkBlasCalls(RK, blasPatterns);
  //checkBlasCalls2(newBlasPatterns);
  emit_handleBLAS(newBlasPatterns, os);
  // emitEnumMatcher(blas_modes, os);
  
  for (size_t i = 0; i < blasPatterns.size(); i++) {
  //for (auto pattern : blasPatterns) {
    auto pattern = blasPatterns[i];
    auto newPattern = newBlasPatterns[i];


    std::vector<size_t> posActArgs = getPossiblyActiveArgs(pattern);

    // For each input arg, we store a set including all users (by index).
    llvm::DenseMap<size_t, llvm::SmallSet<size_t, 5>> argUsers = getUsedInputs(pattern, posActArgs);

    emit_beginning(newPattern, os);
    emit_helper(newPattern, os);
    emit_castvals(newPattern, os);
    emit_scalar_types(newPattern, os);

    emit_caching(newPattern, argUsers, os);
    //emit_caching(pattern, posActArgs, argUsers, os);
    emit_extract_calls(pattern, posActArgs, argUsers, os);

    emit_fwd_rewrite_rules(newPattern, os);
    emit_rev_rewrite_rules(patternMap, newPattern, os);

    // writeEnums(pattern, blas_modes, os);
    emit_free_and_ending(newPattern, os);
  }
}


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

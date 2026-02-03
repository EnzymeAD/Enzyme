#ifndef ENZYME_TBLGEN_BLAS_DECL_UPDATER_H
#define ENZYME_TBLGEN_BLAS_DECL_UPDATER_H

#include "datastructures.h"

inline void emit_attributeBLASCaller(ArrayRef<TGPattern> blasPatterns,
                                     raw_ostream &os) {
  os << "void attributeBLAS(BlasInfo blas, llvm::Function *F) {             \n";
  os << "  if (!F->empty())\n";
  os << "    return;\n";
  for (auto &&pattern : blasPatterns) {
    auto name = pattern.getName();
    os << "  if (blas.function == \"" << name << "\") {                   \n"
       << "    attribute_" << name << "(blas, F);                         \n"
       << "    return;                                                    \n"
       << "  }                                                            \n";
  }
  os << "}                                                                \n";
}

inline void emit_attributeBLAS(const TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  bool lv23 = pattern.isBLASLevel2or3();
  os << "llvm::Constant* attribute_" << name
     << "(BlasInfo blas, llvm::Function *F) {\n";
  os << "  if (!F->empty())\n";
  os << "    return F;\n";
  os << "  llvm::Type *fpType = blas.fpType(F->getContext());\n";
  os << "  const bool byRef = blas.prefix == \"\" || blas.prefix == "
        "\"cublas_\";\n";
  os << "const bool byRefFloat = byRef || blas.prefix == \"cublas\";\n";
  os << "(void)byRefFloat;\n";
  os << "  const bool cblas = blas.prefix == \"cblas_\";\n";
  os << "  const bool cublas = blas.prefix == \"cublas_\" || blas.prefix == "
        "\"cublas\";\n";
  os << "#if LLVM_VERSION_MAJOR >= 16\n"
     << "  F->setOnlyAccessesArgMemory();\n"
     << "#else\n"
     << "  F->addFnAttr(llvm::Attribute::ArgMemOnly);\n"
     << "#endif\n"
     << "  F->addFnAttr(llvm::Attribute::NoUnwind);\n"
     << "  F->addFnAttr(llvm::Attribute::NoRecurse);\n"
     << "#if LLVM_VERSION_MAJOR >= 14\n"
     << "  F->addFnAttr(llvm::Attribute::WillReturn);\n"
     << "  F->addFnAttr(llvm::Attribute::MustProgress);\n"
     << "#elif LLVM_VERSION_MAJOR >= 12\n"
     << "  F->addAttribute(llvm::AttributeList::FunctionIndex, "
        "llvm::Attribute::WillReturn);\n"
     << "  F->addAttribute(llvm::AttributeList::FunctionIndex, "
        "llvm::Attribute::MustProgress);\n"
     << "#endif\n"
     << "  F->addFnAttr(llvm::Attribute::NoFree);\n"
     << "  F->addFnAttr(llvm::Attribute::NoSync);\n"
     << "  F->addFnAttr(\"enzyme_no_escaping_allocation\");\n";

  auto argTypeMap = pattern.getArgTypeMap();
  DenseSet<size_t> mutableArgs = pattern.getMutableArgs();

  if (mutableArgs.size() == 0) {
    // under cublas, these functions have an extra write-only return ptr
    // argument
    if (has_active_return(name)) {
      os << "  if (!cublas) {\n";
    }
    os << "#if LLVM_VERSION_MAJOR >= 16\n";
    os << "  F->setOnlyReadsMemory();\n";
    os << "#else\n";
    os << "  F->removeFnAttr(llvm::Attribute::ReadNone);\n";
    os << "  F->addFnAttr(llvm::Attribute::ReadOnly);\n";
    os << "#endif\n";
    if (has_active_return(name)) {
      os << "  }\n";
    }
  }

  os << "  const int offset = (";
  if (lv23) {
    os << "(cblas || cublas)";
  } else {
    os << "cublas";
  }
  os << " ? 1 : 0);\n";

  os << "  llvm::Constant *res = F;\n";
  os << "  llvm::SmallVector<llvm::Type*, 1> argTys;\n";
  os << "  auto prevFT = F->getFunctionType();\n";
  os << "  if (offset) argTys.push_back(prevFT->getParamType(0));\n";

  size_t numArgs = argTypeMap.size();

  int numChars = 0;
  for (size_t argPos = 0; argPos < numArgs; argPos++) {
    if (argPos == 0 && lv23)
      continue;
    auto typeOfArg = argTypeMap.lookup(argPos);
    if (typeOfArg == ArgType::vincData || typeOfArg == ArgType::mldData) {
      os << "  "
            "argTys.push_back(llvm::isa<llvm::PointerType>(prevFT->"
            "getParamType(argTys.size())) ? "
            "prevFT->getParamType(argTys.size()) : "
            "getUnqual(fpType));\n";
    } else {
      os << "  argTys.push_back(prevFT->getParamType(argTys.size()));\n";
      if (typeOfArg == ArgType::uplo || typeOfArg == ArgType::trans ||
          typeOfArg == ArgType::diag || typeOfArg == ArgType::side) {
        numChars++;
      }
    }
  }
  os << "  if (!cublas && !cblas) {\n";
  for (int i = 0; i < numChars; i++) {
    os << "  if (prevFT->getNumParams() > argTys.size())";
    os << "    argTys.push_back(prevFT->getParamType(argTys.size()));\n";
    os << "  else";
    os << "    argTys.push_back(blas.intType(F->getContext()));\n";
    os << "  F->addParamAttr(argTys.size()-1, "
          "llvm::Attribute::get(F->getContext(), llvm::Attribute::ZExt));\n";
  }
  os << "  }\n";

  if (has_active_return(name)) {
    os << "const bool cublasv2 = blas.prefix == "
          "\"cublas\" && llvm::StringRef(blas.suffix).contains(\"v2\");\n";
    os << "  if (cublasv2) argTys.push_back(getUnqual(fpType));\n";
  }

  os << "  auto nextFT = llvm::FunctionType::get(prevFT->getReturnType(), "
        "argTys, false);\n";
  os << "  if (nextFT != prevFT && F->empty()) {\n";
  os << "    auto F2 = llvm::Function::Create(nextFT, F->getLinkage(), \"\", "
        "F->getParent());\n";
  os << "    F->replaceAllUsesWith(llvm::ConstantExpr::getPointerCast(F2, "
        "F->getType()));\n";
  os << "    res = llvm::ConstantExpr::getPointerCast(F2, "
        "F->getType());\n";
  os << "    F2->copyAttributesFrom(F);\n";
  os << "    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 1> MD;\n";
  os << "    F->getAllMetadata(MD);\n";
  os << "    for (auto pair : MD)\n";
  os << "      F2->addMetadata(pair.first, *pair.second);\n";
  os << "    F2->takeName(F);\n";
  os << "    F2->setCallingConv(F->getCallingConv());\n";
  os << "    F->eraseFromParent();\n";
  os << "    F = F2;\n";
  os << "  }\n";

  for (size_t argPos = 0; argPos < numArgs; argPos++) {
    const auto typeOfArg = argTypeMap.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);

    if (is_char_arg(typeOfArg) || typeOfArg == ArgType::len ||
        typeOfArg == ArgType::vincInc || typeOfArg == ArgType::mldLD) {
      os << "  F->addParamAttr(" << i << " + offset"
         << ", llvm::Attribute::get(F->getContext(), \"enzyme_inactive\"));\n";
    }
  }

  for (size_t argPos = 0; argPos < numArgs; argPos++) {
    const auto typeOfArg = argTypeMap.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);

    if (is_char_arg(typeOfArg) || typeOfArg == ArgType::len ||
        typeOfArg == ArgType::vincInc || typeOfArg == ArgType::fp ||
        typeOfArg == ArgType::mldLD) {
      os << "  if (" << (typeOfArg == ArgType::fp ? "byRefFloat" : "byRef")
         << ") {\n";
      os << "      F->removeParamAttr(" << i << " + offset"
         << ", llvm::Attribute::ReadNone);\n"
         << "      if (F->getFunctionType()->getParamType(" << i
         << " + offset)->isPointerTy()) {\n"
         << "        F->addParamAttr(" << i
         << " + offset, llvm::Attribute::ReadOnly);\n"
         << "        addFunctionNoCapture(F, " << i << " + offset);\n"
         << "      }\n"
         << "  }\n";
    }
  }

  for (size_t argPos = 0; argPos < numArgs; argPos++) {
    auto typeOfArg = argTypeMap.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);
    if (typeOfArg == ArgType::vincData || typeOfArg == ArgType::mldData) {
      os << "  addFunctionNoCapture(F, " << i << " + offset);\n";
      if (mutableArgs.count(argPos) == 0) {
        // Only emit ReadOnly if the arg isn't mutable
        os << "  F->removeParamAttr(" << i << " + offset"
           << ", llvm::Attribute::ReadNone);\n"
           << "  if (F->getFunctionType()->getParamType(" << i
           << " + offset)->isPointerTy())\n"
           << "    F->addParamAttr(" << i << " + offset"
           << ", llvm::Attribute::ReadOnly);\n";
      }
    }
  }

  if (has_active_return(name)) {
    // under cublas, these functions have an extra return ptr argument
    size_t ptrRetArg = argTypeMap.size();
    os << "  if (cublas) {\n"
       << "      F->removeParamAttr(" << ptrRetArg << " + offset"
       << ", llvm::Attribute::ReadNone);\n"
       << "      if (F->getFunctionType()->getParamType(" << ptrRetArg
       << " + offset)->isPointerTy()) {\n"
       << "        F->addParamAttr(" << ptrRetArg
       << " + offset, llvm::Attribute::WriteOnly);\n"
       << "        addFunctionNoCapture(F, " << ptrRetArg << " + offset);\n"
       << "      }\n"
       << "  }\n";
  }
  os << "  return res;\n";
  os << "}\n";
}

inline void emitBlasDeclUpdater(const RecordKeeper &RK, raw_ostream &os) {
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

  for (auto &&newPattern : newBlasPatterns) {
    emit_attributeBLAS(newPattern, os);
  }
  emit_attributeBLASCaller(newBlasPatterns, os);

  os << "bool attributeTablegen(llvm::Function &F) {\n";
  os << "  auto name = getFuncName(&F);\n";
  os << "  auto changed = false;\n";
  os << "  auto blasMetaData = extractBLAS(name);\n";
  os << "  if (F.empty() && blasMetaData) {\n";
  os << "    attributeBLAS(*blasMetaData, &F);\n";
  os << "    changed = true;\n";
  os << "  }\n";
  {
    const auto &patterns = RK.getAllDerivedDefinitions("CallPattern");
    for (const Record *pattern : patterns) {
      auto tree = pattern->getValueAsDag("PatternToMatch");
      os << "  if ((";
      bool prev = false;
      for (auto nameI : *pattern->getValueAsListInit("names")) {
        if (prev)
          os << " ||\n      ";
        os << "name == " << cast<StringInit>(nameI)->getAsString() << "";
        prev = true;
      }
      os << ") && F.getFunctionType()->getNumParams() == " << tree->getNumArgs()
         << " ){\n"
         << "    changed = true;\n";

      for (auto attr : *pattern->getValueAsListInit("FnAttrs")) {
        auto attrDef = cast<DefInit>(attr)->getDef();
        auto attrName = attrDef->getValueAsString("name");
        if (attrName == "ReadNone") {
          os << "  #if LLVM_VERSION_MAJOR >= 16\n";
          os << "    F.setOnlyReadsMemory();\n";
          os << "    F.setOnlyWritesMemory();\n";
          os << "  #elif LLVM_VERSION_MAJOR >= 14\n";
        } else if (attrName == "ReadOnly") {
          os << "  #if LLVM_VERSION_MAJOR >= 16\n";
          os << "    F.setOnlyReadsMemory();\n";
          os << "  #elif LLVM_VERSION_MAJOR >= 14\n";
        } else if (attrName == "WriteOnly") {
          os << "  #if LLVM_VERSION_MAJOR >= 16\n";
          os << "    F.setOnlyWritesMemory();\n";
          os << "  #elif LLVM_VERSION_MAJOR >= 14\n";
        } else
          os << "  #if LLVM_VERSION_MAJOR >= 14\n";
        os << "    F.addAttributeAtIndex(llvm::AttributeList::FunctionIndex, "
              "llvm::Attribute::get(F.getContext(), llvm::Attribute::"
           << attrName << "));\n";
        os << "  #else \n";
        os << "    F.addAttribute(llvm::AttributeList::FunctionIndex, "
              "llvm::Attribute::get(F.getContext(), llvm::Attribute::"
           << attrName << "));\n";
        os << "  #endif \n";
      }
      auto argOps = pattern->getValueAsListInit("ArgDerivatives");
      for (auto argOpEn : enumerate(*argOps)) {
        if (auto resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
          auto opName = resultRoot->getOperator()->getAsString();
          auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
          if (opName == "InactiveArgSpec" ||
              Def->isSubClassOf("InactiveArgSpec")) {
            if (!Def->getValueAsBit("asserting"))
              os << "    F.addParamAttr(" << argOpEn.index()
                 << ", llvm::Attribute::get(F.getContext(), "
                    "\"enzyme_inactive\"));\n";
            continue;
          }
        }
      }
      os << "  }\n";
    }
  }
  os << "  return changed;\n";
  os << "}\n";
}

#endif // ENZYME_TBLGEN_BLAS_DECL_UPDATER_H

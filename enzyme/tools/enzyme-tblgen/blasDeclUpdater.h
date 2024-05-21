#include "datastructures.h"

void emit_attributeBLASCaller(ArrayRef<TGPattern> blasPatterns,
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

void emit_attributeBLAS(const TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  bool lv23 = pattern.isBLASLevel2or3();
  os << "void attribute_" << name << "(BlasInfo blas, llvm::Function *F) {\n";
  os << "  if (!F->empty())\n";
  os << "    return;\n";
  os << "  const bool byRef = blas.prefix == \"\" || blas.prefix == "
        "\"cublas_\";\n";
  os << "const bool byRefFloat = byRef || blas.prefix == \"cublas\";\n";
  os << "(void)byRefFloat;\n";
  if (lv23)
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

  for (size_t i = 0; i < argTypeMap.size(); i++) {
    std::string floatPtrPos = std::to_string(lv23 ? (i - 1) : i);
    floatPtrPos += " + offset";

    auto ty = argTypeMap.lookup(i);
    if (ty == ArgType::vincData || ty == ArgType::mldData) {
      os << "const bool julia_decl = !F->getFunctionType()->getParamType("
         << floatPtrPos << ")->isPointerTy();\n";
      break;
    }
    if (i + 1 == argTypeMap.size()) {
      llvm::errs() << "Tablegen bug: BLAS fnc without vector of matrix!\n";
      llvm_unreachable("Tablegen bug: BLAS fnc without vector of matrix!");
    }
  }

  size_t numArgs = argTypeMap.size();

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
         << "      F->addParamAttr(" << i << " + offset"
         << ", llvm::Attribute::ReadOnly);\n"
         << "      F->addParamAttr(" << i << " + offset"
         << ", llvm::Attribute::NoCapture);\n";
      os << "  }\n";
    }
  }

  os << "  // Julia declares double* pointers as Int64,\n"
     << "  //  so LLVM won't let us add these Attributes.\n"
     << "  if (!julia_decl) {\n";
  for (size_t argPos = 0; argPos < numArgs; argPos++) {
    auto typeOfArg = argTypeMap.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);
    if (typeOfArg == ArgType::vincData || typeOfArg == ArgType::mldData) {
      os << "    F->addParamAttr(" << i << " + offset"
         << ", llvm::Attribute::NoCapture);\n";
      if (mutableArgs.count(argPos) == 0) {
        // Only emit ReadOnly if the arg isn't mutable
        os << "    F->removeParamAttr(" << i << " + offset"
           << ", llvm::Attribute::ReadNone);\n"
           << "    F->addParamAttr(" << i << " + offset"
           << ", llvm::Attribute::ReadOnly);\n";
      }
    }
  }
  os << "  } else {\n";
  for (size_t argPos = 0; argPos < argTypeMap.size(); argPos++) {
    auto typeOfArg = argTypeMap.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);
    if (typeOfArg == ArgType::vincData || typeOfArg == ArgType::mldData) {
      os << "    F->addParamAttr(" << i << " + offset"
         << ", llvm::Attribute::get(F->getContext(), \"enzyme_NoCapture\"));\n";
      if (mutableArgs.count(argPos) == 0) {
        // Only emit ReadOnly if the arg isn't mutable
        os << "    F->addParamAttr(" << i << " + offset"
           << ", llvm::Attribute::get(F->getContext(), "
              "\"enzyme_ReadOnly\"));\n";
      }
    }
  }
  os << "  }\n";

  if (has_active_return(name)) {
    // under cublas, these functions have an extra return ptr argument
    size_t ptrRetArg = argTypeMap.size();
    os << "  if (cublas) {\n"
       << "      F->removeParamAttr(" << ptrRetArg << " + offset"
       << ", llvm::Attribute::ReadNone);\n"
       << "      F->addParamAttr(" << ptrRetArg << " + offset"
       << ", llvm::Attribute::WriteOnly);\n"
       << "      F->addParamAttr(" << ptrRetArg << " + offset"
       << ", llvm::Attribute::NoCapture);\n"
       << "  }\n";
  }
  os << "}\n";
}

void emitBlasDeclUpdater(const RecordKeeper &RK, raw_ostream &os) {
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
    for (Record *pattern : patterns) {
      DagInit *tree = pattern->getValueAsDag("PatternToMatch");
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
      ListInit *argOps = pattern->getValueAsListInit("ArgDerivatives");
      for (auto argOpEn : enumerate(*argOps)) {
        if (DagInit *resultRoot = dyn_cast<DagInit>(argOpEn.value())) {
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

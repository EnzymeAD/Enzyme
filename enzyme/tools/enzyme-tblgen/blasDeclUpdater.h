#include "datastructures.h"

void emit_attributeBLASCaller(ArrayRef<TGPattern> blasPatterns,
                              raw_ostream &os) {
  os << "void attributeBLAS(BlasInfo blas, llvm::Function *F) {             \n";
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
  os << "void attribute_" << name << "(BlasInfo blas, llvm::Function *F) {\n"
     << "#if LLVM_VERSION_MAJOR >= 16\n"
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
     << "  F->addFnAttr(llvm::Attribute::NoSync);\n";

  auto argTypeMap = pattern.getArgTypeMap();
  DenseSet<size_t> mutableArgs = pattern.getMutableArgs();

  if (mutableArgs.size() == 0) {
    os << "#if LLVM_VERSION_MAJOR >= 16\n";
    os << "  F->setOnlyReadsMemory();\n";
    os << "#else\n";
    os << "  F->addFnAttr(llvm::Attribute::ReadOnly);\n";
    os << "#endif\n";
  }

  os << "const bool byRef = blas.prefix == \"\";\n";
  if (lv23)
    os << "const int offset = (byRef ? 0 : 1);\n";

  for (size_t i = 0; i < argTypeMap.size(); i++) {
    std::string floatPtrPos = std::to_string(lv23 ? (i - 1) : i);
    if (lv23)
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

  os << "  if (byRef) {\n";
  for (size_t argPos = 0; argPos < argTypeMap.size(); argPos++) {
    const auto typeOfArg = argTypeMap.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);
    if (typeOfArg == ArgType::len || typeOfArg == ArgType::vincInc ||
        typeOfArg == ArgType::fp || typeOfArg == ArgType::trans ||
        typeOfArg == ArgType::mldLD || typeOfArg == ArgType::uplo ||
        typeOfArg == ArgType::diag || typeOfArg == ArgType::side) {
      os << "      F->addParamAttr(" << i << (lv23 ? " + offset" : "")
         << ", llvm::Attribute::ReadOnly);\n"
         << "      F->addParamAttr(" << i << (lv23 ? " + offset" : "")
         << ", llvm::Attribute::NoCapture);\n";
    }
  }

  os << "  }\n"
     << "  // Julia declares double* pointers as Int64,\n"
     << "  //  so LLVM won't let us add these Attributes.\n"
     << "  if (!julia_decl) {\n";
  for (size_t argPos = 0; argPos < argTypeMap.size(); argPos++) {
    auto typeOfArg = argTypeMap.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);
    if (typeOfArg == ArgType::vincData || typeOfArg == ArgType::mldData) {
      os << "    F->addParamAttr(" << i << (lv23 ? " + offset" : "")
         << ", llvm::Attribute::NoCapture);\n";
      if (mutableArgs.count(argPos) == 0) {
        // Only emit ReadOnly if the arg isn't mutable
        os << "    F->addParamAttr(" << i << (lv23 ? " + offset" : "")
           << ", llvm::Attribute::ReadOnly);\n";
      }
    }
  }
  os << "  } else {\n";
  for (size_t argPos = 0; argPos < argTypeMap.size(); argPos++) {
    auto typeOfArg = argTypeMap.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);
    if (typeOfArg == ArgType::vincData || typeOfArg == ArgType::mldData) {
      os << "    F->addParamAttr(" << i << (lv23 ? " + offset" : "")
         << ", llvm::Attribute::get(F->getContext(), \"enzyme_NoCapture\"));\n";
      if (mutableArgs.count(argPos) == 0) {
        // Only emit ReadOnly if the arg isn't mutable
        os << "    F->addParamAttr(" << i << (lv23 ? " + offset" : "")
           << ", llvm::Attribute::get(F->getContext(), "
              "\"enzyme_ReadOnly\"));\n";
      }
    }
  }
  os << "  }\n"
     << "}\n";
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

  os << "void attributeTablegen(llvm::Function &F) {\n";
  os << "  auto name = getFuncName(&F);\n";
  os << "  auto blasMetaData = extractBLAS(name);\n";
  os << "  #if LLVM_VERSION_MAJOR >= 16\n";
  os << "    if (blasMetaData.has_value())\n";
  os << "      attributeBLAS(blasMetaData.value(), &F);\n";
  os << "  #else\n";
  os << "    if (blasMetaData.hasValue())\n";
  os << "      attributeBLAS(blasMetaData.getValue(), &F);\n";
  os << "  #endif\n";
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
         << " ){\n";
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
      os << "  }\n";
    }
  }

  os << "}\n";
}

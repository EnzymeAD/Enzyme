
void emit_attributeBLASCaller(const std::vector<TGPattern> &blasPatterns,
                     raw_ostream &os) {
  os << "void attributeBLAS(BlasInfo blas, llvm::Function *F) {             \n";
  for (auto pattern : blasPatterns) {
    auto name = pattern.getName();
    // only one which we expose right now.
    //if (name != "dot")
      continue;
    os << "  if (blas.function == \"" << name << "\") {                   \n"
       << "    attribute_" << name << "(blas, F);                         \n"
       << "    return;                                                    \n"
       << "  }                                                            \n";
  }
  os << "}                                                                \n";
}

void emit_attributeBLAS(TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  os << "void attribute_" << name << "(BlasInfo blas, llvm::Function *F) {\n"
     << "  F->addFnAttr(llvm::Attribute::ArgMemOnly);\n"
     << "  F->addFnAttr(llvm::Attribute::NoUnwind);\n"
     << "  F->addFnAttr(llvm::Attribute::NoRecurse);\n"
     << "#if LLVM_VERSION_MAJOR >= 14\n"
     << "  F->addFnAttr(llvm::Attribute::WillReturn);\n"
     << "  F->addFnAttr(llvm::Attribute::MustProgress);\n"
     << "#elif LLVM_VERSION_MAJOR >= 12\n"
     << "  F->addAttribute(AttributeList::FunctionIndex, "
        "Attribute::WillReturn);\n"
     << "  F->addAttribute(AttributeList::FunctionIndex, "
        "Attribute::MustProgress);\n"
     << "#endif\n"
     << "#if LLVM_VERSION_MAJOR >= 9\n"
     << "  F->addFnAttr(llvm::Attribute::NoFree);\n"
     << "  F->addFnAttr(llvm::Attribute::NoSync);\n"
     << "#else\n"
     << "    F->addFnAttr(\"nofree\");\n"
     << "#endif\n";

  auto argTypeMap = pattern.getArgTypeMap();
  DenseSet<size_t> mutableArgs = pattern.getMutableArgs();

  if (mutableArgs.size() == 0)
    os << "  F->addFnAttr(llvm::Attribute::ReadOnly);\n";

  for (size_t i = 0; i < argTypeMap.size(); i++) {
    if (argTypeMap.lookup(i) == argType::vincData) {
      os << "#if LLVM_VERSION_MAJOR >= 10\n"
         << "  const bool julia_decl = !F->getArg(" << i
         << ")->getType()->isPointerTy();\n"
         << "#else\n"
         << "  const bool julia_decl = !F->getOperand(" << i
         << ")->getType()->isPointerTy();\n"
         << "#endif\n";
      break;
    }
    if (i+1 == argTypeMap.size()) {
      llvm::errs() << "Tablegen bug: BLAS fnc without vector!\n";
      llvm_unreachable("Tablegen bug: BLAS fnc without vector!");
    }
  }
  for (size_t i = 0; i < argTypeMap.size(); i++) {
    if (argTypeMap.lookup(i) == argType::len) {
      os << "#if LLVM_VERSION_MAJOR >= 10\n"
         << "  const bool byRef = !F->getArg(" << i
         << ")->getType()->isIntegerTy() && blas.prefix == \"\";\n"
         << "#else\n"
         << "  const bool byRef = !F->getOperand(" << i
         << ")->getType()->isIntegerTy() && blas.prefix == \"\";\n"
         << "#endif\n";
      break;
    }
    if (i+1 == argTypeMap.size()) {
      llvm::errs() << "Tablegen bug: BLAS fnc without vector length!\n";
      llvm_unreachable("Tablegen bug: BLAS fnc without vector length!");
    }
  }

  os   << "  if (byRef) {\n";
  for (size_t argPos = 0; argPos < argTypeMap.size(); argPos++) {
    const auto typeOfArg = argTypeMap.lookup(argPos);
    if (typeOfArg == argType::len || typeOfArg == argType::vincInc) {
      os << "      F->addParamAttr(" << argPos
         << ", llvm::Attribute::ReadOnly);\n"
         << "      F->addParamAttr(" << argPos
         << ", llvm::Attribute::NoCapture);\n";
    }
  }
  os << "  }\n"
     << "  // Julia declares double* pointers as Int64,\n"
     << "  //  so LLVM won't let us add these Attributes.\n"
     << "  if (!julia_decl) {\n";
  for (size_t argPos = 0; argPos < argTypeMap.size(); argPos++) {
    auto typeOfArg = argTypeMap.lookup(argPos);
    if (typeOfArg == argType::vincData) {
      os << "    F->addParamAttr(" << argPos << ", llvm::Attribute::NoCapture);\n";
      if (mutableArgs.count(argPos) == 0) {
        // Only emit ReadOnly if the arg isn't mutable
        os << "    F->addParamAttr(" << argPos << ", llvm::Attribute::ReadOnly);\n";
      }
    }
  }
  os << "  }\n"
     << "}\n";
}

void emitBlasDeclUpdater(const RecordKeeper &RK, raw_ostream &os) {
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

  emit_recognizeBLAS(newBlasPatterns, os);
  for (auto newPattern : newBlasPatterns) {
    emit_attributeBLAS(newPattern, os);
  }
  emit_attributeBLASCaller(newBlasPatterns, os);
}

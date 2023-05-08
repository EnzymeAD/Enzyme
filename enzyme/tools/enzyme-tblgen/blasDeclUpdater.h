
void emit_attributeBLASCaller(const std::vector<TGPattern> &blasPatterns,
                     raw_ostream &os) {
  os << "void attributeBLAS(BlasInfo blas, llvm::Function *F) {             \n";
  for (auto pattern : blasPatterns) {
    auto name = pattern.getName();
    // only one which we expose right now.
    if (name != "dot")
      continue;
    os << "  if (blas.function == \"" << name << "\") {                     \n"
       << "      attribute_" << name << "(blas, F);                         \n";
  }
  os << "  } else {                                                       \n"
     << "    return;                                                      \n"
     << "  }                                                              \n"
     << "}                                                                \n";
}

void emit_attributeBLAS(TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  os << "void attribute_" << name << "(BlasInfo blas, llvm::Function *F) {\n"
     << "  auto name = F->getName();\n"
     << "  llvm::Optional<BlasInfo> blasName = extractBLAS(name);\n"
     << "  if (!blasName.hasValue()) return;\n"
     << "  F->addFnAttr(llvm::Attribute::ArgMemOnly);\n";
  
  auto argTypeMap = pattern.getArgTypeMap();
  DenseSet<size_t> mutableArgs = pattern.getMutableArgs();

  for (size_t i = 0; i < argTypeMap.size(); i++) {
    if (argTypeMap.lookup(i) == argType::vincData) {
      os << "  const bool julia_decl = !F->getArg(" << i << ")->getType()->isPointerTy();\n";
      break;
    }
    if (i+1 == argTypeMap.size()) {
      llvm::errs() << "Tablegen bug: BLAS fnc without vector!\n";
      llvm_unreachable("Tablegen bug: BLAS fnc without vector!");
    }
  }
  for (size_t i = 0; i < argTypeMap.size(); i++) {
    if (argTypeMap.lookup(i) == argType::len) {
      os << "  const bool byRef = !F->getArg(" << i << ")->getType()->isIntegerTy();\n";
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

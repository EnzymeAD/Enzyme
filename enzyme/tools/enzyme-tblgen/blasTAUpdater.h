void emit_BLASTypes(raw_ostream &os) {

  os << "size_t firstIntPos = getFirstLenOrIncPosition(blas);\n";

  os << "#if LLVM_VERSION_MAJOR >= 10\n"
     << "  const bool byRef = !call.getArgOperand(firstIntPos\n"
     << ")->getType()->isIntegerTy() && blas.prefix == \"\";\n"
     << "#else\n"
     << "  const bool byRef = !call.getOperand(firstIntPos\n"
     << ")->getType()->isIntegerTy() && blas.prefix == \"\";\n"
     << "#endif\n";

  os << "TypeTree ttFloat;\n"
     << "if (blas.floatType == \"s\") {\n"
     << "  ttFloat.insert({-1},Type::getFloatTy(call.getContext()));\n"
     << "} else {\n"
     << "  ttFloat.insert({-1},Type::getDoubleTy(call.getContext()));\n"
     << "}\n";

  os << "TypeTree ttInt;\n"
     << "if (byRef) {\n"
     << "  ttInt.insert({-1},BaseType::Pointer);\n"
     << "  ttInt.insert({-1,0},BaseType::Integer);\n"
     << "  ttInt.insert({-1,1},BaseType::Integer);\n"
     << "  ttInt.insert({-1,2},BaseType::Integer);\n"
     << "  ttInt.insert({-1,3},BaseType::Integer);\n"
     << "  if (blas.suffix == \"_64_\" || blas.suffix == \"64_\") {\n"
     << "    ttInt.insert({-1,4},BaseType::Integer);\n"
     << "    ttInt.insert({-1,5},BaseType::Integer);\n"
     << "    ttInt.insert({-1,6},BaseType::Integer);\n"
     << "    ttInt.insert({-1,7},BaseType::Integer);\n"
     << "  }\n"
     << "} else {\n"
     << "  ttInt.insert({-1},BaseType::Integer);\n"
     << "}\n";

  os << "TypeTree ttPtr;\n"
     << "ttPtr.insert({-1},BaseType::Pointer);\n";
}
     
void emit_BLASTA(TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();


  os << "if (blas.function == \"" << name << "\") {\n";

  auto argTypeMap = pattern.getArgTypeMap();
  DenseSet<size_t> mutableArgs = pattern.getMutableArgs();

  // next calls to have byRef available
  for (size_t i = 0; i < argTypeMap.size(); i++) {
    if (argTypeMap.lookup(i) == argType::len) {
      os << "#if LLVM_VERSION_MAJOR >= 10\n"
         << "  const bool byRef = !call.getArgOperand(" << i
         << ")->getType()->isIntegerTy() && blas.prefix == \"\";\n"
         << "#else\n"
         << "  const bool byRef = !call.getOperand(" << i
         << ")->getType()->isIntegerTy() && blas.prefix == \"\";\n"
         << "#endif\n";
      break;
    }
    if (i + 1 == argTypeMap.size()) {
      llvm::errs() << "Tablegen bug: BLAS fnc without vector length!\n";
      llvm_unreachable("Tablegen bug: BLAS fnc without vector length!");
    }
  }
  // Now we can build TypeTrees
  for (size_t i = 0; i < argTypeMap.size(); i++) {
    auto currentType = argTypeMap.lookup(i);
    if (currentType == argType::len || currentType == argType::vincInc) {
      os << "  updateAnalysis(call.getArgOperand(" << i << "), ttInt, &call);\n";
    } else if (currentType == argType::vincData) {
    os << "  updateAnalysis(call.getArgOperand(" << i << "), ttPtr, &call);\n";
    } else if (currentType == argType::fp) {
      // TODO: check if byRef will pass this by Reference
    os << "  updateAnalysis(call.getArgOperand(" << i << "), ttFloat, &call);\n";
    }
  }
  os << "  Type *T = call.getType();\n"
     << "  if (T->isFloatingPointTy()) {\n"
     << "    updateAnalysis(&call, ttFloat, &call);\n"
     << "  }\n";

  os << "}\n";
}

void emitBlasTAUpdater(const RecordKeeper &RK, raw_ostream &os) {
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

  emit_BLASTypes(os);
  for (auto newPattern : newBlasPatterns) {
    emit_BLASTA(newPattern, os);
  }
}

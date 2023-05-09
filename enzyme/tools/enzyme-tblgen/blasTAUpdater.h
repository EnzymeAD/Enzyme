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
     << "llvm::Type *floatType; \n"
     << "if (blas.floatType == \"s\") {\n"
     << "  floatType = Type::getFloatTy(call.getContext());\n"
     << "} else {\n"
     << "  floatType = Type::getDoubleTy(call.getContext());\n"
     << "}\n"
     << "ttFloat.insert({-1},floatType);\n";

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
  if (name == "dot" || name == "asum" || name == "nrm2") {
    os << "  assert(call.getType()->isFloatingPointTy());\n"
       << "  updateAnalysis(&call, ttFloat, &call);\n";
  }

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

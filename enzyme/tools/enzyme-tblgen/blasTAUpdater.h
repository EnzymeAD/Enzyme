#include "datastructures.h"

void emit_BLASTypes(raw_ostream &os) {
  os << "const bool byRef = blas.prefix == \"\";\n";
  os << "const int offset = (byRef ? 0 : 1);\n";

  os << "TypeTree ttFloat;\n"
     << "llvm::Type *floatType; \n"
     << "if (blas.floatType == \"s\") {\n"
     << "  floatType = Type::getFloatTy(call.getContext());\n"
     << "} else {\n"
     << "  floatType = Type::getDoubleTy(call.getContext());\n"
     << "}\n"
     << "if (byRef) {\n"
     << "  ttFloat.insert({-1},BaseType::Pointer);\n"
     << "  ttFloat.insert({-1,0},floatType);\n"
     << "} else { \n"
     << "  ttFloat.insert({-1},floatType);\n"
     << "}\n";

  os << "TypeTree ttFloatRet;\n"
     << "ttFloatRet.insert({-1},floatType);\n";

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
     << "ttPtr.insert({-1},BaseType::Pointer);\n"
     << "ttPtr.insert({-1,0},floatType);\n";
}

void emit_BLASTA(TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  bool lv23 = pattern.isBLASLevel2or3();

  os << "if (blas.function == \"" << name << "\") {\n";

  auto argTypeMap = pattern.getArgTypeMap();
  DenseSet<size_t> mutableArgs = pattern.getMutableArgs();

  // Now we can build TypeTrees
  for (size_t j = 0; j < argTypeMap.size(); j++) {
    auto currentType = argTypeMap.lookup(j);
    // sorry. will fix later. effectively, skip arg 0 for for lv23,
    // because we have the cblas layout in the .td declaration
    size_t i = (lv23 ? j - 1 : j);
    if (currentType == ArgType::len || currentType == ArgType::vincInc) {
      os << "  updateAnalysis(call.getArgOperand(" << i
         << (lv23 ? " + offset" : "") << "), ttInt, &call);\n";
    } else if (currentType == ArgType::vincData) {
      assert(argTypeMap.lookup(j + 1) == ArgType::vincInc);
      if (!lv23)
        os << "  if (auto n = dyn_cast<ConstantInt>(call.getArgOperand(0"
           << (lv23 ? " + offset" : "") << "))) {\n"
           << "    if (auto inc = dyn_cast<ConstantInt>(call.getArgOperand("
           << i << (lv23 ? " + offset" : "") << "))) {\n"
           << "      assert(!inc->isNegative());\n"
           << "      TypeTree ttData = ttPtr;\n"
           << "      for (size_t i = 1; i < n->getZExtValue(); i++)\n"
           << "          ttData.insert({-1, int(i * inc->getZExtValue())}, "
              "floatType);\n"
           << "      updateAnalysis(call.getArgOperand(" << i
           << (lv23 ? " + offset" : "") << "), ttData, &call);\n"
           << "    } else {\n"
           << "      updateAnalysis(call.getArgOperand(" << i
           << (lv23 ? " + offset" : "") << "), ttPtr, &call);\n"
           << "    }\n"
           << "  } else {\n"
           << "    updateAnalysis(call.getArgOperand(" << i
           << (lv23 ? " + offset" : "") << "), ttPtr, &call);\n"
           << "  }\n";
      else
        os << "    updateAnalysis(call.getArgOperand(" << i
           << (lv23 ? " + offset" : "") << "), ttPtr, &call);\n";
    } else if (currentType == ArgType::fp) {
      os << "  updateAnalysis(call.getArgOperand(" << i
         << (lv23 ? " + offset" : "") << "), ttFloat, &call);\n";
    }
  }
  if (name == "dot" || name == "asum" || name == "nrm2") {
    os << "  assert(call.getType()->isFloatingPointTy());\n"
       << "  updateAnalysis(&call, ttFloatRet, &call);\n";
  }

  os << "}\n";
}

void emitBlasTAUpdater(const RecordKeeper &RK, raw_ostream &os) {
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

  emit_BLASTypes(os);
  for (auto &newPattern : newBlasPatterns) {
    emit_BLASTA(newPattern, os);
  }
}

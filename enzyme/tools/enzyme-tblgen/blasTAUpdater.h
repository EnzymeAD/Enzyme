#ifndef ENZYME_TBLGEN_BLAS_TA_UPDATER_H
#define ENZYME_TBLGEN_BLAS_TA_UPDATER_H

#include "datastructures.h"

void emit_BLASTypes(raw_ostream &os) {
  os << "const bool byRef = blas.prefix == \"\" || blas.prefix == "
        "\"cublas_\";\n";
  os << "const bool byRefFloat = byRef || blas.prefix == "
        "\"cublas\";\n";
  os << "(void)byRefFloat;\n";
  os << "const bool cblas = blas.prefix == \"cblas_\";\n";
  os << "const bool cublas = blas.prefix == \"cublas_\" || blas.prefix == "
        "\"cublas\";\n";
  os << "const bool cublasv2 = blas.prefix == "
        "\"cublas\" && StringRef(blas.suffix).contains(\"v2\");\n";

  os << "TypeTree ttFloat;\n"
     << "llvm::Type *floatType = blas.fpType(call.getContext()); \n"
     << "if (byRefFloat) {\n"
     << "  ttFloat.insert({-1},BaseType::Pointer);\n"
     << "  ttFloat.insert({-1,0},floatType);\n"
     << "} else { \n"
     << "  ttFloat.insert({-1},floatType);\n"
     << "}\n";

  os << "TypeTree ttFloatRet;\n"
     << "ttFloatRet.insert({-1},floatType);\n"
     << "TypeTree ttCuBlasRet;\n"
     << "ttCuBlasRet.insert({-1},BaseType::Integer);\n";

  os << "TypeTree ttPtrInt;\n"
     << "ttPtrInt.insert({-1},BaseType::Pointer);\n"
     << "ttPtrInt.insert({-1, -1},BaseType::Integer);\n";

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

  os << "TypeTree ttChar;\n"
     << "if (byRef) {\n"
     << "  ttChar.insert({-1},BaseType::Pointer);\n"
     << "  ttChar.insert({-1,0},BaseType::Integer);\n"
     << "} else {\n"
     << "  ttChar.insert({-1},BaseType::Integer);\n"
     << "}\n";

  os << "TypeTree ttCuHandle;\n"
     << "ttCuHandle.insert({-1},BaseType::Pointer);\n";

  os << "TypeTree ttPtr;\n"
     << "ttPtr.insert({-1},BaseType::Pointer);\n"
     << "ttPtr.insert({-1,0},floatType);\n";
}

// cblas lv23 => layout
// cublas => always handle
void emit_BLASTA(TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();
  bool lv23 = pattern.isBLASLevel2or3();

  os << "if (blas.function == \"" << name << "\") {\n";

  os << "  const int offset = (";
  if (lv23) {
    os << "(cblas || cublas)";
  } else {
    os << "cublas";
  }
  os << " ? 1 : 0);\n";

  auto argTypeMap = pattern.getArgTypeMap();
  DenseSet<size_t> mutableArgs = pattern.getMutableArgs();

  // Now we can build TypeTrees
  for (size_t j = 0; j < argTypeMap.size(); j++) {
    auto currentType = argTypeMap.lookup(j);
    // sorry. will fix later. effectively, skip arg 0 for for lv23,
    // because we have the cblas layout in the .td declaration
    size_t i = (lv23 ? j - 1 : j);
    if (pattern.getArgNames().size() <= j) {
      PrintFatalError(pattern.getLoc(),
                      Twine("Too few argnames for pattern '") + name +
                          "' found " +
                          std::to_string(pattern.getArgNames().size()) +
                          " expected " + std::to_string(argTypeMap.size()));
    }
    os << "  // " << currentType << " " << pattern.getArgNames()[j] << "\n";
    switch (currentType) {
    case ArgType::info:
      os << "  updateAnalysis(call.getArgOperand(" << i
         << " + offset), ttPtrInt, &call);\n";
      break;
    case ArgType::len:
    case ArgType::vincInc:
    case ArgType::mldLD:
      os << "  updateAnalysis(call.getArgOperand(" << i
         << " + offset), ttInt, &call);\n";
      break;
    case ArgType::vincData:
      assert(argTypeMap.lookup(j + 1) == ArgType::vincInc);
      // TODO, we need a get length arg number from vector since always assuming
      // it is arg 0 is wrong.
      if (!lv23)
        os << "  if (auto n = dyn_cast<ConstantInt>(call.getArgOperand(0 + "
              "offset))) {\n"
           << "    if (auto inc = dyn_cast<ConstantInt>(call.getArgOperand("
           << i << " + offset))) {\n"
           << "      assert(!inc->isNegative());\n"
           << "      TypeTree ttData = ttPtr;\n"
           << "      for (size_t i = 1; i < n->getZExtValue(); i++)\n"
           << "          ttData.insert({-1, int(i * inc->getZExtValue())}, "
              "floatType);\n"
           << "      updateAnalysis(call.getArgOperand(" << i
           << " + offset), ttData, &call);\n"
           << "    } else {\n"
           << "      updateAnalysis(call.getArgOperand(" << i
           << " + offset), ttPtr, &call);\n"
           << "    }\n"
           << "  } else {\n"
           << "    updateAnalysis(call.getArgOperand(" << i
           << " + offset), ttPtr, &call);\n"
           << "  }\n";
      else
        os << "  updateAnalysis(call.getArgOperand(" << i
           << (lv23 ? " + offset" : "") << "), ttPtr, &call);\n";
      break;
    case ArgType::mldData:
      os << "  updateAnalysis(call.getArgOperand(" << i
         << " + offset), ttPtr, &call);\n";
      break;
    case ArgType::fp:
      os << "  updateAnalysis(call.getArgOperand(" << i
         << " + offset), ttFloat, &call);\n";
      break;
    case ArgType::ap:
      // TODO
      break;
    case ArgType::cblas_layout:
      // TODO
      break;
    case ArgType::uplo:
    case ArgType::trans:
      os << "  updateAnalysis(call.getArgOperand(" << i
         << " + offset), ttChar, &call);\n";
      break;
    case ArgType::diag:
    case ArgType::side:
      // TODO
      break;
    }
  }
  if (has_active_return(name)) {
    // under cublas, these functions have an extra return ptr argument
    size_t ptrRetArg = argTypeMap.size();
    os << "  if (cublasv2) {\n"
       << "    updateAnalysis(call.getArgOperand(" << ptrRetArg
       << " + offset), ttPtr, &call);\n"
       << "    updateAnalysis(&call, ttCuBlasRet, &call);\n"
       << "  } else {\n"
       << "    assert(call.getType()->isFloatingPointTy());\n"
       << "    updateAnalysis(&call, ttFloatRet, &call);\n"
       << "  }\n";
  } else {
    os << "  if (cublas) {\n"
       << "    updateAnalysis(&call, ttCuBlasRet, &call);\n"
       << "  }\n";
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

#endif // ENZYME_TBLGEN_BLAS_TA_UPDATER_H

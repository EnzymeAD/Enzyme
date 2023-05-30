void emit_BLASDiffUse(TGPattern &pattern, raw_ostream &os) {
  auto name = pattern.getName();

  os << "if (blas.function == \"" << name << "\") {\n";

  auto argTypeMap = pattern.getArgTypeMap();
  auto argUsers = pattern.getArgUsers();

  for (size_t i = 0; i < argTypeMap.size(); i++) {
    os << "  if (val == CI->getArgOperand(" << i << ")) {\n";
    auto users = argUsers.lookup(i);
    if (users.size() == 0) {
      os << "    return false;\n"
         << "  }\n";
      continue;
    } else {
      for (auto a : users) {
        // The following shows that I probably should change the tblgen
        // logic and the Blas.td declaration
        if (a == i)
          continue;
        os << "    if (!gutils->isConstantValue(CI->getOperand(" << a
           << "))) return true;\n";
      }
      os << "  }\n";
    }
  }

  os << "}\n";
}

void emitBlasDiffUse(const RecordKeeper &RK, raw_ostream &os) {
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

  for (auto newPattern : newBlasPatterns) {
    emit_BLASDiffUse(newPattern, os);
  }
}

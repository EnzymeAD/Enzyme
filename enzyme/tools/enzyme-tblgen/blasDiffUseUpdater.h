void emit_BLASDiffUse(TGPattern &pattern, raw_ostream &os) {
  auto argTypeMap = pattern.getArgTypeMap();
  auto argUsers = pattern.getArgUsers();
  bool lv23 = pattern.isBLASLevel2or3();

  auto name = pattern.getName();

  os << "if (blas.function == \"" << name << "\") {\n";
  os << "const bool byRef = blas.prefix == \"\";\n";
  if (lv23) {
    os << "const int offset = (byRef ? 0 : 1);\n";

    auto layout_users = argUsers.lookup(0);
    os << "if (!byRef && val == CI->getArgOperand(0)) {\n";
    for (auto user : layout_users) {
      os << "  if (!gutils->isConstantValue(CI->getOperand(" << user
         << "))) return true;\n";
    }
    os << "}\n";
  }

  for (size_t argPos = (lv23 ? 1 : 0); argPos < argTypeMap.size(); argPos++) {
    auto users = argUsers.lookup(argPos);
    size_t i = (lv23 ? argPos - 1 : argPos);
    os << "  if (val == CI->getArgOperand(" << i << (lv23 ? " + offset" : "")
       << ")) {\n";
    for (auto a : users) {
      // The following shows that I probably should change the tblgen
      // logic and the Blas.td declaration
      if (a == i) // a == i? argpos ?
        continue;
      os << "    if (!gutils->isConstantValue(CI->getOperand(" << a
         << (lv23 ? " + offset" : "") << "))) return true;\n";
    }
    os << "  }\n";
  }

  os << "  return false;\n";
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

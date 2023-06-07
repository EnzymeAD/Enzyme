void emit_BLASDiffUse(TGPattern &pattern, raw_ostream &os) {
  auto typeMap = pattern.getArgTypeMap();
  auto argUsers = pattern.getArgUsers();
  bool lv23 = pattern.isBLASLevel2or3();
  auto nameVec = pattern.getArgNames();
  auto actArgs = pattern.getActiveArgs();

  auto name = pattern.getName();
  if (name == "lascl")
    return;

  os << "if (blas.function == \"" << name << "\") {\n";
  os << "  const bool byRef = blas.prefix == \"\";\n";

  if (lv23) {
    os << "  const int offset = (byRef ? 0 : 1);\n";

    auto layout_users = argUsers.lookup(0);
    os << "  if (!byRef && val == CI->getArgOperand(0)) {\n";
    for (auto user : layout_users) {
      os << "    if (!gutils->isConstantValue(CI->getOperand(" << user
         << "))) return true;\n";
    }
    os << "  }\n";
  }

  // initialize arg_ arguments
  for (size_t argPos = (lv23 ? 1 : 0); argPos < typeMap.size(); argPos++) {
    assert(argPos < nameVec.size());
    auto name = nameVec[argPos];
    size_t i = (lv23 ? argPos - 1 : argPos);
    os << "  auto pos_" << name << " = " << i << (lv23 ? " + offset " : "")
       << ";\n";
    os << "  auto arg_" << name << " = CI->getArgOperand(pos_" << name
       << ");\n";
    os << "  const bool overwritten_" << name
       << " = (cacheMode ? (overwritten_args_ptr ? (*overwritten_args_ptr)[pos_" << name << "] : true ) : false);\n\n";
  }

  // initialize active_ arguments
  for (auto arg : actArgs) {
    auto name = nameVec[arg];
    os << "  bool active_" << name << " = !gutils->isConstantValue(arg_" << name
       << ");\n";
  }

  emit_scalar_caching(pattern, os);
  // we currently cache all vecs before we cache all matrices
  // once fixed we can merge this calls
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != vincData)
      continue;
    assert(typeMap.lookup(i + 1) == vincInc);
    emit_mat_vec_caching(pattern, i, os);
  }
  for (size_t i = 0; i < nameVec.size(); i++) {
    auto ty = typeMap.lookup(i);
    if (ty != mldData)
      continue;
    assert(typeMap.lookup(i + 1) == mldLD);
    emit_mat_vec_caching(pattern, i, os);
  }

  for (size_t argPos = (lv23 ? 1 : 0); argPos < typeMap.size(); argPos++) {
    auto users = argUsers.lookup(argPos);
    auto name = nameVec[argPos];
    size_t i = (lv23 ? argPos - 1 : argPos);
    os << "  if (val == arg_" << name << " && !cache_" << name << ") {\n";
    for (auto a : users) {
      auto name = nameVec[a];
      // The following shows that I probably should change the tblgen
      // logic and the Blas.td declaration
      if (a == i) // a == i? argpos ?
        continue;
      os << "    if (active_" << name << ") return true;\n";
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

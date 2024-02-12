#include "blas-tblgen.h"
#include "caching.h"
#include "datastructures.h"
#include "enzyme-tblgen.h"
#include "llvm/Support/raw_ostream.h"

void emit_BLASDiffUse(TGPattern &pattern, llvm::raw_ostream &os) {
  auto typeMap = pattern.getArgTypeMap();
  auto argUsers = pattern.getArgUsers();
  bool lv23 = pattern.isBLASLevel2or3();
  auto nameVec = pattern.getArgNames();
  auto actArgs = pattern.getActiveArgs();

  auto name = pattern.getName();
  if (name == "lascl")
    return;

  os << "if (blas.function == \"" << name << "\") {\n";

  os << "  const bool byRef = blas.prefix == \"\" || blas.prefix == "
        "\"cublas_\";\n";
  if (lv23)
    os << "  const bool cblas = blas.prefix == \"cblas_\";\n";
  os << "  const bool cublas = blas.prefix == \"cublas_\" || blas.prefix == "
        "\"cublas\";\n";
  // lv 2 or 3 functions have an extra arg under the cblas_ abi
  os << "  const int offset = (";
  if (lv23) {
    os << "(cblas || cublas)";
  } else {
    os << "cublas";
  }
  os << " ? 1 : 0);\n";

  os << " if (cublas && !shadow && val == CI->getArgOperand(0)) return true;\n";

  if (lv23) {
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
    os << "  auto pos_" << name << " = " << i << " + offset "
       << ";\n";
    os << "  auto arg_" << name << " = CI->getArgOperand(pos_" << name
       << ");\n";
    os << "  const bool overwritten_" << name
       << " = (cacheMode ? (overwritten_args_ptr ? (*overwritten_args_ptr)[pos_"
       << name << "] : true ) : false);\n\n";
  }

  // initialize active_ arguments
  for (auto arg : actArgs) {
    auto name = nameVec[arg];
    os << "  bool active_" << name << " = !gutils->isConstantValue(arg_" << name
       << ");\n";
    os << "  if (!shadow && EnzymeRuntimeActivityCheck && active_" << name
       << ") return true;\n";
  }

  emit_need_cache_info(pattern, os);

  for (size_t argPos = (lv23 ? 1 : 0); argPos < typeMap.size(); argPos++) {
    auto users = argUsers.lookup(argPos);
    auto argname = nameVec[argPos];

    os << "  if (val == arg_" << argname << ") {\n";

    // We need the shadow of the value we're updating
    if (typeMap[argPos] == ArgType::fp) {
      os << "    if (shadow && byRef && active_" << argname
         << ") return true;\n";
    } else if (typeMap[argPos] == ArgType::vincData ||
               typeMap[argPos] == ArgType::mldData) {
      for (auto derivOp : pattern.getRules()) {
        if (hasAdjoint(derivOp.getRuleDag(), argname)) {
          os << "    if (shadow && active_"
             << nameVec[derivOp.getHandledArgIdx()] << ") return true;\n";
        } else {
          bool isNoop = false;
          if (DagInit *resultRoot = dyn_cast<DagInit>(derivOp.getRuleDag())) {
            auto opName = resultRoot->getOperator()->getAsString();
            auto Def = cast<DefInit>(resultRoot->getOperator())->getDef();
            if (Def->getName() == "noop" || Def->getName() == "inactive") {
              isNoop = true;
            }
          }
          if (DefInit *DefArg = dyn_cast<DefInit>(derivOp.getRuleDag())) {
            auto Def = DefArg->getDef();
            if (Def->getName() == "noop" || Def->getName() == "inactive") {
              isNoop = true;
            }
          }
          // updates to a vector/matrix must definitionally use the shadow of
          // the input, unless a noop-update
          if (!isNoop) {
            if (derivOp.getHandledArgIdx() == argPos) {
              llvm::errs() << " fnname: " << name << " argPos: " << argPos
                           << " argname: " << argname
                           << " rule: " << *derivOp.getRuleDag() << "\n";
            }
            assert(derivOp.getHandledArgIdx() != argPos);
          }
        }
      }
    }

    os << "    if (!shadow && need_" << argname
       << " && ((cacheMode && overwritten_args_ptr && (mode == "
          "DerivativeMode::ReverseModeGradient)) ? !cache_"
       << argname << " : true ))\n"
       << "      return true;\n";
    os << "  }\n";
  }

  // If any of the rule uses DiffeRet, the primary function has a ret val
  // and we should emit the code for handling it.
  bool hasDiffeRetVal = false;
  for (auto derivOp : pattern.getRules()) {
    hasDiffeRetVal |= hasDiffeRet(derivOp.getRuleDag());
  }

  if (hasDiffeRetVal) {
    size_t ptrRetArg = typeMap.size();
    auto retarg =
        "CI->getArgOperand(" + std::to_string(ptrRetArg) + " + offset)";
    os << "  if (cublas) {\n";
    os << "    if (!gutils->isConstantValue(" << retarg << "))\n";
    os << "      if ((shadow || EnzymeRuntimeActivityCheck) && val == "
       << retarg << ") return true;\n";
    os << "    if (mode != DerivativeMode::ReverseModeGradient && !shadow && "
          "val == "
       << retarg << ") return true;\n";
    os << "  }\n";
  }

  os << "  return false;\n";
  os << "}\n";
}

void emitBlasDiffUse(const RecordKeeper &RK, llvm::raw_ostream &os) {
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

  os << "switch(user->getOpcode()) {\n";
  emitDiffUse(RK, os, InstDerivatives);
  os << "  default: break;\n";
  os << "}\n";
  os << "if (auto BO = dyn_cast<BinaryOperator>(user)) {\n";
  os << "  switch(BO->getOpcode()) {\n";
  emitDiffUse(RK, os, BinopDerivatives);
  os << "    default: break;\n";
  os << "  }\n";
  os << "}\n";
  os << "if (auto CI = dyn_cast<CallInst>(user)) {\n";
  os << "  switch(ID) {\n";
  os << "    default: break;\n";
  emitDiffUse(RK, os, IntrDerivatives);
  os << "  }\n";

  os << "  auto funcName = getFuncNameFromCall(CI);\n";

  emitDiffUse(RK, os, CallDerivatives);

  os << "    auto blasMetaData = extractBLAS(funcName);\n";
  os << "    if (blasMetaData)\n";
  os << "    {\n";
  os << "      auto Mode = gutils->mode;\n";
  os << "      const bool cacheMode = (Mode != DerivativeMode::ForwardMode);\n";
  os << "      const std::vector<bool> *overwritten_args_ptr = nullptr;\n";
  os << "      if (gutils->overwritten_args_map_ptr) {\n";
  os << "        auto found = \n";
  os << "          gutils->overwritten_args_map_ptr->find(const_cast<CallInst "
        "*>(CI));\n";
  os << "        assert(found != gutils->overwritten_args_map_ptr->end());\n";
  os << "        overwritten_args_ptr = &found->second;\n";
  os << "      }\n";
  os << "      BlasInfo blas = *blasMetaData;\n";
  for (auto &&newPattern : newBlasPatterns) {
    emit_BLASDiffUse(newPattern, os);
  }
  os << "    }\n";
  os << "}\n";
}

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "datastructures.h"

namespace llvm {
// llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &os, const argType
// &arg) {
raw_ostream &operator<<(raw_ostream &os, const argType &arg) {
  return (os << TyToString(arg));
}
} // namespace llvm
using namespace llvm;

const char *TyToString(argType ty) {
  switch (ty) {
  case argType::fp:
    return "fp";
  case argType::len:
    return "len";
  case argType::vincData:
    return "vincData";
  case argType::vincInc:
    return "vincInc";
  case argType::cblas_layout:
    return "layout";
  case argType::mldData:
    return "mldData";
  case argType::mldLD:
    return "mldLD";
  case argType::uplo:
    return "uplo";
  case argType::trans:
    return "trans";
  case argType::diag:
    return "diag";
  case argType::side:
    return "side";
  default:
    return "unknown";
  }
}

bool isArgUsed(const StringRef toFind, const DagInit *toSearch) {
  for (size_t i = 0; i < toSearch->getNumArgs(); i++) {
    if (DagInit *arg = dyn_cast<DagInit>(toSearch->getArg(i))) {
      // os << " Recursing. Magic!\n";
      if (isArgUsed(toFind, arg))
        return true;
    } else {
      auto name = toSearch->getArgNameStr(i);
      if (name == "") {
        // handle input<"x">, adj<"x">, transpose<"transa"> and similar
        // we look up the trans arg inside of transpose<"transX">,
        // because it's based on the same trans arg.
        // we ignore adj<"x"> because the shadow of x is not based on x
        auto opName = toSearch->getArg(i)->getAsString();
        auto Def = cast<DefInit>(toSearch->getArg(i))->getDef();
        if (opName == "transpose" || Def->isSubClassOf("transpose")) {
          auto transName = Def->getValueAsString("name");
          if (toFind == transName) {
            return true;
          }
        } else if (opName == "adj" || Def->isSubClassOf("adj")) {
          // shadow is unrelated, ignore it
        }
      } else {
        if (name == toFind) {
          return true;
        }
      }
    }
  }
  return false;
}

Rule::Rule(DagInit *dag, size_t activeArgIdx, StringMap<size_t> &patternArgs,
           DenseMap<size_t, argType> &patternTypes,
           DenseSet<size_t> &patternMutables) {

  rewriteRule = dag;
  activeArg = activeArgIdx;

  argNameToPos = StringMap<size_t>();
  argTypes = DenseMap<size_t, argType>();
  mutables = DenseSet<size_t>();

  // For each arg found in the dag:
  //        1) copy patternArgs to ruleArgs if arg shows up in this rule
  for (auto argName : patternArgs.keys()) {
    assert(patternArgs.count(argName) == 1);
    size_t argPos = patternArgs.lookup(argName);
    bool argUsedInRule = isArgUsed(argName, rewriteRule);
    if (argUsedInRule) {
      argNameToPos.insert(std::pair<std::string, size_t>(argName, argPos));
      //        2) look up and copy the corresponding argType
      assert(patternTypes.find(argPos) != patternTypes.end() &&
             "arg without corresponding type");
      argTypes.insert(*patternTypes.find(argPos));
    }
  }
  if (argTypes.lookup(0) == argType::cblas_layout) {
    BLASLevel2or3 = true;
  } else {
    BLASLevel2or3 = false;
  }

  for (auto ruleArgKey : argNameToPos.keys()) {
    //        3) look up and eventually copy mutable
    auto val = argNameToPos.lookup(ruleArgKey);
    if (patternMutables.find(val) != patternMutables.end()) {
      mutables.insert(*patternMutables.find(val));
    }
  }
  assert(argTypes.size() == argNameToPos.size());
}
bool Rule::isBLASLevel2or3() { return BLASLevel2or3; }
DagInit *Rule::getRuleDag() { return rewriteRule; }
size_t Rule::getHandledArgIdx() { return activeArg; }
StringMap<size_t> Rule::getArgNameMap() { return argNameToPos; }
DenseMap<size_t, argType> Rule::getArgTypeMap() { return argTypes; }
std::string Rule::to_string() {
  std::string res = "handling rule for argument ";
  res += std::to_string(activeArg);
  res += " with " + std::to_string(argTypes.size()) + " types: \n";
  for (size_t i = 0; i < argTypes.size(); i++) {
    auto ty = argTypes.lookup(i);
    res += (i > 0) ? ", " : "";
    res += std::to_string(i) + " " + TyToString(ty);
  }
  return res;
}

void fillActiveArgSet(const Record *pattern,
                      SmallVector<size_t, 4> &activeArgs) {

  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  size_t numTypes = 0;
  for (auto val : inputTypes) {
    if (val->getValueAsBit("active")) {
      activeArgs.push_back(numTypes);
    }
    numTypes += val->getValueAsInt("nelem");
  }
}

void fillMutableArgSet(const Record *pattern,
                       DenseSet<size_t> &mutables) {

  auto args = pattern->getValueAsDag("PatternToMatch");
  auto mutableArgs = pattern->getValueAsListOfStrings("mutable");
  // We must replace their names by their position
  for (auto mutableArg : mutableArgs) {
    size_t pos = 0;
    while (args->getArgNameStr(pos) != mutableArg) {
      pos++;
      if (pos == args->getNumArgs()) {
        PrintFatalError("mutable arg isn't an input Arg!");
      }
    }
    mutables.insert(pos);
  }

  assert(mutables.size() == mutableArgs.size());
}

void fillArgTypes(const Record *pattern, DenseMap<size_t, argType> &argTypes) {

  std::vector<Record *> inputTypes =
    pattern->getValueAsListOfDefs("inputTypes");
  size_t pos = 0;
  for (auto val : inputTypes) {
    if (val->getName() == "len") {
      argTypes.insert(std::make_pair(pos, argType::len));
    } else if (val->getName() == "fp") {
      argTypes.insert(std::make_pair(pos, argType::fp));
    } else if (val->getName() == "vinc") {
      argTypes.insert(std::make_pair(pos, argType::vincData));
      argTypes.insert(std::make_pair(pos + 1, argType::vincInc));
    } else if (val->getName() == "cblas_layout") {
      assert(pos == 0);
      argTypes.insert(std::make_pair(pos, argType::cblas_layout));
    } else if (val->getName() == "trans") {
      argTypes.insert(std::make_pair(pos, argType::trans));
    } else if (val->getName() == "diag") {
      argTypes.insert(std::make_pair(pos, argType::diag));
    } else if (val->getName() == "uplo") {
      argTypes.insert(std::make_pair(pos, argType::uplo));
    } else if (val->getName() == "side") {
      argTypes.insert(std::make_pair(pos, argType::side));
    } else if (val->getName() == "mld") {
      argTypes.insert(std::make_pair(pos, argType::mldData));
      argTypes.insert(std::make_pair(pos + 1, argType::mldLD));
    } else {
      llvm::errs() << "val->getName: " << val->getName() << "\n";
      PrintFatalError("Unknown type!");
    }
    pos += val->getValueAsInt("nelem");
  }
}

void fillArgs(const Record *r, SmallVectorImpl<std::string> &args,
              StringMap<size_t> &argNameToPos) {
  DagInit *argOps = r->getValueAsDag("PatternToMatch");
  size_t numArgs = argOps->getNumArgs();
  args.reserve(numArgs);
  for (size_t i = 0; i < numArgs; i++) {
    args.push_back(argOps->getArgNameStr(i).str());
    argNameToPos.insert(std::pair<std::string, size_t>(args[i], i));
  }
  assert(args.size() == numArgs);
  assert(argNameToPos.size() == numArgs);
}

void fillArgUserMap(SmallVectorImpl<Rule> &rules, ArrayRef<std::string> nameVec,
                    // ArrayRef<StringRef> nameVec,
                    ArrayRef<size_t> activeArgs,
                    DenseMap<size_t, DenseSet<size_t>> &argUsers) {

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto name = nameVec[i];
    DenseSet<size_t> set{};
    for (auto&& rule : llvm::enumerate(rules)) {
      auto nameMap = rule.value().getArgNameMap();
      if (nameMap.count(name) == 1) {
        size_t val = activeArgs[rule.index()];
        set.insert(val);
      }
    }
    auto newVal = std::make_pair<>(i, set);
    argUsers.insert(newVal);
  }
}

TGPattern::TGPattern(Record &r) {
  blasName = r.getNameInitAsString();

  args = llvm::SmallVector<std::string, 6>();
  argNameToPos = StringMap<size_t>{};
  fillArgs(&r, args, argNameToPos);

  argTypes = DenseMap<size_t, argType>();
  fillArgTypes(&r, argTypes);
  if (argTypes.lookup(0) == argType::cblas_layout) {
    BLASLevel2or3 = true;
  } else {
    BLASLevel2or3 = false;
  }

  posActArgs = SmallVector<size_t, 4>();
  fillActiveArgSet(&r, posActArgs);

  mutables = DenseSet<size_t>();
  fillMutableArgSet(&r, mutables);

  // Now create the rules for this pattern
  {
    rules = llvm::SmallVector<Rule, 3>{};
    ListInit *derivOps = r.getValueAsListInit("ArgDerivatives");
    for (auto&& derivOp : llvm::enumerate(*derivOps)) {
      DagInit *derivRule = cast<DagInit>(derivOp.value());
      size_t actIdx = posActArgs[derivOp.index()];
      rules.push_back(
          Rule(derivRule, actIdx, argNameToPos, argTypes, mutables));
    }
  }

  argUsers = DenseMap<size_t, DenseSet<size_t>>();
  ArrayRef<std::string> nameVec =
      ArrayRef<std::string>(args.begin(), args.end());
  fillArgUserMap(rules, nameVec, posActArgs, argUsers);
  // fillArgUserMap(rules, ArrayRef(args), posActArgs, argUsers);
}
SmallVector<size_t, 2> TGPattern::getRelatedLengthArgs(size_t arg) {
  if (!BLASLevel2or3) {
    assert(argTypes.lookup(arg) == argType::vincData);
    assert(argTypes.lookup(0) == argType::len);
    return {0};
  }
  assert(argTypes.lookup(arg) == argType::vincData ||
         argTypes.lookup(arg) == argType::mldData);
  // This is terribly wrong because it will burn
  // once someone sets TRANS to T
  if (blasName == "gemm") {
    if (arg == 7)
      return {3, 5};
    if (arg == 9)
      return {5, 4};
    if (arg == 12)
      return {3, 4};
    assert(false);
  }
  if (blasName == "gemv") {
    if (arg == 5)
      return {2, 3};
    if (arg == 7)
      return {3};
    if (arg == 10)
      return {2};
    assert(false);
  }
  if (blasName == "ger") {
    if (arg == 4)
      return {1};
    if (arg == 6)
      return {2};
    if (arg == 8)
      return {1, 2};
    assert(false);
  }
  llvm::errs() << "failed for: " << blasName << "\n";
  assert(false);
}
bool TGPattern::isBLASLevel2or3() { return BLASLevel2or3; }
DenseMap<size_t, DenseSet<size_t>> TGPattern::getArgUsers() { return argUsers; }
std::string TGPattern::getName() { return blasName; }
SmallVector<std::string, 6> TGPattern::getArgNames() { return args; }
StringMap<size_t> TGPattern::getArgNameMap() { return argNameToPos; }
DenseMap<size_t, argType> TGPattern::getArgTypeMap() { return argTypes; }
DenseSet<size_t> TGPattern::getMutableArgs() { return mutables; }
SmallVector<size_t, 4> TGPattern::getActiveArgs() { return posActArgs; }
SmallVector<Rule, 3> TGPattern::getRules() { return rules; }

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
raw_ostream &operator<<(raw_ostream &os, ArgType arg) {
  return (os << TyToString(arg));
}
raw_ostream &operator<<(raw_fd_ostream &os, ArgType arg) {
  return (os << TyToString(arg));
}
} // namespace llvm

using namespace llvm;

const char *TyToString(ArgType ty) {
  switch (ty) {
  case ArgType::fp:
    return "fp";
  case ArgType::ap:
    return "ap";
  case ArgType::len:
    return "len";
  case ArgType::vincData:
    return "vincData";
  case ArgType::vincInc:
    return "vincInc";
  case ArgType::cblas_layout:
    return "layout";
  case ArgType::mldData:
    return "mldData";
  case ArgType::mldLD:
    return "mldLD";
  case ArgType::uplo:
    return "uplo";
  case ArgType::trans:
    return "trans";
  case ArgType::diag:
    return "diag";
  case ArgType::side:
    return "side";
  default:
    return "unknown";
  }
}

bool isVecLikeArg(ArgType ty) {
  if (ty == ArgType::vincData || ty == ArgType::mldData || ty == ArgType::ap)
    return true;
  return false;
}

bool isArgUsed(StringRef toFind, const DagInit *toSearch) {
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

Rule::Rule(DagInit *dag, size_t activeArgIdx,
           const StringMap<size_t> &patternArgs,
           const DenseMap<size_t, ArgType> &patternTypes,
           const DenseSet<size_t> &patternMutables)
    : rewriteRule(dag), activeArg(activeArgIdx) {
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
  if (argTypes.lookup(0) == ArgType::cblas_layout) {
    BLASLevel2or3 = true;
  } else {
    BLASLevel2or3 = false;
  }

  for (auto ruleArgKey : argNameToPos.keys()) {
    // 3) look up and eventually copy mutable
    auto val = argNameToPos.lookup(ruleArgKey);
    if (patternMutables.find(val) != patternMutables.end()) {
      mutables.insert(*patternMutables.find(val));
    }
  }
  assert(argTypes.size() == argNameToPos.size());
}

bool Rule::isBLASLevel2or3() const { return BLASLevel2or3; }

DagInit *Rule::getRuleDag() { return rewriteRule; }

size_t Rule::getHandledArgIdx() const { return activeArg; }

const StringMap<size_t> &Rule::getArgNameMap() const { return argNameToPos; }

const DenseMap<size_t, ArgType> &Rule::getArgTypeMap() const {
  return argTypes;
}

std::string Rule::to_string() const {
  std::string res = ("handling rule for argument " + Twine(activeArg) +
                     " with " + Twine(argTypes.size()) + " types: \n")
                        .str();
  for (size_t i = 0; i < argTypes.size(); i++) {
    auto ty = argTypes.lookup(i);
    res += (Twine((i > 0) ? ", " : "") + Twine(i) + " " + TyToString(ty)).str();
  }
  return res;
}

void fillActiveArgSet(const Record *pattern,
                      SmallVectorImpl<size_t> &activeArgs) {

  auto inputTypes = pattern->getValueAsListOfDefs("inputTypes");
  size_t numTypes = 0;
  for (auto val : inputTypes) {
    if (val->getValueAsBit("active")) {
      activeArgs.push_back(numTypes);
    }
    numTypes += val->getValueAsInt("nelem");
  }
}

void fillMutableArgSet(const Record *pattern, DenseSet<size_t> &mutables) {

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

void fillArgTypes(const Record *pattern, DenseMap<size_t, ArgType> &argTypes) {

  auto inputTypes = pattern->getValueAsListOfDefs("inputTypes");
  size_t pos = 0;
  for (auto val : inputTypes) {
    if (val->isSubClassOf("vinc")) {
      argTypes.insert(std::make_pair(pos, ArgType::vincData));
      argTypes.insert(std::make_pair(pos + 1, ArgType::vincInc));
    } else if (val->isSubClassOf("mld")) {
      argTypes.insert(std::make_pair(pos, ArgType::mldData));
      argTypes.insert(std::make_pair(pos + 1, ArgType::mldLD));
    } else if (val->isSubClassOf("ap")) {
      argTypes.insert(std::make_pair(pos, ArgType::ap));
    } else {
      // TODO: fix assertion
      // assert(isa<DefInit>(val));
      auto name = val->getName();
      if (name == "len") {
        argTypes.insert(std::make_pair(pos, ArgType::len));
      } else if (name == "fp") {
        argTypes.insert(std::make_pair(pos, ArgType::fp));
      } else if (name == "cblas_layout") {
        assert(pos == 0);
        argTypes.insert(std::make_pair(pos, ArgType::cblas_layout));
      } else if (name == "trans") {
        argTypes.insert(std::make_pair(pos, ArgType::trans));
      } else if (name == "diag") {
        argTypes.insert(std::make_pair(pos, ArgType::diag));
      } else if (name == "uplo") {
        argTypes.insert(std::make_pair(pos, ArgType::uplo));
      } else if (name == "side") {
        argTypes.insert(std::make_pair(pos, ArgType::side));
      } else {
        errs() << "val->getName: " << name << "\n";
        PrintFatalError("Unknown type!");
      }
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

void fillRelatedLenghts(
    const Record *pattern, const StringMap<size_t> &argNameToPos,
    const DenseMap<size_t, ArgType> &argTypes,
    DenseMap<size_t, SmallVector<size_t, 3>> &relatedLengths) {
  auto inputTypes = pattern->getValueAsListOfDefs("inputTypes");
  size_t pos = 0;
  for (auto val : inputTypes) {
    if (!val->isSubClassOf("vinc") && !val->isSubClassOf("mld") &&
        !val->isSubClassOf("ap")) {
      pos += val->getValueAsInt("nelem");
      continue;
    }

    auto args = val->getValueAsListOfStrings("args");
    auto argsSize = args.size();
    SmallVector<size_t, 3> lengths;
    for (auto arg : args) {
      lengths.push_back(argNameToPos.lookup(arg));
    }

    if (val->isSubClassOf("vinc")) {
      assert(argsSize == 1 || argsSize == 3);
      assert(argTypes.lookup(pos) == ArgType::vincData);
      assert(argTypes.lookup(pos + 1) == ArgType::vincInc);
      if (argsSize == 1) {
        assert(argTypes.lookup(lengths[0]) == ArgType::len);
      } else {
        assert(argTypes.lookup(lengths[0]) == ArgType::trans);
        assert(argTypes.lookup(lengths[1]) == ArgType::len);
        assert(argTypes.lookup(lengths[2]) == ArgType::len);
      }
      relatedLengths.insert(std::make_pair(pos, lengths));
    } else if (val->isSubClassOf("ap")) {
      assert(argsSize == 1);
      assert(argTypes.lookup(lengths[0]) == ArgType::len);
      relatedLengths.insert(std::make_pair(pos, lengths));
    } else if (val->isSubClassOf("mld")) {
      assert(argsSize == 2 || argsSize == 3);
      assert(argTypes.lookup(pos) == ArgType::mldData);
      assert(argTypes.lookup(pos + 1) == ArgType::mldLD);
      if (argsSize == 2) {
        assert(argTypes.lookup(lengths[0]) == ArgType::len);
        assert(argTypes.lookup(lengths[1]) == ArgType::len);
      } else {
        assert(argTypes.lookup(lengths[0]) == ArgType::trans);
        assert(argTypes.lookup(lengths[1]) == ArgType::len);
        assert(argTypes.lookup(lengths[2]) == ArgType::len);
      }
      relatedLengths.insert(std::make_pair(pos, lengths));
    }
    pos += val->getValueAsInt("nelem");
  }
}

void fillArgUserMap(ArrayRef<Rule> rules, ArrayRef<std::string> nameVec,
                    ArrayRef<size_t> activeArgs,
                    DenseMap<size_t, DenseSet<size_t>> &argUsers) {

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto name = nameVec[i];
    DenseSet<size_t> set{};
    for (auto &&rule : enumerate(rules)) {
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

TGPattern::TGPattern(Record *r) : blasName(r->getNameInitAsString()) {
  fillArgs(r, args, argNameToPos);
  fillArgTypes(r, argTypes);
  fillRelatedLenghts(r, argNameToPos, argTypes, relatedLengths);

  if (argTypes.lookup(0) == ArgType::cblas_layout) {
    BLASLevel2or3 = true;
  } else {
    BLASLevel2or3 = false;
  }

  fillActiveArgSet(r, posActArgs);
  fillMutableArgSet(r, mutables);

  // Now create the rules for this pattern
  {
    ListInit *derivOps = r->getValueAsListInit("ArgDerivatives");
    for (auto &&derivOp : enumerate(*derivOps)) {
      DagInit *derivRule = cast<DagInit>(derivOp.value());
      size_t actIdx = posActArgs[derivOp.index()];
      rules.push_back(
          Rule(derivRule, actIdx, argNameToPos, argTypes, mutables));
    }
  }

  fillArgUserMap(rules, args, posActArgs, argUsers);
}

SmallVector<size_t, 3> TGPattern::getRelatedLengthArgs(size_t arg) const {
  auto ty = argTypes.lookup(arg);
  // other args are unrelated to length args
  assert(ty == ArgType::vincData || ty == ArgType::mldData ||
         ty == ArgType::ap);

  assert(relatedLengths.count(arg) == 1);
  auto related = relatedLengths.lookup(arg);

  if (related.size() == 3) {
    assert(argTypes.lookup(related[0]) == ArgType::trans);
  }

  return related;
}

bool TGPattern::isBLASLevel2or3() const { return BLASLevel2or3; }

const DenseMap<size_t, DenseSet<size_t>> &TGPattern::getArgUsers() const {
  return argUsers;
}

StringRef TGPattern::getName() const { return blasName; }

ArrayRef<std::string> TGPattern::getArgNames() const { return args; }

const StringMap<size_t> &TGPattern::getArgNameMap() const {
  return argNameToPos;
}

const DenseMap<size_t, ArgType> &TGPattern::getArgTypeMap() const {
  return argTypes;
}

const DenseSet<size_t> &TGPattern::getMutableArgs() const { return mutables; }

ArrayRef<size_t> TGPattern::getActiveArgs() const { return posActArgs; }

ArrayRef<Rule> TGPattern::getRules() const { return rules; }

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

using namespace llvm;

enum argType {
  fp,
  len,
  vincData,
  vincInc,
  cblas_layout, // is special (no non-blas equivalent)
  // following for lv2 only
  mldData,
  mldLD,
  uplo,
  trans,
  diag,
  side
};

class Arg {
  public:
    size_t pos;
    std::string name;
};

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

// Subset of the general pattern info, 
// but only the part that affects the specific argument being active.
class Rule {
  private: 
    DagInit *rewriteRule;
    // which argument from the primary function do we handle here?
    size_t activeArg;
    StringMap<size_t> argNameToPos;
    DenseMap<size_t, argType> argTypes;
    DenseSet<size_t> mutables;
    bool BLASLevel2or3;

  public:
    Rule(DagInit *dag, size_t activeArgIdx, StringMap<size_t> &patternArgs,
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
    bool isBLASLevel2or3() { return BLASLevel2or3; }
    DagInit *getRuleDag() { return rewriteRule; }
    size_t getHandledArgIdx() { return activeArg; }
    StringMap<size_t> getArgNameMap() { return argNameToPos; }
    DenseMap<size_t, argType> getArgTypeMap() { return argTypes; }
    //std::string to_string(Rule const&r) {
    //  std::string res = "function: " + r.blasName + "\n";
    //  res += "handling

    //  for (auto rule : r.rules) {
    //  }
    //}
};

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

void fillArgs(const Record *r, SmallVector<std::string, 6> &args,
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

void fillArgUserMap(SmallVector<Rule, 3> &rules,
                    const SmallVector<std::string, 6> &nameVec,
                    const SmallVector<size_t, 4> &activeArgs,
                    DenseMap<size_t, DenseSet<size_t>> &argUsers) {

  for (size_t i = 0; i < nameVec.size(); i++) {
    auto name = nameVec[i];
    DenseSet<size_t> set{};
    for (auto rule : llvm::enumerate(rules)) {
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

// A single Blas function, including replacement rules. E.g. scal, axpy, ...
class TGPattern {
private:
  std::string blasName;
  bool BLASLevel2or3;

  // All args from the primary blas function
  SmallVector<std::string, 6> args;

  // Map arg name to their position (in primary fnc)
  StringMap<size_t> argNameToPos;

  // Type of these args, e.g. FP-scalar, int, FP-vec, ..
  DenseMap<size_t, argType> argTypes;

  // Args that could be set to active (thus fp based)
  SmallVector<size_t, 4> posActArgs;

  // Args that will be modified by primary function (e.g. x in scal)
  DenseSet<size_t> mutables;

  // One rule for each possibly active arg
  SmallVector<Rule, 3> rules;

  // Based on an argument name, which rules use this argument?
  DenseMap<size_t, DenseSet<size_t>> argUsers;

public:
  TGPattern(Record &r) {
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
      for (auto derivOp : llvm::enumerate(*derivOps)) {
        DagInit *derivRule = cast<DagInit>(derivOp.value());
        size_t actIdx = posActArgs[derivOp.index()];
        rules.push_back(
            Rule(derivRule, actIdx, argNameToPos, argTypes, mutables));
      }
    }

    argUsers = DenseMap<size_t, DenseSet<size_t>>();
    fillArgUserMap(rules, args, posActArgs, argUsers);
    // for (auto key : argUsers) {
    //   DenseSet<size_t> users = key.second; // argUsers.lookup(key);
    //   llvm::errs() << "\nKey " << key.first << ": ";
    //   for (auto user: users) {
    //     llvm::errs() << user << " ";
    //   }
    //   llvm::errs() << "\n";
    // }
  }
  SmallVector<size_t, 2> getRelatedLengthArgs(size_t arg) {
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
  bool isBLASLevel2or3() { return BLASLevel2or3; }
  DenseMap<size_t, DenseSet<size_t>> getArgUsers() { return argUsers; }
  std::string getName() { return blasName; }
  SmallVector<std::string, 6> getArgNames() { return args; }
  StringMap<size_t> getArgNameMap() { return argNameToPos; }
  DenseMap<size_t, argType> getArgTypeMap() { return argTypes; }
  DenseSet<size_t> getMutableArgs() { return mutables; }
  SmallVector<size_t, 4> getActiveArgs() { return posActArgs; }
  SmallVector<Rule, 3> getRules() { return rules; }
};

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

enum argType { fp, len, vincData, vincInc };

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
      // TODO: handle input<"x">, adj<"x"> and similar
      auto name = toSearch->getArgNameStr(i);
      if (name == toFind)
        return true;
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

      for (auto ruleArgKey : argNameToPos.keys()) {
        //        3) look up and eventually copy mutable
        auto val = argNameToPos.lookup(ruleArgKey);
        if (patternMutables.find(val) != patternMutables.end()) {
          mutables.insert(*patternMutables.find(val));
        }
      }
    }
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
    } else {
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
    // if (blasName != "scal") {
    //   llvm::errs() << blasName << " skipped!\n";
    //   return;
    // }
    // llvm::errs() << blasName << "\n";

    args = llvm::SmallVector<std::string, 6>();
    argNameToPos = StringMap<size_t>{};
    fillArgs(&r, args, argNameToPos);

    argTypes = DenseMap<size_t, argType>();
    fillArgTypes(&r, argTypes);

    posActArgs = SmallVector<size_t, 4>();
    fillActiveArgSet(&r, posActArgs);

    mutables = DenseSet<size_t>();
    fillMutableArgSet(&r, mutables);

    // Now create the rules for this pattern
    {
      rules = llvm::SmallVector<Rule, 3>{};
      ListInit *derivOps = r.getValueAsListInit("ArgDerivatives");
      for (auto derivOp : llvm::enumerate(*derivOps)) {
        // llvm::errs() << derivOp.index() << ": \n";
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
    // needs to be adjusted for the gemv branch
    assert(argTypes.lookup(arg) == argType::vincData);
    return {0};
  }
  DenseMap<size_t, DenseSet<size_t>> getArgUsers() { return argUsers; }
  std::string getName() { return blasName; }
  SmallVector<std::string, 6> getArgNames() { return args; }
  StringMap<size_t> getArgNameMap() { return argNameToPos; }
  DenseMap<size_t, argType> getArgTypeMap() { return argTypes; }
  DenseSet<size_t> getMutableArgs() { return mutables; }
  SmallVector<size_t, 4> getActiveArgs() { return posActArgs; }
  SmallVector<Rule, 3> getRules() { return rules; }
};

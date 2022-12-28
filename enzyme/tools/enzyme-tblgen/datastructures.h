

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

enum argType { fp, len, vinc, vincData, vincInc };

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
    DenseMap<StringRef, size_t> argNameToPos;
    DenseMap<size_t, argType> argTypes;
    DenseSet<size_t> mutables;
    // Eventually also add posActArg ?

  public:
    Rule(DagInit *dag, DenseMap<StringRef, size_t> &patternArgs,
         DenseMap<size_t, argType> &patternTypes,
         DenseSet<size_t> &patternMutables) {

      rewriteRule = dag;

      argNameToPos = DenseMap<StringRef, size_t>();
      argTypes = DenseMap<size_t, argType>();
      mutables = DenseSet<size_t>();

      // For each arg found in the dag: 
      //        1) copy patternArgs to ruleArgs if arg shows up in this rule
      for (auto patternArg : patternArgs) {
        StringRef argName = patternArg.first;
        size_t argPos = patternArg.second;
        bool argUsedInRule = isArgUsed(argName, rewriteRule);
        if (argUsedInRule) {
          argNameToPos.insert(patternArg);
          //        2) look up and copy the corresponding argType
          assert(patternTypes.find(argPos) != patternTypes.end() &&
                 "arg without corresponding type");
          argTypes.insert(*patternTypes.find(argPos));
        }
      }

      for (auto ruleArg : argNameToPos) {
        //        3) look up and eventually copy mutable
        if (patternMutables.find(ruleArg.second) != patternMutables.end()) {
          mutables.insert(*patternMutables.find(ruleArg.second));
        }
      }
    }
};

void fillActiveArgSet(const Record *pattern, SmallVector<size_t> &activeArgs) {

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
              DenseMap<StringRef, size_t> &argNameToPos) {
  DagInit *argOps = r->getValueAsDag("PatternToMatch");
  size_t numArgs = argOps->getNumArgs();
  args.reserve(numArgs);
  for (size_t i = 0; i < numArgs; i++) {
    args.push_back(argOps->getArgNameStr(i).str());
    argNameToPos.insert(std::pair<StringRef, size_t>(StringRef(args[i]), i));
  }
  assert(args.size() == numArgs);
  assert(argNameToPos.size() == numArgs);
}

// A single Blas function, including replacement rules. E.g. scal, axpy, ...
class TGPattern {
private:
  std::string blasName;
  // All args from the primary blas function
  SmallVector<std::string, 6> args;
  // Map arg name to their position (in primary fnc)
  DenseMap<StringRef, size_t> argNameToPos;
  // Type of these args, e.g. FP-scalar, int, FP-vec, ..
  DenseMap<size_t, argType> argTypes;
  // Args that could be set to active (thus fp based)
  // Vector, since insertion order is important
  SmallVector<size_t> posActArgs;
  // Args that will be modified by primary function (e.g. x in scal)
  DenseSet<size_t> mutables;
  // One rule for each possibly active arg
  SmallVector<Rule, 3> rules;
  // Based on an argument name, which rules use this argument?
  // DenseMap<StringRef, DenseSet<size_t>> argUsers;

public:
  TGPattern(Record &r) {
    blasName = r.getNameInitAsString();
    PrintNote("blasName: " + blasName);

    args = llvm::SmallVector<std::string, 6>();
    argNameToPos = DenseMap<StringRef, size_t>{};
    fillArgs(&r, args, argNameToPos);

    argTypes = DenseMap<size_t, argType>();
    fillArgTypes(&r, argTypes);

    posActArgs = SmallVector<size_t>();
    fillActiveArgSet(&r, posActArgs);

    mutables = DenseSet<size_t>();
    fillMutableArgSet(&r, mutables);

    // Now create the rules for this pattern
    {
      rules = llvm::SmallVector<Rule, 3>{};
      ListInit *argOps = r.getValueAsListInit("ArgDerivatives");
      for (auto argOp : *argOps) {
        DagInit *derivRule = cast<DagInit>(argOp);
        rules.push_back(Rule(derivRule, argNameToPos, argTypes, mutables));
      }
    }

    // argUsers = DenseMap<StringRef, DenseSet<size_t>>();
    // TODO: fill
  }
  static int elemPerArgType(argType a) {
    if (a == argType::vinc)
      return 2;
    return 1;
    // TODO: adjust later for blas 2 and so on
  }
  std::string getName() { return blasName; }
  SmallVector<std::string, 6> getArgNames() { return args; }
  DenseMap<StringRef, size_t> getArgNameMap() { return argNameToPos; }
  DenseMap<size_t, argType> getArgTypeMap() { return argTypes; }
  SmallVector<size_t> getActiveArgs() { return posActArgs; }
};

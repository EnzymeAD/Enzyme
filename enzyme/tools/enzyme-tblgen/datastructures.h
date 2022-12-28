

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


enum argType {fp=1, len=1, vinc=2, vincData=1, vincInc=1};

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
    DenseMap<StringRef, std::string> argTypes;
    DenseSet<StringRef> mutables;
    // Eventually also add posActArg ?

  public:
    Rule(DagInit * dag, 
        DenseMap<StringRef, size_t> &patternArgs, 
        DenseMap<StringRef, std::string> &patternTypes,
        DenseSet<StringRef> &patternMutables) {

      rewriteRule = dag;

      argNameToPos = DenseMap<StringRef, size_t>();
      argTypes = DenseMap<StringRef, std::string>();
      mutables = DenseSet<StringRef>();

      // For each arg found in the dag: 
      //        1) copy patternArgs to ruleArgs if arg shows up in this rule
      for (auto patternArg : patternArgs) {
        StringRef argName = patternArg.first;
        bool argUsedInRule = isArgUsed(argName, rewriteRule);
        if (argUsedInRule) {
          argNameToPos.insert(patternArg);
          //        2) look up and copy the corresponding argType
          assert(patternTypes.find(argName) != patternTypes.end() &&
                 "arg without corresponding type");
          argTypes.insert(*patternTypes.find(argName));
        }
      }

      for (auto ruleArg : argNameToPos) {
        //        3) look up and eventually copy mutable
        if (patternMutables.find(ruleArg.first) != patternMutables.end()) {
          mutables.insert(*patternMutables.find(ruleArg.first));
        }

      }

    }
};

void fillActiveArgSet(const Record *pattern, 
    const SmallVector<std::string, 6>  &patternArgs, 
    DenseSet<StringRef> &activeArgs ) {

  std::vector<Record *> inputTypes =
      pattern->getValueAsListOfDefs("inputTypes");
  size_t numTypes = 0;
  for (auto val : inputTypes) {
    if (val->getValueAsBit("active")) {
      activeArgs.insert(StringRef(patternArgs[numTypes]));
    }
    numTypes += val->getValueAsInt("nelem");
  }
}

void fillMutableArgSet(const Record *pattern, 
    const SmallVector<std::string, 6>  &patternArgs, 
    DenseSet<StringRef> &mutables ) {

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
      mutables.insert(StringRef(patternArgs[pos]));
    }

    assert(mutables.size() == mutableArgs.size());
}

void fillArgTypes(const Record *pattern, 
    const SmallVector<std::string, 6> &args,
    DenseMap<StringRef, std::string> &argTypes) {

  std::vector<Record *> inputTypes =
    pattern->getValueAsListOfDefs("inputTypes");
  size_t pos = 0;
  for (auto val : inputTypes) {
    if (val->getName() == "len") {
      argTypes.insert(std::make_pair(StringRef(args[pos]), "len"));
    } else if (val->getName() == "fp") {
      argTypes.insert(std::make_pair(StringRef(args[pos]), "fp"));
    } else if (val->getName() == "vinc") {
      argTypes.insert(std::make_pair(StringRef(args[pos]), "vincData"));
      argTypes.insert(std::make_pair(StringRef(args[pos + 1]), "vincInc"));
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
  // Just fill a set with the names to assert uniquenes
  DenseSet<std::string> uniqueNames;
  for (size_t i = 0; i < numArgs; i++) {
    args.push_back(argOps->getArgNameStr(i).str());
    uniqueNames.insert(args[i]);
    argNameToPos.insert(std::pair<StringRef, size_t>(StringRef(args[i]), i));
  }
  assert(uniqueNames.size() == numArgs);
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
  DenseMap<StringRef, std::string> argTypes;
  // Args that could be set to active (thus fp based)
  DenseSet<StringRef> posActArgs;
  // Args that will be modified by primary function (e.g. x in scal)
  DenseSet<StringRef> mutables;
  // One rule for each possibly active arg
  SmallVector<Rule, 3> rules;
  // Based on an argument name, which rules use this argument?
  DenseMap<StringRef, DenseSet<size_t>> argUsers;

public:
  TGPattern(Record &r) {
    blasName = r.getNameInitAsString();
    PrintNote("blasName: " + blasName);

    args = llvm::SmallVector<std::string, 6>();
    argNameToPos = DenseMap<StringRef, size_t>{};
    fillArgs(&r, args, argNameToPos);

    argTypes = DenseMap<StringRef, std::string>();
    fillArgTypes(&r, args, argTypes);

    posActArgs = DenseSet<StringRef>();
    fillActiveArgSet(&r, args, posActArgs);

    mutables = DenseSet<StringRef>();
    fillMutableArgSet(&r, args, mutables);

    // Now create the rules for this pattern
    {
      rules = llvm::SmallVector<Rule, 3>{};
      ListInit *argOps = r.getValueAsListInit("ArgDerivatives");
      for (auto argOp : *argOps) {
        DagInit *derivRule = cast<DagInit>(argOp);
        rules.push_back(Rule(derivRule, argNameToPos, argTypes, mutables));
      }
    }

    argUsers = DenseMap<StringRef, DenseSet<size_t>>();
    // TODO: fill
  }
};

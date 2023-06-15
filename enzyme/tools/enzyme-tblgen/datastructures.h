
#ifndef ENZYME_TBLGEN_DATASTRUCT_H
#define ENZYME_TBLGEN_DATASTRUCT_H 1

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


enum class argType {
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
namespace llvm {
raw_ostream &operator<<(raw_ostream &os, const argType &arg);
}
using namespace llvm;

const char *TyToString(argType ty);

bool isArgUsed(const StringRef toFind, const DagInit *toSearch);

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
       DenseSet<size_t> &patternMutables);
  bool isBLASLevel2or3();
  DagInit *getRuleDag();
  size_t getHandledArgIdx();
  StringMap<size_t> getArgNameMap();
  DenseMap<size_t, argType> getArgTypeMap();
  std::string to_string();
};

void fillActiveArgSet(const Record *pattern,
                      SmallVectorImpl<size_t> &activeArgs);

void fillMutableArgSet(const Record *pattern, DenseSet<size_t> &mutables);

void fillArgTypes(const Record *pattern, DenseMap<size_t, argType> &argTypes);

void fillArgs(const Record *r, SmallVectorImpl<std::string> &args,
              StringMap<size_t> &argNameToPos);

void fillArgUserMap(SmallVectorImpl<Rule> &rules, ArrayRef<std::string> nameVec,
                    // ArrayRef<StringRef> nameVec,
                    ArrayRef<size_t> activeArgs,
                    DenseMap<size_t, DenseSet<size_t>> &argUsers);

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
  TGPattern(Record &r);
  SmallVector<size_t, 2> getRelatedLengthArgs(size_t arg);
  bool isBLASLevel2or3();
  DenseMap<size_t, DenseSet<size_t>> getArgUsers();
  std::string getName();
  SmallVector<std::string, 6> getArgNames();
  StringMap<size_t> getArgNameMap();
  DenseMap<size_t, argType> getArgTypeMap();
  DenseSet<size_t> getMutableArgs();
  SmallVector<size_t, 4> getActiveArgs();
  SmallVector<Rule, 3> getRules();
};

#endif

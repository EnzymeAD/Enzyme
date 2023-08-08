
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

enum class ArgType {
  fp,
  ap,
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
raw_ostream &operator<<(raw_ostream &os, ArgType arg);
raw_ostream &operator<<(raw_fd_ostream &os, ArgType arg);
} // namespace llvm

using namespace llvm;

const char *TyToString(ArgType ty);
bool isVecLikeArg(ArgType ty);

bool isArgUsed(StringRef toFind, const DagInit *toSearch);

/// Subset of the general pattern info,
/// but only the part that affects the specific argument being active.
class Rule {
private:
  DagInit *rewriteRule;
  // which argument from the primary function do we handle here?
  size_t activeArg;
  StringMap<size_t> argNameToPos;
  DenseMap<size_t, ArgType> argTypes;
  DenseSet<size_t> mutables;
  bool BLASLevel2or3;

public:
  Rule(DagInit *dag, size_t activeArgIdx, const StringMap<size_t> &patternArgs,
       const DenseMap<size_t, ArgType> &patternTypes,
       const DenseSet<size_t> &patternMutables);
  bool isBLASLevel2or3() const;
  DagInit *getRuleDag();
  size_t getHandledArgIdx() const;
  const StringMap<size_t> &getArgNameMap() const;
  const DenseMap<size_t, ArgType> &getArgTypeMap() const;
  std::string to_string() const;
};

void fillActiveArgSet(const Record *pattern,
                      SmallVectorImpl<size_t> &activeArgs);

void fillMutableArgSet(const Record *pattern, DenseSet<size_t> &mutables);

void fillArgTypes(const Record *pattern, DenseMap<size_t, ArgType> &argTypes);

void fillArgs(const Record *r, SmallVectorImpl<std::string> &args,
              const StringMap<size_t> &argNameToPos);

void fillArgUserMap(ArrayRef<Rule> rules, ArrayRef<std::string> nameVec,
                    ArrayRef<size_t> activeArgs,
                    DenseMap<size_t, DenseSet<size_t>> &argUsers);

/// A single Blas function, including replacement rules. E.g. scal, axpy, ...
class TGPattern {
private:
  std::string blasName;
  bool BLASLevel2or3;

  /// All args from the primary blas function
  SmallVector<std::string, 6> args;

  /// Map arg name to their position (in primary fnc)
  StringMap<size_t> argNameToPos;

  /// Type of these args, e.g. FP-scalar, int, FP-vec, ..
  DenseMap<size_t, ArgType> argTypes;

  /// Args that could be set to active (thus fp based)
  SmallVector<size_t, 4> posActArgs;

  /// Args that will be modified by primary function (e.g. x in scal)
  DenseSet<size_t> mutables;

  /// One rule for each possibly active arg
  SmallVector<Rule, 3> rules;

  /// Based on an argument name, which rules use this argument?
  DenseMap<size_t, DenseSet<size_t>> argUsers;

  /// For matrix or vactor types, helps to find the length argument(s)
  DenseMap<size_t, SmallVector<size_t, 3>> relatedLengths;

public:
  TGPattern(Record *r);
  SmallVector<size_t, 3> getRelatedLengthArgs(size_t arg) const;
  bool isBLASLevel2or3() const;
  const DenseMap<size_t, DenseSet<size_t>> &getArgUsers() const;
  StringRef getName() const;
  ArrayRef<std::string> getArgNames() const;
  const StringMap<size_t> &getArgNameMap() const;
  const DenseMap<size_t, ArgType> &getArgTypeMap() const;
  const DenseSet<size_t> &getMutableArgs() const;
  ArrayRef<size_t> getActiveArgs() const;
  ArrayRef<Rule> getRules() const;
};

#endif

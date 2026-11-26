//===- Flags.cpp - the cl::opt of every Poseidon flag ---------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sites/Annotations.h"
#include "Support/Diagnostics.h"
#include "Support/Options.h"
#include "Support/Utils.h"

#include "poseidon/poseidon.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"

#include <cmath>
#include <limits>
#include <string>

using namespace llvm;

namespace poseidon {

// The directory a profiling run writes and a solve reads when neither is told
// otherwise; the same constant the profiler runtimes default to.
static constexpr StringLiteral kDefaultProfileDir =
    POSEIDON_DEFAULT_PROFILE_DIR;

static cl::OptionCategory Category("Poseidon options");

// Reads an empty value as NaN, where cl::parser<double> reads 0, so that the
// solve refuses `-poseidon-tau=` from a variable that was never set instead of
// solving without a target.
class NumberParser : public cl::parser<double> {
public:
  using cl::parser<double>::parser;
  bool parse(cl::Option &O, StringRef ArgName, StringRef Arg, double &Value) {
    if (!Arg.empty())
      return cl::parser<double>::parse(O, ArgName, Arg, Value);
    Value = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
};

// The cl::opt or cl::list of a flag of each kind.
template <OptionKind K> struct Flag {
  using type = cl::opt<OptionValue<K>>;
};
template <> struct Flag<OptionKind::OptionalUnsigned> {
  using type = cl::opt<unsigned>;
};
template <> struct Flag<OptionKind::Number> {
  using type = cl::opt<double, false, NumberParser>;
};
template <> struct Flag<OptionKind::StringList> {
  using type = cl::list<std::string>;
};
template <> struct Flag<OptionKind::UnsignedList> {
  using type = cl::list<unsigned>;
};

// What a flag of kind K takes besides its name and text: its default, or how
// a list or an optional value is spelled.
template <OptionKind K, typename T>
static auto valueModifier(const T &Default) {
  if constexpr (K == OptionKind::StringList || K == OptionKind::UnsignedList)
    return cl::CommaSeparated;
  else if constexpr (K == OptionKind::Directory)
    return cl::ValueOptional;
  else if constexpr (K == OptionKind::OptionalUnsigned)
    return cl::ValueRequired;
  else
    return cl::init(Default);
}

static constexpr cl::OptionHidden visibility(OptionRole Role) {
  return Role == OptionRole::User ? cl::NotHidden : cl::Hidden;
}

template <typename T, typename P>
static T valueOf(const cl::opt<T, false, P> &Flag) {
  return Flag.getValue();
}
template <typename T> static std::vector<T> valueOf(const cl::list<T> &Flag) {
  return std::vector<T>(Flag.begin(), Flag.end());
}
// The value of a flag of kind K: for an optional one, none unless it is given.
template <OptionKind K, typename F> static OptionValue<K> read(const F &Flag) {
  if constexpr (K == OptionKind::OptionalUnsigned)
    return Flag.getNumOccurrences() ? OptionValue<K>(Flag.getValue())
                                    : std::nullopt;
  else
    return valueOf(Flag);
}

// Before the flags, which take their defaults from it.
static const Options kDefaults;

namespace flags {
#define POSEIDON_OPTION(Group, Field, Kind, Default, Spelling, Role, Key,      \
                        ValueName, Description)                                \
  static Flag<OptionKind::Kind>::type Group##Field(                            \
      Spelling, cl::desc(Description), cl::value_desc(ValueName),              \
      visibility(OptionRole::Role), cl::cat(Category),                         \
      valueModifier<OptionKind::Kind>(kDefaults.Group.Field));
#include "Support/Options.def"
} // namespace flags

Options readOptions() {
  Options O;
#define POSEIDON_OPTION(Group, Field, Kind, ...)                               \
  O.Group.Field = read<OptionKind::Kind>(flags::Group##Field);
#include "Support/Options.def"
  for (auto [flag, dir] :
       {std::pair(&flags::ProfilingProfileGenerate,
                  &O.Profiling.ProfileGenerate),
        std::pair(&flags::ProfilingProfileUse, &O.Profiling.ProfileUse)})
    if (flag->getNumOccurrences() && dir->empty())
      *dir = kDefaultProfileDir.str();
  // Rewriting runs Herbie: left to its default, it is off when no Herbie can
  // run, except under -poseidon-strict-mode, where a missing Herbie stays an
  // error.
  if (!flags::CandidatesEnableRewrite.getNumOccurrences() &&
      !O.Model.StrictMode && !sys::fs::exists(herbieProgram(O.Herbie))) {
    O.Candidates.EnableRewrite = false;
    O.Candidates.RewritesLackHerbie = true;
  }
  std::vector<unsigned> &terms = O.Herbie.NumTaylorTerms;
  llvm::sort(terms);
  terms.erase(llvm::unique(terms), terms.end());
  return O;
}

static Error noNumber(StringRef flag) {
  return userError("-" + flag +
                   " was given an empty value or nan; write a number after "
                   "'=', or leave the flag out for its default");
}

// No solution saves more than the whole profiled cost, so a budget below -1 is
// a mistake in the flag, not a solve that finds nothing. The DP's cost axis
// puts the profiled cost below 2^33 of its units in 2^63, so it holds every
// share up to 2^30.
static Error validateCostBudget(double budget) {
  if (std::isnan(budget))
    return noNumber("poseidon-cost-budget");
  if (std::isfinite(budget) && budget > 0x1p30)
    return userError("-poseidon-cost-budget=" + Twine(exactDouble(budget)) +
                     " is above 2^30, the largest share of the profiled cost "
                     "the solve holds; give a share of at most 1073741824");
  if (std::isfinite(budget) && budget >= -1.0)
    return Error::success();
  return userError(
      "-poseidon-cost-budget=" + Twine(exactDouble(budget)) +
      (std::isfinite(budget)
           ? " asks a solution to save more than the whole profiled cost"
           : " is not a finite number") +
      "; give the share of a site's profiled cost a solution may add, "
      "negative for one it has to save (-0.25 saves a quarter)");
}

Error validateOptions(const Options &O) {
  const HerbieOptions &H = O.Herbie;
  if (!is_contained({"inputs", "executions"}, H.ArrayGrouping))
    return userError(
        "-poseidon-herbie-array-grouping=" + Twine(H.ArrayGrouping) +
        " is not 'inputs' or 'executions'");
  if (!is_contained({"sensitivity", "gradient", "none"}, H.ArrayWeights))
    return userError("-poseidon-herbie-array-weights=" + Twine(H.ArrayWeights) +
                     " is not 'sensitivity', 'gradient' or 'none'");
  if (std::string err; !O.Profiling.Kernels.empty() &&
                       O.Profiling.Kernels != "all" &&
                       !Regex(O.Profiling.Kernels).isValid(err))
    return userError(Twine("-poseidon-kernels=") + O.Profiling.Kernels +
                     " is not a valid regular expression (" + err +
                     "); give a regular expression of kernel names, or "
                     "'all'");
  if (!O.Profiling.usesProfile())
    return Error::success();
  const std::pair<StringRef, double> numbers[] = {
      {"poseidon-tau", O.Solver.Tau},
      {"poseidon-min-cost-share", O.Profiling.MinCostShare},
      {"poseidon-rewrite-max-broken-share", O.Model.RewriteMaxBrokenShare}};
  for (const auto &[flag, value] : numbers)
    if (std::isnan(value))
      return noNumber(flag);
  // Only a solve with no accuracy target selects by the budget.
  if (O.Solver.Tau <= 0.0 && O.Solver.Pin.empty())
    return validateCostBudget(O.Solver.CostBudget);
  return Error::success();
}

// What one compile was asked to do, as the spellings and values of its flags.

static std::string value(bool v) { return v ? "1" : "0"; }
static std::string value(double v) { return formatv("{0:e16}", v).str(); }
static std::string value(const std::string &v) { return v; }
static std::string value(const std::optional<unsigned> &v) {
  return v ? std::to_string(*v) : "";
}
static std::string value(const std::vector<std::string> &v) {
  return join(v, ",");
}
static std::string value(const std::vector<unsigned> &v) {
  return join(map_range(v, [](unsigned n) { return std::to_string(n); }), ",");
}
template <typename T> static std::string value(T v) {
  return std::to_string(v);
}

std::string herbieProgram(const HerbieOptions &H) {
  return H.Binary.empty() ? POSEIDON_HERBIE_DEFAULT_BINARY : H.Binary;
}

FlagValues solveFlagValues(const Options &O) {
  FlagValues out;
#define POSEIDON_OPTION(Group, Field, Kind, Default, Spelling, Role, Key, ...) \
  if (OptionKey::Key == OptionKey::SolveInput)                                 \
    out.emplace_back(Spelling, value(O.Group.Field));
#include "Support/Options.def"
  return out;
}

FlagValues notSolveInputs(const Options &O) {
  FlagValues out;
#define POSEIDON_OPTION(Group, Field, Kind, Default, Spelling, Role, Key, ...) \
  if (OptionKey::Key == OptionKey::Other)                                      \
    out.emplace_back(Spelling, value(O.Group.Field));
#include "Support/Options.def"
  return out;
}

FlagValues allFlagValues(const Options &O) {
  FlagValues all = solveFlagValues(O);
  all.append(notSolveInputs(O));
  return all;
}

} // namespace poseidon

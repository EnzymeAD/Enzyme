//===- Flags.cpp - the cl::opt of every Poseidon flag ---------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/Options.h"
#include "IR/Utils.h"
#include "Sites/Annotations.h"

#include "poseidon/poseidon.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"

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

// Before the flags, which take their defaults from it.
static const Options kDefaults;

namespace flags {
#define POSEIDON_OPTION(Group, Field, Kind, Default, Spelling, Role, Key,      \
                        ValueName, Description)                                \
  static Flag<OptionKind::Kind>::type Field(                                   \
      Spelling, cl::desc(Description), cl::value_desc(ValueName),              \
      visibility(OptionRole::Role), cl::cat(Category),                         \
      valueModifier<OptionKind::Kind>(kDefaults.Group.Field));
#include "Core/Options.def"
} // namespace flags

Options readOptions() {
  Options O;
#define POSEIDON_OPTION(Group, Field, ...)                                     \
  O.Group.Field = valueOf(flags::Field);
#include "Core/Options.def"
  if (flags::ProfileUse.getNumOccurrences() && O.Profiling.ProfileUse.empty())
    O.Profiling.ProfileUse = kDefaultProfileDir.str();
  std::vector<unsigned> &terms = O.Herbie.HerbieNumTaylorTerms;
  llvm::sort(terms);
  terms.erase(llvm::unique(terms), terms.end());
  return O;
}

[[noreturn]] static void reportNoNumber(LLVMContext &Ctx, StringRef flag) {
  reportUserError(Ctx, "-" + flag +
                           " was given no number: its value is empty or nan. "
                           "Write a number after '=', or leave the flag out "
                           "for its default.");
}

// No solution saves more than the whole profiled cost, so a budget below -1 is
// a mistake in the flag, not a solve that finds nothing. The DP's cost axis
// puts the profiled cost below 2^33 of its units in 2^63, so it holds every
// share up to 2^30.
static void validateCostBudget(double budget, LLVMContext &Ctx) {
  if (std::isnan(budget))
    reportNoNumber(Ctx, "poseidon-cost-budget");
  if (std::isfinite(budget) && budget > 0x1p30)
    reportUserError(Ctx, "-poseidon-cost-budget=" + Twine(exactDouble(budget)) +
                             " lets a solution add more than 2^30 times the "
                             "profiled cost, which the DP's cost axis does "
                             "not hold; give a share of at most 1073741824.");
  if (std::isfinite(budget) && budget >= -1.0)
    return;
  reportUserError(
      Ctx, "-poseidon-cost-budget=" + Twine(exactDouble(budget)) +
               (std::isfinite(budget)
                    ? " asks a solution to save more than the whole profiled "
                      "cost"
                    : " is not a finite number") +
               ". It takes the share of a site's profiled cost a solution may "
               "add, negative for one it has to save (-0.25 requires saving a "
               "quarter), not a cost. The frontier of a site's solve, with "
               "the budget that selects each point, is in the "
               "<site>.explain.json -poseidon-explain=<dir> writes.");
}

void validateOptions(const Options &O, LLVMContext &Ctx) {
  const HerbieOptions &H = O.Herbie;
  if (!is_contained({"inputs", "executions"}, H.HerbieArrayGrouping))
    reportUserError(
        Ctx, "-poseidon-herbie-array-grouping=" + Twine(H.HerbieArrayGrouping) +
                 " is not 'inputs' or 'executions'.");
  if (!is_contained({"sensitivity", "gradient", "none"}, H.HerbieArrayWeights))
    reportUserError(
        Ctx, "-poseidon-herbie-array-weights=" + Twine(H.HerbieArrayWeights) +
                 " is not 'sensitivity', 'gradient' or 'none'.");
  if (!O.Profiling.usesProfile())
    return;
  const std::pair<StringRef, double> numbers[] = {
      {"poseidon-tau", O.Solver.Tau},
      {"poseidon-min-cost-share", O.Profiling.MinCostShare},
      {"poseidon-max-broken-share", O.Model.MaxBrokenShare},
      {"poseidon-exponent-penalty", O.Model.ExponentPenalty}};
  for (const auto &[flag, value] : numbers)
    if (std::isnan(value))
      reportNoNumber(Ctx, flag);
  // Only a solve with no accuracy target selects by the budget.
  if (O.Solver.Tau <= 0.0 && O.Solver.ApplyRewrites.empty())
    validateCostBudget(O.Solver.CostBudget, Ctx);
}

} // namespace poseidon

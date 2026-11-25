//===- Options.h - what one compile was asked to do -----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One field per -poseidon-* flag of Options.def, in the struct of the flag's
// group. A field's initializer is its flag's default, so a default-constructed
// Options is a compile given no flag.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_OPTIONS_H
#define POSEIDON_OPTIONS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace poseidon {

/// The type of an option's field and how its flag is parsed.
enum class OptionKind {
  Bool,
  Unsigned,
  /// An unsigned that may be left unset.
  OptionalUnsigned,
  Int,
  /// A double; an empty value reads as NaN, which validateOptions refuses.
  Number,
  String,
  /// Comma-separated, in the order given.
  StringList,
  UnsignedList,
  /// A path that may be given without a value.
  Directory,
};

/// Whether -help lists an option's flag or only -help-hidden does.
enum class OptionRole { User, Hidden };

/// Whether a DP table depends on an option.
enum class OptionKey { SolveInput, Other };

/// The C++ type of an option of kind `K`, as `type`.
template <OptionKind K> struct OptionField;
template <> struct OptionField<OptionKind::Bool> {
  using type = bool;
};
template <> struct OptionField<OptionKind::Unsigned> {
  using type = unsigned;
};
template <> struct OptionField<OptionKind::OptionalUnsigned> {
  using type = std::optional<unsigned>;
};
template <> struct OptionField<OptionKind::Int> {
  using type = int;
};
template <> struct OptionField<OptionKind::Number> {
  using type = double;
};
template <> struct OptionField<OptionKind::String> {
  using type = std::string;
};
template <> struct OptionField<OptionKind::StringList> {
  using type = std::vector<std::string>;
};
template <> struct OptionField<OptionKind::UnsignedList> {
  using type = std::vector<unsigned>;
};
template <> struct OptionField<OptionKind::Directory> {
  using type = std::string;
};
/// The C++ type of an option of kind `K`.
template <OptionKind K> using OptionValue = typename OptionField<K>::type;

#define POSEIDON_FIELD(Field, Kind, Default, ...)                              \
  OptionValue<OptionKind::Kind> Field = Default;

/// Profile generation and use, and which kernels are sites.
struct ProfilingOptions {
#define POSEIDON_PROFILING_OPTION POSEIDON_FIELD
#include "Options.def"

  /// The compile builds the program to write profiles.
  bool generatesProfile() const { return !ProfileGenerate.empty(); }
  /// The compile solves from profiles.
  bool usesProfile() const { return !ProfileUse.empty(); }
  /// Whether this compile generates or uses a profile: anything else leaves
  /// every site as written.
  bool profileAction() const { return generatesProfile() || usesProfile(); }
};

/// Which candidates the solve proposes, and how a site is cut into units.
struct CandidateOptions {
#define POSEIDON_CANDIDATE_OPTION POSEIDON_FIELD
#include "Options.def"
};

/// The accuracy target, the cost budget, how the DP searches, and the rewrites
/// applied by hand.
struct SolverOptions {
#define POSEIDON_SOLVER_OPTION POSEIDON_FIELD
#include "Options.def"
};

/// The cost and accuracy model.
struct ModelOptions {
#define POSEIDON_MODEL_OPTION POSEIDON_FIELD
#include "Options.def"
};

/// Herbie and the result cache.
struct HerbieOptions {
#define POSEIDON_HERBIE_OPTION POSEIDON_FIELD
#include "Options.def"
};

/// Matrix raising and host dispatch.
struct MatmulOptions {
#define POSEIDON_MATMUL_OPTION POSEIDON_FIELD
#include "Options.def"
};

/// What the compile prints and writes about its decisions.
struct DiagnosticOptions {
#define POSEIDON_DIAGNOSTIC_OPTION POSEIDON_FIELD
#include "Options.def"
};

#undef POSEIDON_FIELD

/// Every option of the compile, by group; Core/Options.def documents each one.
struct Options {
  ProfilingOptions Profiling;
  CandidateOptions Candidates;
  SolverOptions Solver;
  ModelOptions Model;
  HerbieOptions Herbie;
  MatmulOptions Matmul;
  DiagnosticOptions Diagnostics;
};

/// The options of this compile, read from the -poseidon-* flags.
Options readOptions();

/// The first mistake in the flags' values: a flag outside its choices, or a
/// number flag a solve reads that was given no number or an impossible one. A
/// compile that uses no profile reads no number flag.
llvm::Error validateOptions(const Options &O);

/// An option's flag name, without its leading dash, and its value as text.
using FlagValue = std::pair<llvm::StringRef, std::string>;
/// Flag values, in the order a listing gives them.
using FlagValues = llvm::SmallVector<FlagValue>;

/// The name and value of every option a DP table depends on, that is every
/// option that changes the candidate set or its pricing, in the order the DP
/// key lists them.
FlagValues solveFlagValues(const Options &O);
/// What a solved table does not depend on: what is profiled or which sites are
/// solved, what is chosen from a solved table, a path whose contents the key
/// holds instead (profiles, cost model, Herbie binary and platform), where
/// results computed earlier are kept or handed to the host half, a solve that
/// never reads a table (joint, manual), how long Herbie may run and on how many
/// workers, and output.
FlagValues notSolveInputs(const Options &O);
/// Both, in that order.
FlagValues allFlagValues(const Options &O);

} // namespace poseidon
#endif // POSEIDON_OPTIONS_H

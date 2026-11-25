//===- Options.cpp - what one compile was asked to do ---------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Options.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

namespace poseidon {

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

FlagValues solveFlagValues(const Options &O) {
  FlagValues out;
#define POSEIDON_OPTION(Group, Field, Kind, Default, Spelling, Role, Key, ...) \
  if (OptionKey::Key == OptionKey::SolveInput)                                 \
    out.emplace_back(Spelling, value(O.Group.Field));
#include "Options.def"
  return out;
}

FlagValues notSolveInputs(const Options &O) {
  FlagValues out;
#define POSEIDON_OPTION(Group, Field, Kind, Default, Spelling, Role, Key, ...) \
  if (OptionKey::Key == OptionKey::Other)                                      \
    out.emplace_back(Spelling, value(O.Group.Field));
#include "Options.def"
  return out;
}

FlagValues allFlagValues(const Options &O) {
  FlagValues all = solveFlagValues(O);
  all.append(notSolveInputs(O));
  return all;
}

} // namespace poseidon

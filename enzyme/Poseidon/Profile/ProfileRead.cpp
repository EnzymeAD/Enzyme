//===- ProfileRead.cpp - reading the profile of a site --------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProfileRead.h"
#include "IR/Utils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>

using namespace llvm;

namespace poseidon {

static constexpr double kSensFloorRatio = 1e-6;
static constexpr double kSensNullRatio = 1e-9;

unsigned applySensitivityFloor(ProfileMap &profileMap,
                               llvm::StringRef functionName, bool print) {
  if (profileMap.empty())
    return 0;

  double ref = 0.0;
  for (const auto &kv : profileMap) {
    if (kv.second.exec == 0)
      continue;
    ref = std::max(ref, kv.second.sumSens / (double)kv.second.exec);
  }
  if (!(ref > 0.0) || !std::isfinite(ref))
    return 0;

  unsigned floored = 0;
  for (auto &kv : profileMap) {
    ProfileInfo &p = kv.second;
    if (p.exec == 0)
      continue;
    // A value that is always zero has no sensitivity to floor or report.
    const double mag = std::max(std::fabs(p.minRes), std::fabs(p.maxRes));
    if (!(mag > 0.0) || !std::isfinite(mag))
      continue;
    const double perExec = p.sumSens / (double)p.exec;
    const double ratio = perExec / ref;
    if (print && ratio < kSensNullRatio) {
      llvm::errs()
          << "Poseidon: profiled instruction " << kv.first << " of "
          << functionName << " has per-execution sensitivity " << perExec
          << ", " << ratio
          << " of this site's largest. Either the value is negligible where "
             "it is used, the largest is amplified by a cancellation, or the "
             "adjoint seed lies in the left null space of the operator (a "
             "constant seed does for any partition-of-unity basis with a "
             "derivative contraction); in the last case re-profile with a "
             "non-degenerate seed.\n";
    }
    // The weight multiplies an absolute error, so the gradient a sensitivity
    // floor implies scales with 1 / the value's magnitude.
    const double floorW = kSensFloorRatio * ref / mag * (double)p.exec;
    if (p.weight() < floorW) {
      p.sumGrad = std::copysign(floorW, p.sumGrad == 0.0 ? 1.0 : p.sumGrad);
      p.sumAbsGrad = floorW;
      ++floored;
    }
  }
  if (print && floored)
    llvm::errs() << "Poseidon: sensitivity floor raised the accuracy weight of "
                 << floored << " of " << profileMap.size()
                 << " profiled instruction(s) in " << functionName << " to "
                 << kSensFloorRatio << " x the site maximum sensitivity\n";
  return floored;
}

void mergeOutputProfiles(ProfileMap &profileMap, ArrayRef<std::string> dirs,
                         StringRef fileName, StringRef canonicalHash,
                         const Function &F, bool print) {
  const StringRef functionName = F.getName();
  auto refuse = [&](const Twine &why) {
    reportUserError(F, F.getSubprogram(), why);
  };
  std::unordered_map<size_t, double> grad, sens;
  for (const std::string &dir : dirs) {
    SmallString<128> path(dir);
    sys::path::append(path, fileName);
    ProfileMap out;
    FunctionProfileHeader header;
    parseProfileFile(path.str().str(), F, out, header);
    double total = 0.0;
    for (const auto &kv : out)
      total += kv.second.sumSens;
    if (!(total > 0.0) || !std::isfinite(total))
      refuse(Twine("output profile ") + path + " of " + functionName +
             " carries no sensitivity; re-run profile "
             "generation with that output seeded.");
    if (!canonicalHash.empty() && header.canonicalHash != canonicalHash)
      refuse(Twine("output profile ") + path +
             " was recorded against a different canonical form than the "
             "site's profile; re-run profile generation.");
    if (out.size() != profileMap.size())
      refuse(Twine("output profile ") + path + " records " + Twine(out.size()) +
             " instructions, the site's profile " + Twine(profileMap.size()) +
             "; both must come from the same profiling binary.");
    for (const auto &kv : out) {
      if (!profileMap.count(kv.first))
        refuse(Twine("output profile ") + path + " names instruction " +
               Twine(kv.first) +
               " that the site's profile does not; both must come from the "
               "same profiling binary.");
      const ProfileInfo &p = kv.second;
      grad[kv.first] = std::max(grad[kv.first], p.weight() / total);
      sens[kv.first] = std::max(sens[kv.first], p.sumSens / total);
    }
  }
  for (auto &kv : profileMap) {
    ProfileInfo &p = kv.second;
    p.sumGrad = grad[kv.first];
    p.sumAbsGrad = p.sumGrad;
    p.sumSens = sens[kv.first];
  }
  if (print)
    errs() << "[poseidon] " << functionName
           << ": accuracy weighted per output (" << dirs.size()
           << " output profiles)\n";
}

namespace {

/// One profile file read line by line; every error names the file and the line
/// and stops the compile at the site.
class ProfileParser {
public:
  ProfileParser(const MemoryBuffer &Buffer, StringRef Path, const Function &F)
      : Lines(Buffer, /*SkipBlanks=*/true), Path(Path), F(F) {}

  void parse(ProfileMap &Records, FunctionProfileHeader &Header);

private:
  line_iterator Lines;
  StringRef Path;
  const Function &F;
  int64_t LineNo = 0;

  [[noreturn]] void refuse(const Twine &What) {
    reportUserError(F, F.getSubprogram(),
                    Path + ":" + Twine(LineNo) + ": " + What +
                        "; re-run profile generation.");
  }

  /// The next line, trimmed; none at the end of the file.
  std::optional<StringRef> next() {
    if (Lines.is_at_end())
      return std::nullopt;
    LineNo = Lines.line_number();
    StringRef Line = (*Lines).trim();
    ++Lines;
    return Line;
  }

  template <typename T> T integer(StringRef Text, const Twine &Field) {
    T Out;
    if (Text.getAsInteger(10, Out))
      refuse(Field + " " + Text +
             (all_of(Text, isDigit) && !Text.empty() ? " is out of range"
                                                     : " is not a number"));
    return Out;
  }

  double real(StringRef Text, const Twine &Field) {
    double Out;
    if (Text.getAsDouble(Out))
      refuse(Field + " " + Text + " is not a number");
    return Out;
  }

  template <typename T>
  void integers(StringRef Text, const Twine &Field, MutableArrayRef<T> Out) {
    SmallVector<StringRef, 3> Parts;
    Text.split(Parts, ' ', -1, /*KeepEmpty=*/false);
    if (Parts.size() != Out.size())
      refuse(Field + " " + Text + " is not " + Twine(Out.size()) + " numbers");
    for (auto [Part, To] : zip(Parts, Out))
      To = integer<T>(Part, Field);
  }

  /// The value of the next line, which is the `Key` line of the record of
  /// `Slot`.
  StringRef field(size_t Slot, StringRef Key) {
    std::optional<StringRef> Line = next();
    if (!Line)
      refuse("the record of slot " + Twine(Slot) + " ends before its " + Key +
             " line");
    auto [K, V] = Line->split(" = ");
    if (K != Key)
      refuse("expected the " + Key + " line of slot " + Twine(Slot));
    return V;
  }

  void header(StringRef Key, StringRef Value, FunctionProfileHeader &Header);
  void record(size_t Slot, ProfileInfo &Info);
};

} // namespace

void ProfileParser::header(StringRef Key, StringRef Value,
                           FunctionProfileHeader &Header) {
  if (Key == "MaxBlockDims") {
    integers<uint32_t>(Value, Key, Header.maxBlockDim);
  } else if (Key == "MaxGridDims") {
    integers<uint32_t>(Value, Key, Header.maxGridDim);
  } else if (Key == "LaunchCount") {
    Header.launchCount = integer<uint64_t>(Value, Key);
  } else if (Key == "RedTrip") {
    size_t Trip[2];
    integers<size_t>(Value, Key, Trip);
    if (Trip[1] > std::numeric_limits<unsigned>::max())
      refuse("RedTrip trip count " + Twine(Trip[1]) + " is out of range");
    Header.redTrip[Trip[0]] = Trip[1];
  } else if (Key == kProfileKeyCanonicalHash) {
    if (Value.empty() || !all_of(Value, isHexDigit))
      refuse(Twine(Key) + " " + Value + " is not a hexadecimal digest");
    Header.canonicalHash = Value.str();
  } else if (Key == kProfileKeySiteId) {
    // The probe driver's; checked so that the driver reads it back.
    (void)integer<int>(Value, Key);
  } else if (Key == kProfileKeyKappa) {
    Header.kappa = real(Value, Key);
    Header.hasKappa = true;
  } else if (Key == kProfileKeyKappaMark) {
    if (Value.empty() || Value.contains(' '))
      refuse(Twine(Key) + " " + Value + " is not one word");
    Header.kappaMark = Value.str();
  } else if (Key == kProfileKeyExecuted) {
    if (Value != "0")
      refuse(Twine(Key) + " " + Value + " is not 0");
    Header.executed = false;
  } else {
    refuse("'" + Key + "' is not a header key of a profile");
  }
}

void ProfileParser::record(size_t Slot, ProfileInfo &Info) {
  Info.minRes = real(field(Slot, "MinRes"), "MinRes");
  Info.maxRes = real(field(Slot, "MaxRes"), "MaxRes");
  Info.sumSens = real(field(Slot, "SumSens"), "SumSens");
  Info.sumGrad = real(field(Slot, "SumGrad"), "SumGrad");
  Info.sumAbsGrad = real(field(Slot, "SumAbsGrad"), "SumAbsGrad");
  Info.exec = integer<uint64_t>(field(Slot, "Exec"), "Exec");
  unsigned NumOperands =
      integer<unsigned>(field(Slot, "NumOperands"), "NumOperands");
  // Read before the vectors are sized, so that a count the file does not hold
  // is refused rather than allocated.
  SmallVector<std::array<double, 3>, 2> Operands;
  for (unsigned I = 0; I < NumOperands; ++I) {
    SmallString<16> Key;
    (Twine("Operand[") + Twine(I) + "]").toVector(Key);
    StringRef Value = field(Slot, Key);
    SmallVector<StringRef, 3> Parts;
    if (!Value.consume_front("[") || !Value.consume_back("]") ||
        (Value.split(Parts, ", "), Parts.size() != 3))
      refuse(Key + " of slot " + Twine(Slot) +
             " is not [minimum, maximum, smallest nonzero magnitude]");
    std::array<double, 3> &Range = Operands.emplace_back();
    for (unsigned J = 0; J < 3; ++J)
      Range[J] = real(Parts[J], Key);
  }
  for (const std::array<double, 3> &Range : Operands) {
    Info.minOperands.push_back(Range[0]);
    Info.maxOperands.push_back(Range[1]);
    Info.minMagOperands.push_back(Range[2]);
  }
}

void ProfileParser::parse(ProfileMap &Records, FunctionProfileHeader &Header) {
  while (std::optional<StringRef> Line = next()) {
    if (all_of(*Line, isDigit)) {
      if (!Header.executed)
        refuse("the profile says its site was never executed and records "
               "executions of it");
      size_t Slot = integer<size_t>(*Line, "slot");
      if (!Records.emplace(Slot, ProfileInfo()).second)
        refuse("slot " + Twine(Slot) + " is recorded twice");
      record(Slot, Records[Slot]);
      continue;
    }
    if (!Records.empty())
      refuse("'" + *Line + "' follows the records");
    auto [Key, Value] = Line->split(" = ");
    if (Key == *Line)
      refuse("'" + *Line + "' is neither a slot index nor 'key = value'");
    header(Key, Value, Header);
  }
}

void parseProfileFile(const std::string &profilePath, const Function &F,
                      ProfileMap &profileMap, FunctionProfileHeader &header) {
  profileMap.clear();
  header = FunctionProfileHeader{};
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFile(profilePath, /*IsText=*/true);
  if (!Buffer)
    reportUserError(F, F.getSubprogram(),
                    "cannot read the profile " + profilePath + " (" +
                        Buffer.getError().message() +
                        "); run the program built with "
                        "-poseidon-profile-generate, which writes it into "
                        "POSEIDON_PROFILE_DIR (default " +
                        POSEIDON_DEFAULT_PROFILE_DIR + ")");
  ProfileParser(**Buffer, profilePath, F).parse(profileMap, header);
}

void requireCanonicalHash(const FunctionProfileHeader &header,
                          StringRef canonicalHash, StringRef profilePath,
                          const Function &F) {
  if (header.canonicalHash != canonicalHash)
    reportUserError(
        F, F.getSubprogram(),
        Twine("the profile of ") + F.getName() +
            " was recorded against a different canonical form "
            "(profile CanonicalHash " +
            (header.canonicalHash.empty() ? "none" : header.canonicalHash) +
            ", this compile " + canonicalHash +
            "). Its slot indices do not name the same "
            "instructions; re-run profile generation.");
}

void requireMeasuredKappa(const FunctionProfileHeader &header,
                          StringRef profilePath, const Function &F) {
  if (header.hasKappa || header.kappaMark != kKappaUnmeasured)
    return;
  reportUserError(
      F, F.getSubprogram(),
      Twine("the profiling run declared a quantity of interest with "
            "poseidon_metric and did not measure the condition number of ") +
          F.getName() + " against it (" + profilePath +
          "). Run the profiling build again as 'poseidon-probe -- <program> "
          "[args...]', the command the profiling run printed, to measure it, "
          "or with POSEIDON_PROBE=0 in its environment to solve each site "
          "against its own target.");
}

} // namespace poseidon

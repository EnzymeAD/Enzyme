//===- ProfileReader.cpp - reading the profile of a site ------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProfileReader.h"
#include "Support/Diagnostics.h"
#include "Support/Utils.h"
#include "runtime/profile/ProfileFormat.h"

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
          << "[poseidon] profiled instruction " << kv.first << " of "
          << functionName << " has per-execution sensitivity " << perExec
          << ", " << ratio
          << " of this site's largest; if the adjoint seed lies in the left "
             "null space of the operator, re-profile with a non-degenerate "
             "seed\n";
    }
    // The weight multiplies an absolute error, so the gradient a sensitivity
    // floor implies scales with 1 / the value's magnitude.
    const double floorW = kSensFloorRatio * ref / mag * (double)p.exec;
    if (p.sumAbsGrad < floorW) {
      p.sumGrad = std::copysign(floorW, p.sumGrad == 0.0 ? 1.0 : p.sumGrad);
      p.sumAbsGrad = floorW;
      ++floored;
    }
  }
  if (print && floored)
    llvm::errs()
        << "[poseidon] sensitivity floor raised the accuracy weight of "
        << floored << " of " << profileMap.size()
        << " profiled instruction(s) in " << functionName << " to "
        << kSensFloorRatio << " x the site maximum sensitivity\n";
  return floored;
}

Error mergeOutputProfiles(ProfileMap &profileMap, ArrayRef<std::string> dirs,
                          StringRef fileName, StringRef canonicalHash,
                          const Function &F, bool print) {
  const StringRef functionName = F.getName();
  auto refusal = [&](const Twine &why) {
    return userError(F, F.getSubprogram(), why);
  };
  std::unordered_map<size_t, double> grad, sens;
  for (const std::string &dir : dirs) {
    SmallString<128> path(dir);
    sys::path::append(path, fileName);
    ProfileMap out;
    SiteProfileHeader header;
    if (Error E = parseProfileFile(path.str().str(), F, out, header))
      return E;
    double total = 0.0;
    for (const auto &kv : out)
      total += kv.second.sumSens;
    if (!(total > 0.0) || !std::isfinite(total))
      return refusal(Twine("output profile ") + path + " of " + functionName +
                     " carries no sensitivity; re-run profile "
                     "generation with that output seeded.");
    if (!canonicalHash.empty() && header.canonicalHash != canonicalHash)
      return refusal(Twine("output profile ") + path +
                     " was recorded against a different canonical form than "
                     "the site's profile; re-run profile generation.");
    if (out.size() != profileMap.size())
      return refusal(Twine("output profile ") + path + " records " +
                     Twine(out.size()) + " instructions, the site's profile " +
                     Twine(profileMap.size()) +
                     "; both must come from the same profiling binary.");
    for (const auto &kv : out) {
      if (!profileMap.count(kv.first))
        return refusal(Twine("output profile ") + path + " names instruction " +
                       Twine(kv.first) +
                       " that the site's profile does not; both must come "
                       "from the same profiling binary.");
      const ProfileInfo &p = kv.second;
      grad[kv.first] = std::max(grad[kv.first], p.sumAbsGrad / total);
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
  return Error::success();
}

namespace {

/// One profile file read line by line; every error names the file and the line
/// and stops the compile at the site.
class ProfileParser {
public:
  ProfileParser(const MemoryBuffer &Buffer, StringRef Path, const Function &F)
      : Lines(Buffer, /*SkipBlanks=*/true), Path(Path), F(F) {}

  Error parse(ProfileMap &Records, SiteProfileHeader &Header);

private:
  line_iterator Lines;
  StringRef Path;
  const Function &F;
  int64_t LineNo = 0;

  Error refusal(const Twine &What) const {
    return userError(F, F.getSubprogram(),
                     Path + ":" + Twine(LineNo) + ": " + What +
                         "; re-run profile generation");
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

  template <typename T>
  Expected<T> integer(StringRef Text, const Twine &Field) {
    T Out;
    if (Text.getAsInteger(10, Out))
      return refusal(Field + " " + Text +
                     (all_of(Text, isDigit) && !Text.empty()
                          ? " is out of range"
                          : " is not a number"));
    return Out;
  }

  Expected<double> real(StringRef Text, const Twine &Field) {
    double Out;
    if (Text.getAsDouble(Out))
      return refusal(Field + " " + Text + " is not a number");
    return Out;
  }

  template <typename T>
  Error integers(StringRef Text, const Twine &Field, MutableArrayRef<T> Out) {
    SmallVector<StringRef, 3> Parts;
    Text.split(Parts, ' ', -1, /*KeepEmpty=*/false);
    if (Parts.size() != Out.size())
      return refusal(Field + " " + Text + " is not " + Twine(Out.size()) +
                     " numbers");
    for (auto [Part, To] : zip(Parts, Out)) {
      Expected<T> Value = integer<T>(Part, Field);
      if (!Value)
        return Value.takeError();
      To = *Value;
    }
    return Error::success();
  }

  /// The value of the next line, which is the `Key` line of the record of
  /// `Slot`.
  Expected<StringRef> field(size_t Slot, StringRef Key) {
    std::optional<StringRef> Line = next();
    if (!Line)
      return refusal("the record of slot " + Twine(Slot) + " ends before its " +
                     Key + " line");
    auto [K, V] = Line->split(" = ");
    if (K != Key)
      return refusal("expected the " + Key + " line of slot " + Twine(Slot));
    return V;
  }

  /// The number on the `Key` line of the record of `Slot`, into `Out`.
  template <typename T> Error number(size_t Slot, StringRef Key, T &Out) {
    Expected<StringRef> Text = field(Slot, Key);
    if (!Text)
      return Text.takeError();
    Expected<T> Value = [&]() -> Expected<T> {
      if constexpr (std::is_floating_point_v<T>)
        return real(*Text, Key);
      else
        return integer<T>(*Text, Key);
    }();
    if (!Value)
      return Value.takeError();
    Out = *Value;
    return Error::success();
  }

  Error header(StringRef Key, StringRef Value, SiteProfileHeader &Header);
  Error record(size_t Slot, ProfileInfo &Info);
};

} // namespace

Error ProfileParser::header(StringRef Key, StringRef Value,
                            SiteProfileHeader &Header) {
  if (Key == "MaxBlockDims")
    return integers<uint32_t>(Value, Key, Header.maxBlockDim);
  if (Key == "MaxGridDims")
    return integers<uint32_t>(Value, Key, Header.maxGridDim);
  if (Key == "LaunchCount") {
    Expected<uint64_t> Count = integer<uint64_t>(Value, Key);
    if (!Count)
      return Count.takeError();
    Header.launchCount = *Count;
  } else if (Key == "ReductionTrip") {
    size_t Trip[2];
    if (Error E = integers<size_t>(Value, Key, Trip))
      return E;
    if (Trip[1] > std::numeric_limits<unsigned>::max())
      return refusal("ReductionTrip trip count " + Twine(Trip[1]) +
                     " is out of range");
    Header.redTrip[Trip[0]] = Trip[1];
  } else if (Key == kProfileKeyCanonicalHash) {
    if (Value.empty() || !all_of(Value, isHexDigit))
      return refusal(Twine(Key) + " " + Value + " is not a hexadecimal digest");
    Header.canonicalHash = Value.str();
  } else if (Key == kProfileKeySiteId) {
    // poseidon-kappa's; checked so that poseidon-kappa reads it back.
    if (Expected<int> Id = integer<int>(Value, Key); !Id)
      return Id.takeError();
  } else if (Key == kProfileKeyKappa) {
    Expected<double> Kappa = real(Value, Key);
    if (!Kappa)
      return Kappa.takeError();
    Header.kappa = *Kappa;
    Header.hasKappa = true;
  } else if (Key == kProfileKeyKappaMark) {
    Header.kappaMark = StringSwitch<KappaMark>(Value)
                           .Case(kProfileKappaOk, KappaMark::Ok)
                           .Case(kProfileKappaBelowNoise, KappaMark::BelowNoise)
                           .Case(kProfileKappaNonlinear, KappaMark::Nonlinear)
                           .Case(kProfileKappaDiverged, KappaMark::Diverged)
                           .Case(kProfileKappaUnmeasured, KappaMark::Unmeasured)
                           .Default(KappaMark::None);
    if (Header.kappaMark == KappaMark::None)
      return refusal(Twine(Key) + " " + Value +
                     " is not ok, below-noise, nonlinear, diverged or " +
                     kProfileKappaUnmeasured);
  } else if (Key == kProfileKeyExecuted) {
    if (Value != "0")
      return refusal(Twine(Key) + " " + Value + " is not 0");
    Header.executed = false;
  } else {
    return refusal("'" + Key + "' is not a header key of a profile");
  }
  return Error::success();
}

Error ProfileParser::record(size_t Slot, ProfileInfo &Info) {
  for (auto [Key, Out] :
       {std::pair<StringRef, double *>{"MinRes", &Info.minRes},
        {"MaxRes", &Info.maxRes},
        {"SumSens", &Info.sumSens},
        {"SumGrad", &Info.sumGrad},
        {"SumAbsGrad", &Info.sumAbsGrad}})
    if (Error E = number(Slot, Key, *Out))
      return E;
  if (Error E = number(Slot, "Exec", Info.exec))
    return E;
  unsigned NumOperands = 0;
  if (Error E = number(Slot, "NumOperands", NumOperands))
    return E;
  // Read before the vectors are sized, so that a count the file does not hold
  // is refused rather than allocated.
  SmallVector<std::array<double, 3>, 2> Operands;
  for (unsigned I = 0; I < NumOperands; ++I) {
    SmallString<16> Key;
    (Twine("Operand[") + Twine(I) + "]").toVector(Key);
    Expected<StringRef> Value = field(Slot, Key);
    if (!Value)
      return Value.takeError();
    SmallVector<StringRef, 3> Parts;
    if (!Value->consume_front("[") || !Value->consume_back("]") ||
        (Value->split(Parts, ", "), Parts.size() != 3))
      return refusal(Key + " of slot " + Twine(Slot) +
                     " is not [minimum, maximum, smallest nonzero magnitude]");
    std::array<double, 3> &Range = Operands.emplace_back();
    for (unsigned J = 0; J < 3; ++J) {
      Expected<double> Bound = real(Parts[J], Key);
      if (!Bound)
        return Bound.takeError();
      Range[J] = *Bound;
    }
  }
  for (const std::array<double, 3> &Range : Operands) {
    Info.minOperands.push_back(Range[0]);
    Info.maxOperands.push_back(Range[1]);
    Info.minMagOperands.push_back(Range[2]);
  }
  return Error::success();
}

Error ProfileParser::parse(ProfileMap &Records, SiteProfileHeader &Header) {
  while (std::optional<StringRef> Line = next()) {
    if (all_of(*Line, isDigit)) {
      if (!Header.executed)
        return refusal("the profile says its site was never executed and "
                       "records executions of it");
      Expected<size_t> Slot = integer<size_t>(*Line, "slot");
      if (!Slot)
        return Slot.takeError();
      if (!Records.emplace(*Slot, ProfileInfo()).second)
        return refusal("slot " + Twine(*Slot) + " is recorded twice");
      if (Error E = record(*Slot, Records[*Slot]))
        return E;
      continue;
    }
    if (!Records.empty())
      return refusal("'" + *Line + "' follows the records");
    auto [Key, Value] = Line->split(" = ");
    if (Key == *Line)
      return refusal("'" + *Line +
                     "' is neither a slot index nor 'key = value'");
    if (Error E = header(Key, Value, Header))
      return E;
  }
  return Error::success();
}

Error parseProfileFile(const std::string &profilePath, const Function &F,
                       ProfileMap &profileMap, SiteProfileHeader &header) {
  profileMap.clear();
  header = SiteProfileHeader{};
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      MemoryBuffer::getFile(profilePath, /*IsText=*/true);
  if (!Buffer)
    return userError(F, F.getSubprogram(),
                     "cannot read the profile " + profilePath + " (" +
                         Buffer.getError().message() +
                         "); run the program built with "
                         "-poseidon-profile-generate, which writes it into "
                         "POSEIDON_PROFILE_DIR (default " +
                         POSEIDON_DEFAULT_PROFILE_DIR + ")");
  return ProfileParser(**Buffer, profilePath, F).parse(profileMap, header);
}

Error requireCanonicalHash(const SiteProfileHeader &header,
                           StringRef canonicalHash, StringRef profilePath,
                           const Function &F) {
  if (header.canonicalHash == canonicalHash)
    return Error::success();
  return userError(
      F, F.getSubprogram(),
      Twine("the profile of ") + F.getName() +
          " was recorded against a different canonical form "
          "(profile CanonicalHash " +
          (header.canonicalHash.empty() ? "none" : header.canonicalHash) +
          ", this compile " + canonicalHash + "); re-run profile generation");
}

StringRef kappaMarkName(KappaMark M) {
  switch (M) {
  case KappaMark::None:
    return "";
  case KappaMark::Ok:
    return kProfileKappaOk;
  case KappaMark::BelowNoise:
    return kProfileKappaBelowNoise;
  case KappaMark::Nonlinear:
    return kProfileKappaNonlinear;
  case KappaMark::Diverged:
    return kProfileKappaDiverged;
  case KappaMark::Unmeasured:
    return kProfileKappaUnmeasured;
  }
  llvm_unreachable("every KappaMark has a word");
}

Error requireMeasuredKappa(const SiteProfileHeader &header,
                           StringRef profilePath, const Function &F) {
  if (header.hasKappa || header.kappaMark != KappaMark::Unmeasured)
    return Error::success();
  return userError(
      F, F.getSubprogram(),
      Twine("the profiling run declared a metric with poseidon_metric and "
            "did not measure the condition number of ") +
          F.getName() + " against it (" + profilePath +
          "); run the profiling build again under 'poseidon-kappa -- "
          "<program> [args...]', or with POSEIDON_KAPPA=0 to solve each site "
          "against its own target");
}

} // namespace poseidon

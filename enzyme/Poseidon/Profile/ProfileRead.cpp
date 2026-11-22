//===- ProfileRead.cpp - Profiling utilities for Poseidon -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements profiling-related utilities for the Poseidon
// optimization pass.
//
//===----------------------------------------------------------------------===//

#include "ProfileRead.h"
#include "IR/Utils.h"
#include "Instrument.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <system_error>
#include <unordered_map>

using namespace llvm;

namespace poseidon {

bool tryReadProfIdxMetadata(const Instruction *I, size_t &out) {
  if (!I)
    return false;
  MDNode *md = I->getMetadata("poseidon.prof.idx");
  if (!md)
    return false;
  out = cast<ConstantInt>(
            cast<ConstantAsMetadata>(md->getOperand(0).get())->getValue())
            ->getZExtValue();
  return true;
}

size_t readProfIdxMetadata(const Instruction *I) {
  if (!I)
    report_fatal_error("readProfIdxMetadata: null instruction");
  size_t out;
  if (!tryReadProfIdxMetadata(I, out))
    report_fatal_error("instruction missing !poseidon.prof.idx metadata "
                       "(forgot to run profgen?)");
  return out;
}

static constexpr double kSensFloorRatio = 1e-6;
static constexpr double kSensNullRatio = 1e-9;

unsigned applySensitivityFloor(ProfileMap &profileMap,
                               llvm::StringRef functionName) {
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
    if (ratio < kSensNullRatio) {
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
      if (p.sumAbsGrad >= 0.0)
        p.sumAbsGrad = floorW;
      ++floored;
    }
  }
  if (floored)
    llvm::errs() << "Poseidon: sensitivity floor raised the accuracy weight of "
                 << floored << " of " << profileMap.size()
                 << " profiled instruction(s) in " << functionName << " to "
                 << kSensFloorRatio << " x the site maximum sensitivity\n";
  return floored;
}

void mergeOutputProfiles(ProfileMap &profileMap, ArrayRef<std::string> dirs,
                         StringRef fileName, StringRef canonicalHash,
                         const Function &F) {
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
    parseProfileFile(path.str().str(), F, out, &header);
    double total = 0.0;
    for (const auto &kv : out)
      total += kv.second.sumSens;
    if (!(total > 0.0) || !std::isfinite(total))
      refuse(Twine("output profile ") + path + " of " + functionName +
             " is missing or carries no sensitivity; re-run profile "
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
    if (p.sumAbsGrad >= 0.0)
      p.sumAbsGrad = p.sumGrad;
    p.sumSens = sens[kv.first];
  }
  errs() << "[poseidon] " << functionName << ": accuracy weighted per output ("
         << dirs.size() << " output profiles)\n";
}

void parseProfileFile(const std::string &profilePath, const Function &F,
                      ProfileMap &profileMap, FunctionProfileHeader *header) {
  profileMap.clear();
  if (header)
    *header = FunctionProfileHeader{};
  std::ifstream file(profilePath);
  if (!file.is_open()) {
    llvm::errs() << "Warning: Could not open profile file: " << profilePath
                 << '\n';
    return;
  }

  std::regex indexPattern(R"(^(\d+)$)");
  std::regex blockDimsPattern(
      R"(^\s*MaxBlockDims\s*=\s*(\d+)\s+(\d+)\s+(\d+))");
  std::regex gridDimsPattern(R"(^\s*MaxGridDims\s*=\s*(\d+)\s+(\d+)\s+(\d+))");
  std::regex launchCountPattern(R"(^\s*LaunchCount\s*=\s*(\d+))");
  std::regex redTripPattern(R"(^\s*RedTrip\s*=\s*(\d+)\s+(\d+))");
  std::regex canonicalHashPattern(R"(^\s*CanonicalHash\s*=\s*([0-9a-fA-F]+))");
  std::regex siteIdPattern(R"(^\s*SiteId\s*=\s*(\d+))");
  static const std::string kReal =
      R"(([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))";
  std::regex kappaPattern(R"(^\s*Kappa\s*=\s*)" + kReal);
  std::regex kappaMarkPattern(R"(^\s*KappaMark\s*=\s*(\S+))");

  std::regex minResPattern(R"(^\s*MinRes\s*=\s*)" + kReal);
  std::regex maxResPattern(R"(^\s*MaxRes\s*=\s*)" + kReal);
  std::regex sumValuePattern(R"(^\s*SumValue\s*=\s*)" + kReal);
  std::regex sumSensPattern(R"(^\s*SumSens\s*=\s*)" + kReal);
  std::regex sumGradPattern(R"(^\s*SumGrad\s*=\s*)" + kReal);
  // Optional line.
  std::regex sumAbsGradPattern(R"(^\s*SumAbsGrad\s*=\s*)" + kReal);
  std::regex execPattern(R"(^\s*Exec\s*=\s*(\d+))");
  std::regex numOperandsPattern(R"(^\s*NumOperands\s*=\s*(\d+))");
  std::regex operandPattern(R"(^\s*Operand\[(\d+)\]\s*=\s*\[)" + kReal +
                            R"(,\s*)" + kReal + R"((?:,\s*)" + kReal +
                            R"()?\])");

  unsigned lineNo = 0;
  auto nextLine = [&](std::string &text) {
    if (!std::getline(file, text))
      return false;
    ++lineNo;
    if (!text.empty() && text.back() == '\r')
      text.pop_back();
    return true;
  };
  auto refuse = [&](const Twine &what) {
    reportUserError(F, F.getSubprogram(),
                    Twine(profilePath) + ":" + Twine(lineNo) + ": " + what +
                        "; re-run profile generation.");
  };
  // A decimal number of the profile, which must fit the field it is read into.
  auto integer = [&](auto &out, const std::string &digits, StringRef field) {
    if (StringRef(digits).getAsInteger(10, out))
      refuse(Twine(field) + " " + digits + " is out of range");
  };

  std::string line;
  while (nextLine(line)) {
    std::smatch match;
    if (header && std::regex_search(line, match, blockDimsPattern)) {
      for (unsigned i = 0; i < 3; ++i)
        integer(header->maxBlockDim[i], match[i + 1], "MaxBlockDims");
      continue;
    }
    if (header && std::regex_search(line, match, gridDimsPattern)) {
      for (unsigned i = 0; i < 3; ++i)
        integer(header->maxGridDim[i], match[i + 1], "MaxGridDims");
      continue;
    }
    if (header && std::regex_search(line, match, launchCountPattern)) {
      integer(header->launchCount, match[1], "LaunchCount");
      continue;
    }
    if (header && std::regex_search(line, match, redTripPattern)) {
      size_t slot;
      integer(slot, match[1], "RedTrip slot");
      integer(header->redTrip[slot], match[2], "RedTrip trip count");
      continue;
    }
    if (header && std::regex_search(line, match, canonicalHashPattern)) {
      header->canonicalHash = match[1];
      continue;
    }
    // The site id is the probe driver's; parsing it still refuses one out of
    // int range.
    if (header && std::regex_search(line, match, siteIdPattern)) {
      int siteId;
      integer(siteId, match[1], "SiteId");
      continue;
    }
    if (header && std::regex_search(line, match, kappaPattern)) {
      errno = 0;
      header->kappa = std::strtod(match[1].str().c_str(), nullptr);
      if (errno == ERANGE)
        refuse("Kappa " + match[1].str() + " is out of range");
      header->hasKappa = true;
      continue;
    }
    if (header && std::regex_search(line, match, kappaMarkPattern)) {
      header->kappaMark = match[1];
      continue;
    }
    if (!std::regex_match(line, match, indexPattern))
      continue;

    size_t idx;
    integer(idx, match[1], "slot");
    // The next line of the record of slot idx, which `pattern` must match.
    auto field = [&](std::string &text, std::smatch &fieldMatch,
                     const std::regex &pattern, StringRef name) {
      if (!nextLine(text))
        refuse("the record of slot " + Twine(idx) + " ends before its " + name +
               " line");
      if (!std::regex_search(text, fieldMatch, pattern))
        refuse("expected the " + name + " line of slot " + Twine(idx));
    };
    auto real = [&](const std::regex &pattern, StringRef name) {
      std::string text;
      std::smatch fieldMatch;
      field(text, fieldMatch, pattern, name);
      return stringToDouble(fieldMatch[1]);
    };
    ProfileInfo info;
    info.minRes = real(minResPattern, "MinRes");
    info.maxRes = real(maxResPattern, "MaxRes");
    info.sumValue = real(sumValuePattern, "SumValue");
    info.sumSens = real(sumSensPattern, "SumSens");
    info.sumGrad = real(sumGradPattern, "SumGrad");
    // SumAbsGrad comes before Exec when the profile records it.
    std::string text;
    std::smatch fieldMatch;
    if (!nextLine(text))
      refuse("the record of slot " + Twine(idx) + " ends before its Exec line");
    if (text.find("SumAbsGrad") != std::string::npos) {
      if (!std::regex_search(text, fieldMatch, sumAbsGradPattern))
        refuse("expected the SumAbsGrad line of slot " + Twine(idx));
      info.sumAbsGrad = stringToDouble(fieldMatch[1]);
      field(text, fieldMatch, execPattern, "Exec");
    } else if (!std::regex_search(text, fieldMatch, execPattern)) {
      refuse("expected the Exec line of slot " + Twine(idx));
    }
    integer(info.exec, fieldMatch[1], "Exec");
    unsigned numOperands;
    field(text, fieldMatch, numOperandsPattern, "NumOperands");
    integer(numOperands, fieldMatch[1], "NumOperands");

    // The operands are read before the profile's vectors are sized, so that a
    // count the file does not hold is refused rather than allocated.
    struct OperandRange {
      unsigned index;
      double min, max, minMag;
    };
    SmallVector<OperandRange, 2> operands;
    for (unsigned i = 0; i < numOperands; ++i) {
      field(text, fieldMatch, operandPattern, "Operand");
      OperandRange &R = operands.emplace_back();
      integer(R.index, fieldMatch[1], "Operand index");
      if (R.index >= numOperands)
        refuse("Operand[" + Twine(R.index) + "] of slot " + Twine(idx) +
               " is beyond its NumOperands " + Twine(numOperands));
      R.min = stringToDouble(fieldMatch[2]);
      R.max = stringToDouble(fieldMatch[3]);
      R.minMag = fieldMatch[4].matched ? stringToDouble(fieldMatch[4]) : 0.0;
    }
    info.minOperands.resize(numOperands, 0.0);
    info.maxOperands.resize(numOperands, 0.0);
    info.minMagOperands.resize(numOperands, 0.0);
    for (const OperandRange &R : operands) {
      info.minOperands[R.index] = R.min;
      info.maxOperands[R.index] = R.max;
      info.minMagOperands[R.index] = R.minMag;
    }

    profileMap[idx] = info;
  }
}

bool isNeverExecutedProfile(const std::string &profilePath, const Function &F) {
  std::ifstream file(profilePath);
  std::string line;
  bool marked = false;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (!line.empty() &&
        line.find_first_not_of("0123456789") == std::string::npos) {
      if (marked)
        reportUserError(F, F.getSubprogram(),
                        Twine(profilePath) +
                            " says its site was never executed and records "
                            "executions of it; re-run profile generation.");
      return false;
    }
    marked |= line == "Executed = 0";
  }
  return marked;
}

bool profileDirNamesFormat(StringMap<bool> &known, StringRef dir) {
  auto [it, inserted] = known.try_emplace(dir, false);
  if (!inserted)
    return it->second;
  std::error_code EC;
  for (sys::fs::directory_iterator entry(dir, EC), end; !EC && entry != end;
       entry.increment(EC)) {
    if (sys::path::extension(entry->path()) != ".fpprofile")
      continue;
    std::ifstream file(entry->path());
    std::string line;
    while (std::getline(file, line)) {
      StringRef rest = StringRef(line).trim();
      if (!rest.empty() &&
          rest.find_first_not_of("0123456789") == StringRef::npos)
        break;
      if (rest.consume_front(kProfileFormatKey) &&
          rest.ltrim().starts_with("="))
        return it->second = true;
    }
  }
  return false;
}

void requireCanonicalHash(const FunctionProfileHeader &header,
                          StringRef canonicalHash, StringRef profilePath,
                          const Function &F) {
  if (header.canonicalHash.empty())
    reportUserError(F, F.getSubprogram(),
                    Twine(profilePath) +
                        " has no CanonicalHash field, so its slot indices "
                        "cannot be checked against the canonical form of " +
                        F.getName() + "; re-run profile generation.");
  if (header.canonicalHash != canonicalHash)
    reportUserError(F, F.getSubprogram(),
                    Twine("the profile of ") + F.getName() +
                        " was recorded against a different canonical form "
                        "(profile CanonicalHash " +
                        header.canonicalHash + ", this compile " +
                        canonicalHash +
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

// The generated logging code declares the probes on first use without
// enzyme_inactive, and with only a declaration in the TU Enzyme demands a
// derivative ("No augmented forward pass found"); so they are declared first,
// with signatures that match the generated ones exactly. The runtime's own
// definitions cannot carry the attribute: __has_attribute(enzyme_inactive) is
// false unless the clang plugin is loaded, so under -fpass-plugin it is lost.
void predeclareInactiveProfilerProbes(llvm::Module &M) {
  using namespace llvm;
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return;
  const ProbeABI &P = T.probes();
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *SizeTy = Type::getInt64Ty(Ctx);
  getProfilerHook(
      M, P.LogValue,
      FunctionType::get(VoidTy,
                        {PtrTy, SizeTy, DoubleTy,
                         IntegerType::get(Ctx, P.OperandCountBits), PtrTy},
                        false));
  getProfilerHook(
      M, P.LogGradient,
      FunctionType::get(VoidTy, {PtrTy, SizeTy, DoubleTy, DoubleTy}, false));
  // The perturbation hooks are inactive too: nothing reads a derivative of the
  // perturbation, and a second run of the pass would otherwise differentiate
  // the calls the first run injected.
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  getProfilerHook(M, P.PerturbF64,
                  FunctionType::get(DoubleTy, {Int32Ty, DoubleTy}, false));
  getProfilerHook(M, P.PerturbF32,
                  FunctionType::get(FloatTy, {Int32Ty, FloatTy}, false));
}

// A static constructor cannot make the call: the runtime publishes its device
// tables with cudaMemcpyToSymbol, and a dynamic initializer can run before
// __cudaRegisterFatBinary has registered the device module in a multi-TU
// binary ("invalid device symbol"). main runs after every static constructor.
//
// Only the host half of an offload unit, kernels or not, links the device
// profiler. Erring towards injection is deliberate: a wrongly skipped
// injection loses the profile silently, a wrongly kept one fails at link.
bool injectHostProfilerInit(llvm::Module &M) {
  using namespace llvm;
  if (Target::of(M).isDevice())
    return false;
  // A host-only build links no CUDA profiler runtime, so the call injected
  // below would be an undefined reference to __poseidon_profile_init_cuda.
  OffloadRuntime R = offloadRuntimeOf(M);
  if (R == OffloadRuntime::None)
    return false;
  Function *mainF = M.getFunction("main");
  if (!mainF || mainF->isDeclaration())
    return false;
  LLVMContext &Ctx = M.getContext();
  FunctionCallee initFn =
      getProfilerHook(M, deviceProfilerInit(),
                      FunctionType::get(Type::getVoidTy(Ctx), {}, false));
  // Idempotence: never place a second call.
  for (Instruction &I : instructions(*mainF))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (auto *CF = CI->getCalledFunction())
        if (CF->getName() == deviceProfilerInit())
          return false;
  IRBuilder<> B(&*mainF->getEntryBlock().getFirstInsertionPt());
  B.CreateCall(initFn, {})->setDoesNotThrow();
  llvm::errs() << "[poseidon] FP profiler CUDA registration injected at main\n";
  return true;
}

// The probe is keyed by the poseidon_site_<name> global the value probes use,
// which exists only after AD.
void injectBlockDimProbes(llvm::Module &M) {
  using namespace llvm;
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return;
  const ProbeABI &P = T.probes();
  Function *logFn = M.getFunction(P.LogValue);
  if (!logFn)
    return;
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = PointerType::getUnqual(Ctx);
  FunctionCallee probeFn = getProfilerHook(
      M, P.BlockDims, FunctionType::get(Type::getVoidTy(Ctx), {PtrTy}, false));

  unsigned placed = 0;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Value *nameArg = nullptr;
    bool already = false;
    for (Instruction &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      if (CI->getCalledOperand()->stripPointerCasts() == logFn) {
        if (!nameArg)
          nameArg = CI->getArgOperand(0);
        continue;
      }
      if (auto *CF = CI->getCalledFunction())
        if (CF->getName() == P.BlockDims)
          already = true; // idempotence
    }
    if (!nameArg || already)
      continue;
    // Rebuild the name pointer at the entry block from the global (the log
    // call's operand may not dominate the entry); same construction as the log
    // probes, so the runtime sees the same registry slot.
    auto *gv = dyn_cast<GlobalVariable>(nameArg->stripPointerCasts());
    if (!gv) {
      llvm::errs() << "[poseidon] block-geometry probe: value probe in "
                   << F.getName()
                   << " has a non-global name operand; cannot key the launch "
                      "geometry to this function\n";
      report_fatal_error("Poseidon: unkeyable poseidon_site_ name operand");
    }
    IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
    Value *nameAtEntry = B.CreateInBoundsGEP(gv->getValueType(), gv,
                                             {B.getInt32(0), B.getInt32(0)});
    if (nameAtEntry->getType() != PtrTy)
      nameAtEntry = B.CreateAddrSpaceCast(nameAtEntry, PtrTy);
    CallInst *call = B.CreateCall(probeFn, {nameAtEntry});
    call->setDoesNotThrow();
    ++placed;
  }
  if (placed)
    llvm::errs() << "[poseidon] block-geometry probe placed in " << placed
                 << " profiled function(s)\n";
}

} // namespace poseidon

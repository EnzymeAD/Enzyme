//=- ProfileRead.cpp - Profiling utilities for Poseidon --------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements profiling-related utilities for the Poseidon
// optimization pass.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cmath>
#include <fstream>
#include <regex>

#include "ProfileRead.h"
#include "Utils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TargetParser/Triple.h"

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

static constexpr double kGradFloorRatio = 1e-6;
static constexpr double kGradNullRatio = 1e-9;

unsigned applyGradientFloor(std::unordered_map<size_t, ProfileInfo> &profileMap,
                            llvm::StringRef functionName) {
  if (profileMap.empty())
    return 0;

  // sumAbsGrad is used when the profile carries it: the signed sum understates
  // a value whose adjoint alternates in sign across executions.
  auto weightOf = [](const ProfileInfo &p) {
    return p.sumAbsGrad >= 0.0 ? p.sumAbsGrad : std::fabs(p.sumGrad);
  };

  double ref = 0.0;
  for (const auto &kv : profileMap) {
    if (kv.second.exec == 0)
      continue;
    ref = std::max(ref, weightOf(kv.second) / (double)kv.second.exec);
  }
  if (!(ref > 0.0) || !std::isfinite(ref))
    return 0;

  unsigned floored = 0;
  for (auto &kv : profileMap) {
    ProfileInfo &p = kv.second;
    if (p.exec == 0)
      continue;
    const double perExec = weightOf(p) / (double)p.exec;
    const double ratio = perExec / ref;
    if (ratio < kGradNullRatio) {
      llvm::errs() << "Poseidon: profiled instruction " << kv.first << " of "
                   << functionName << " has per-execution gradient " << perExec
                   << ", " << ratio
                   << " of this site's largest. A connected site cannot "
                      "produce that spread; the adjoint seed most likely lies "
                      "in the left null space of the operator (a constant seed "
                      "does for any partition-of-unity basis with a derivative "
                      "contraction). Re-profile with a non-degenerate seed.\n";
    }
    const double floorW = kGradFloorRatio * ref * (double)p.exec;
    if (weightOf(p) < floorW) {
      // Write the floor back through sumGrad, which is what every consumer
      // reads (FPNode::grad, CandidateOutput::grad, MatmulProfile::gradD).
      p.sumGrad = std::copysign(floorW, p.sumGrad == 0.0 ? 1.0 : p.sumGrad);
      if (p.sumAbsGrad >= 0.0)
        p.sumAbsGrad = floorW;
      ++floored;
    }
  }
  if (floored)
    llvm::errs() << "Poseidon: gradient floor raised the accuracy weight of "
                 << floored << " of " << profileMap.size()
                 << " profiled instruction(s) in " << functionName << " to "
                 << kGradFloorRatio << " x the site maximum\n";
  return floored;
}

void parseProfileFile(const std::string &profilePath,
                      std::unordered_map<size_t, ProfileInfo> &profileMap,
                      FunctionProfileHeader *header) {
  profileMap.clear();
  if (header)
    *header = FunctionProfileHeader{};
  std::ifstream file(profilePath);
  if (!file.is_open()) {
    llvm::errs() << "Warning: Could not open profile file: " << profilePath
                 << "\n";
    return;
  }

  std::string line;
  std::regex indexPattern(R"(^(\d+)$)");
  std::regex blockDimsPattern(
      R"(^\s*MaxBlockDims\s*=\s*(\d+)\s+(\d+)\s+(\d+))");
  std::regex gridDimsPattern(R"(^\s*MaxGridDims\s*=\s*(\d+)\s+(\d+)\s+(\d+))");
  std::regex launchCountPattern(R"(^\s*LaunchCount\s*=\s*(\d+))");
  std::regex redTripPattern(R"(^\s*RedTrip\s*=\s*(\d+)\s+(\d+))");
  std::regex canonicalHashPattern(R"(^\s*CanonicalHash\s*=\s*([0-9a-fA-F]+))");
  std::regex siteIdPattern(R"(^\s*SiteId\s*=\s*(\d+))");
  std::regex kappaPattern(
      R"(^\s*Kappa\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))");
  std::regex kappaMarkPattern(R"(^\s*KappaMark\s*=\s*(\S+))");

  std::regex minResPattern(
      R"(^\s*MinRes\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))");
  std::regex maxResPattern(
      R"(^\s*MaxRes\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))");
  std::regex sumValuePattern(
      R"(^\s*SumValue\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))");
  std::regex sumSensPattern(
      R"(^\s*SumSens\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))");
  std::regex sumGradPattern(
      R"(^\s*SumGrad\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))");
  // Optional: profiles written before the runtime accumulated it have no line.
  std::regex sumAbsGradPattern(
      R"(^\s*SumAbsGrad\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))");
  std::regex execPattern(R"(^\s*Exec\s*=\s*(\d+))");
  std::regex numOperandsPattern(R"(^\s*NumOperands\s*=\s*(\d+))");
  std::regex operandPattern(
      R"(^\s*Operand\[(\d+)\]\s*=\s*\[([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan),\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan)(?:,\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf|-inf|nan|-nan))?\])");

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    std::smatch match;
    if (header && std::regex_search(line, match, blockDimsPattern)) {
      header->maxBlockDim[0] = static_cast<uint32_t>(std::stoul(match[1]));
      header->maxBlockDim[1] = static_cast<uint32_t>(std::stoul(match[2]));
      header->maxBlockDim[2] = static_cast<uint32_t>(std::stoul(match[3]));
      continue;
    }
    if (header && std::regex_search(line, match, gridDimsPattern)) {
      header->maxGridDim[0] = static_cast<uint32_t>(std::stoul(match[1]));
      header->maxGridDim[1] = static_cast<uint32_t>(std::stoul(match[2]));
      header->maxGridDim[2] = static_cast<uint32_t>(std::stoul(match[3]));
      continue;
    }
    if (header && std::regex_search(line, match, launchCountPattern)) {
      header->launchCount = std::stoull(match[1]);
      continue;
    }
    if (header && std::regex_search(line, match, redTripPattern)) {
      header->redTrip[std::stoull(match[1])] =
          static_cast<unsigned>(std::stoul(match[2]));
      continue;
    }
    if (header && std::regex_search(line, match, canonicalHashPattern)) {
      header->canonicalHash = match[1];
      continue;
    }
    if (header && std::regex_search(line, match, siteIdPattern)) {
      header->siteId = std::stoi(match[1]);
      continue;
    }
    if (header && std::regex_search(line, match, kappaPattern)) {
      header->kappa = std::stod(match[1]);
      header->hasKappa = true;
      continue;
    }
    if (header && std::regex_search(line, match, kappaMarkPattern)) {
      header->kappaMark = match[1];
      continue;
    }
    if (std::regex_match(line, match, indexPattern)) {
      size_t idx = std::stoull(match[1]);
      ProfileInfo info;

      std::string minResLine, maxResLine, sumValueLine, sumSensLine,
          sumGradLine, sumAbsGradLine, execLine, numOperandsLine;

      if (std::getline(file, minResLine) && std::getline(file, maxResLine) &&
          std::getline(file, sumValueLine) && std::getline(file, sumSensLine) &&
          std::getline(file, sumGradLine) && std::getline(file, execLine) &&
          (execLine.find("SumAbsGrad") == std::string::npos ||
           (sumAbsGradLine = execLine, std::getline(file, execLine))) &&
          std::getline(file, numOperandsLine)) {

        auto stripCR = [](std::string &s) {
          if (!s.empty() && s.back() == '\r')
            s.pop_back();
        };
        stripCR(minResLine);
        stripCR(maxResLine);
        stripCR(sumValueLine);
        stripCR(sumSensLine);
        stripCR(sumGradLine);
        stripCR(sumAbsGradLine);
        stripCR(execLine);
        stripCR(numOperandsLine);

        std::smatch mMinRes, mMaxRes, mSumValue, mSumSens, mSumGrad, mExec,
            mNumOperands;
        if (std::regex_search(minResLine, mMinRes, minResPattern) &&
            std::regex_search(maxResLine, mMaxRes, maxResPattern) &&
            std::regex_search(sumValueLine, mSumValue, sumValuePattern) &&
            std::regex_search(sumSensLine, mSumSens, sumSensPattern) &&
            std::regex_search(sumGradLine, mSumGrad, sumGradPattern) &&
            std::regex_search(execLine, mExec, execPattern) &&
            std::regex_search(numOperandsLine, mNumOperands,
                              numOperandsPattern)) {

          info.minRes = stringToDouble(mMinRes[1]);
          info.maxRes = stringToDouble(mMaxRes[1]);
          info.sumValue = stringToDouble(mSumValue[1]);
          info.sumSens = stringToDouble(mSumSens[1]);
          info.sumGrad = stringToDouble(mSumGrad[1]);
          std::smatch mSumAbsGrad;
          if (!sumAbsGradLine.empty() &&
              std::regex_search(sumAbsGradLine, mSumAbsGrad, sumAbsGradPattern))
            info.sumAbsGrad = stringToDouble(mSumAbsGrad[1]);
          info.exec = static_cast<uint64_t>(std::stoull(mExec[1]));
          unsigned numOperands =
              static_cast<unsigned>(std::stoul(mNumOperands[1]));

          info.minOperands.resize(numOperands, 0.0);
          info.maxOperands.resize(numOperands, 0.0);
          info.minMagOperands.resize(numOperands, 0.0);

          for (unsigned i = 0; i < numOperands; ++i) {
            if (std::getline(file, line)) {
              if (!line.empty() && line.back() == '\r')
                line.pop_back();

              std::smatch operandMatch;
              if (std::regex_search(line, operandMatch, operandPattern)) {
                unsigned opIdx =
                    static_cast<unsigned>(std::stoul(operandMatch[1]));
                double minVal = stringToDouble(operandMatch[2]);
                double maxVal = stringToDouble(operandMatch[3]);

                if (opIdx < numOperands) {
                  info.minOperands[opIdx] = minVal;
                  info.maxOperands[opIdx] = maxVal;
                  if (operandMatch[4].matched)
                    info.minMagOperands[opIdx] =
                        stringToDouble(operandMatch[4]);
                }
              }
            }
          }

          profileMap[idx] = info;
        } else {
          llvm::errs() << "Warning: Failed to parse profile fields for index "
                       << idx << "\n";
        }
      } else {
        llvm::errs() << "Warning: Incomplete profile entry for index " << idx
                     << "\n";
      }
    }
  }
}

// Profile-gen only, device module only: pre-declare the probes with
// enzyme_inactive before any AD runs. The tblgen-generated logging code creates
// them on first use without the attribute, and with only a declaration in the
// TU Enzyme demands a derivative ("No augmented forward pass found"). The
// signatures must match the tblgen ones exactly. Annotating the runtime's own
// definitions does not work: __has_attribute(enzyme_inactive) is false unless
// the clang plugin is loaded, so under -fpass-plugin the attribute degrades
// away.
void predeclareInactiveProfilerProbes(llvm::Module &M) {
  using namespace llvm;
  if (!Triple(M.getTargetTriple()).isNVPTX())
    return;
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *SizeTy = Type::getInt64Ty(Ctx);
  auto declare = [&](StringRef name, FunctionType *FT) {
    FunctionCallee fc = M.getOrInsertFunction(name, FT);
    if (auto *fn = dyn_cast<Function>(fc.getCallee())) {
      if (!fn->hasFnAttribute("enzyme_inactive"))
        fn->addFnAttr("enzyme_inactive");
      fn->addFnAttr(Attribute::NoFree);
    }
  };
  declare("poseidonLogValueCUDA",
          FunctionType::get(VoidTy, {PtrTy, SizeTy, DoubleTy, SizeTy, PtrTy},
                            false));
  declare(
      "poseidonLogGradCUDA",
      FunctionType::get(VoidTy, {PtrTy, SizeTy, DoubleTy, DoubleTy}, false));
  // The perturbation hooks are inactive too: nothing reads a derivative of the
  // perturbation (the probe reads the declared metric back from the run), and a
  // second run of the pass would otherwise differentiate the calls the first
  // run injected.
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  declare("poseidonProbePerturbCUDA",
          FunctionType::get(DoubleTy, {Int32Ty, DoubleTy}, false));
  declare("poseidonProbePerturbCUDAf",
          FunctionType::get(FloatTy, {Int32Ty, FloatTy}, false));
}

// Profile-gen only, host module only: call the profiler's CUDA registration
// from the top of main. A static constructor cannot do it: the runtime
// publishes its device tables with cudaMemcpyToSymbol, and a dynamic
// initializer can run before __cudaRegisterFatBinary has registered the device
// module in a multi-TU binary ("invalid device symbol", surfacing later as a
// sticky error on someone else's kernel). main runs after every static
// constructor.
//
// The host side of a .cu and a plain .cpp share the x86 triple, so the CUDA SDK
// version clang records on the module and __cudaRegisterFatBinary are the
// module facts that separate them. Erring towards injection is deliberate: a
// wrongly skipped injection loses the profile silently, a wrongly kept one
// fails loudly at link.
static bool isCUDAHostModule(const llvm::Module &M) {
  return !M.getSDKVersion().empty() || M.getFunction("__cudaRegisterFatBinary");
}

void injectHostProfilerInit(llvm::Module &M) {
  using namespace llvm;
  if (Triple(M.getTargetTriple()).isNVPTX())
    return;
  // A host-only build links no CUDA profiler runtime, so the call injected
  // below would be an undefined reference to __poseidon_profile_init_cuda.
  if (!isCUDAHostModule(M))
    return;
  Function *mainF = M.getFunction("main");
  if (!mainF || mainF->isDeclaration())
    return;
  LLVMContext &Ctx = M.getContext();
  FunctionCallee initFn =
      M.getOrInsertFunction("__poseidon_profile_init_cuda",
                            FunctionType::get(Type::getVoidTy(Ctx), {}, false));
  if (auto *Fn = dyn_cast<Function>(initFn.getCallee())) {
    if (!Fn->hasFnAttribute("enzyme_inactive"))
      Fn->addFnAttr("enzyme_inactive");
    Fn->addFnAttr(Attribute::NoFree);
  }
  // Idempotence: never place a second call.
  for (Instruction &I : instructions(*mainF))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (auto *CF = CI->getCalledFunction())
        if (CF->getName() == "__poseidon_profile_init_cuda")
          return;
  IRBuilder<> B(&*mainF->getEntryBlock().getFirstInsertionPt());
  B.CreateCall(initFn, {})->setDoesNotThrow();
  llvm::errs() << "[poseidon] FP profiler CUDA registration injected at main\n";
}

// Launch geometry is a per-function fact: the probe is keyed by the
// poseidon_site_<name> global the value probes use (which exists only after AD)
// and sits at the entry of the function owning those probes.
void injectBlockDimProbes(llvm::Module &M) {
  using namespace llvm;
  if (!Triple(M.getTargetTriple()).isNVPTX())
    return;
  Function *logFn = M.getFunction("poseidonLogValueCUDA");
  if (!logFn)
    return;
  LLVMContext &Ctx = M.getContext();
  Type *PtrTy = PointerType::getUnqual(Ctx);
  FunctionCallee probeFn = M.getOrInsertFunction(
      "poseidonProfileBlockDimsCUDA",
      FunctionType::get(Type::getVoidTy(Ctx), {PtrTy}, false));
  if (auto *Fn = dyn_cast<Function>(probeFn.getCallee())) {
    if (!Fn->hasFnAttribute("enzyme_inactive"))
      Fn->addFnAttr("enzyme_inactive");
    Fn->addFnAttr(Attribute::NoFree);
  }

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
        if (CF->getName() == "poseidonProfileBlockDimsCUDA")
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

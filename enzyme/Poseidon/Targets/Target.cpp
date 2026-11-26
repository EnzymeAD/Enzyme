//===- Target.cpp - what the pass knows about the target of a module ------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target.h"
#include "MathLibrary.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"

#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

Value *Target::emitLaunchValue(IRBuilderBase &, LaunchValue, unsigned) const {
  llvm_unreachable("the target has no in-kernel matrix raise");
}

const Target &Target::of(const Module &M) {
  const Triple T(M.getTargetTriple());
  if (T.isNVPTX())
    return nvptxTarget();
  if (T.isAMDGPU())
    return amdgpuTarget();
  return hostTarget();
}

const Target &Target::of(const Function &F) { return of(*F.getParent()); }

const Target &Target::measuredOn(StringRef processor) {
  if (processor.starts_with("sm_"))
    return nvptxTarget();
  if (processor.starts_with("gfx"))
    return amdgpuTarget();
  return hostTarget();
}

StringRef Target::processor(const Function &F) const {
  return F.getFnAttribute("target-cpu").getValueAsString();
}

bool Target::isShared(const GlobalValue &G) const {
  std::optional<unsigned> AS = sharedAddressSpace();
  return AS && G.getAddressSpace() == *AS;
}

bool Target::inlinesMathCall(const Function &callee) const {
  return callee.hasFnAttribute(Attribute::AlwaysInline);
}

bool Target::callsMathLibrary(StringRef op, Type *Ty) const {
  return isDevice() && deviceCallsMathLibrary(op, Ty);
}

const ProbeABI &Target::probes() const {
  return isDevice() ? deviceProbes() : hostProbes();
}

// The host CPU: no kernels, libm, and the host profiler
// (runtime/profile/Profiler.cpp).
namespace {

class HostTarget final : public Target {
public:
  bool isDevice() const override { return false; }
  StringRef name() const override { return "host"; }
  std::string herbiePlatform(StringRef) const override { return "default"; }

  bool isKernel(const Function &) const override { return false; }
  unsigned globalAddressSpace() const override { return 0; }
  std::optional<unsigned> sharedAddressSpace() const override {
    return std::nullopt;
  }
  uint64_t maxStaticSharedBytes() const override { return 0; }

  LaunchGeometry decodeLaunchValue(const Value *) const override { return {}; }

  StringRef mathLibrary() const override { return "libm"; }
  std::string mathFunction(StringRef base, Type *Ty) const override {
    return (Twine(base) + (Ty->isDoubleTy() ? "" : "f")).str();
  }
  std::optional<std::string> retypeMathFunction(StringRef name,
                                                Type *Ty) const override {
    return retypeLibmFunction(name, Ty).value_or("");
  }
};

} // namespace

const Target &hostTarget() {
  static const HostTarget T{};
  return T;
}

const ProbeABI &hostProbes() {
  static const ProbeABI P{"__poseidon_log_value",
                          "__poseidon_log_grad",
                          "__poseidon_perturb",
                          "__poseidon_perturbf",
                          /*BlockDims=*/"",
                          /*PublishStatic=*/"",
                          "__poseidon_register_static",
                          "__poseidon_profile_runtime"};
  return P;
}

const ProbeABI &deviceProbes() {
  static const ProbeABI P{"__poseidon_device_log_value",
                          "__poseidon_device_log_grad",
                          "__poseidon_device_perturb",
                          "__poseidon_device_perturbf",
                          "__poseidon_device_log_block_dims",
                          "__poseidon_device_publish_static",
                          /*RegisterStatic=*/"",
                          /*RuntimeAnchor=*/"",
                          "__poseidon_device_overlapping_arguments",
                          "__poseidon_device_log_access"};
  return P;
}

bool isApproxSqrtF32(Intrinsic::ID ID) {
  if (ID == Intrinsic::not_intrinsic)
    return false;
  for (const Target *T : {&hostTarget(), &nvptxTarget(), &amdgpuTarget()})
    if (T->approxSqrtF32() == ID)
      return true;
  return false;
}

Intrinsic::ID approxSqrtF32Of(const CallInst &CI) {
  const Function *Callee = CI.getCalledFunction();
  if (!Callee || Callee->isIntrinsic() || CI.arg_size() != 1)
    return Intrinsic::not_intrinsic;
  if (mathFunctionOf(Callee->getName()) != "sqrt")
    return Intrinsic::not_intrinsic;
  return Target::of(*CI.getModule()).approxSqrtF32();
}

LaunchGeometry launchGeometryOf(const Value *V) {
  const auto *I = dyn_cast<Instruction>(V);
  return I ? Target::of(*I->getModule()).decodeLaunchValue(V)
           : LaunchGeometry();
}

} // namespace poseidon

//===- Host.cpp - the host CPU target -------------------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host CPU: no kernels, libm, and the host profiler
// (runtime/profile/Profiler.cpp).
//
//===----------------------------------------------------------------------===//

#include "MathLibrary.h"
#include "Target.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Type.h"

#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

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

} // namespace poseidon

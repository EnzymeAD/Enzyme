// The host CPU: no kernels, libm, and the host profiler
// (runtime/fpprofiler/FPProfiler.cpp).
#include "Peers.h"

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
    return retypeLibmFunction(name, Ty);
  }

  const ProbeABI &probes() const override {
    static const ProbeABI P{"poseidonLogValue",
                            "poseidonLogGrad",
                            /*PerturbF64=*/"",
                            /*PerturbF32=*/"",
                            /*BlockDims=*/"",
                            /*PublishStatic=*/"",
                            "poseidonRegisterProfileStatic",
                            "POSEIDON_PROFILE_RUNTIME_VAR",
                            /*OperandCountBits=*/32};
    return P;
  }
};

} // namespace

const Target &hostTarget() {
  static const HostTarget T{};
  return T;
}

} // namespace poseidon

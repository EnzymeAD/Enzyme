// NVPTX: CUDA kernels, libdevice, tensor cores (NVPTXTensorCore.h).
#include "Peers.h"

#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Type.h"

using namespace llvm;

namespace poseidon {

namespace {

#ifdef POSEIDON_DEVICE_PROFILER
// runtime/fpprofiler/FPProfilerDevice.cu, compiled by the build to NVPTX
// bitcode.
#include "FPProfilerDevice.inc"
#endif

class NVPTXTarget final : public Target {
public:
  bool isDevice() const override { return true; }
  StringRef name() const override { return "NVPTX"; }
  std::string herbiePlatform(StringRef processor) const override {
    std::string Stripped = processor.str();
    size_t Pos;
    while ((Pos = Stripped.find('_')) != std::string::npos)
      Stripped.erase(Pos, 1);
    return "cuda-" + Stripped;
  }

  bool isKernel(const Function &F) const override {
    return F.getCallingConv() == CallingConv::PTX_Kernel;
  }
  // The launch a profiled kernel is given has a block shape nothing at
  // compile time bounds below the device's limit; unbounded, ptxas sizes the
  // reverse pass's registers for small blocks, and a launch the original makes
  // (MFEM's 375-thread elasticity reduce at p = 4) runs out of registers.
  void boundProfiledLaunch(Function &K) const override {
    if (!K.hasFnAttribute("nvvm.maxntid"))
      K.addFnAttr("nvvm.maxntid", "1024");
  }
  unsigned globalAddressSpace() const override { return 1; }
  std::optional<unsigned> sharedAddressSpace() const override { return 3; }
  // ptxas rejects the whole module above it.
  uint64_t maxStaticSharedBytes() const override { return 48 * 1024; }

  LaunchGeometry decodeLaunchValue(const Value *V) const override {
    const auto *II = dyn_cast<IntrinsicInst>(V);
    if (!II)
      return {};
    switch (II->getIntrinsicID()) {
    case Intrinsic::nvvm_read_ptx_sreg_tid_x:
      return {LaunchValue::ThreadIndex, 0};
    case Intrinsic::nvvm_read_ptx_sreg_tid_y:
      return {LaunchValue::ThreadIndex, 1};
    case Intrinsic::nvvm_read_ptx_sreg_tid_z:
      return {LaunchValue::ThreadIndex, 2};
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
      return {LaunchValue::BlockIndex, 0};
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
      return {LaunchValue::BlockIndex, 1};
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
      return {LaunchValue::BlockIndex, 2};
    case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
      return {LaunchValue::BlockDim, 0};
    case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
      return {LaunchValue::BlockDim, 1};
    case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
      return {LaunchValue::BlockDim, 2};
    default:
      return {};
    }
  }

  StringRef mathLibrary() const override { return "libdevice"; }
  std::string mathFunction(StringRef base, Type *Ty) const override {
    return ("__nv_" + base + (Ty->isDoubleTy() ? "" : "f")).str();
  }
  std::optional<StringRef> mathBase(StringRef name) const override {
    StringRef base = name;
    if (!base.consume_front("__nv_"))
      return std::nullopt;
    if (mathBases().count(base.str()))
      return base;
    if (base.ends_with("f") && mathBases().count(base.drop_back(1).str()))
      return base.drop_back(1);
    return StringRef();
  }
  std::optional<std::string> retypeMathFunction(StringRef name,
                                                Type *Ty) const override {
    if (name.size() <= 5 || !name.starts_with("__nv_"))
      return std::nullopt;
    std::string libm = retypeLibmFunction(name.drop_front(5), Ty);
    return libm.empty() ? libm : "__nv_" + libm;
  }
  bool callsMathLibrary(StringRef op, Type *Ty) const override {
    return deviceCallsMathLibrary(op, Ty);
  }
  bool needsMathLibrary(Intrinsic::ID ID, Type *Ty) const override {
    switch (ID) {
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::exp10:
    case Intrinsic::tan:
    case Intrinsic::asin:
    case Intrinsic::acos:
    case Intrinsic::atan:
    case Intrinsic::atan2:
    case Intrinsic::sinh:
    case Intrinsic::cosh:
    case Intrinsic::tanh:
    case Intrinsic::sin:
    case Intrinsic::cos:
    case Intrinsic::log:
    case Intrinsic::log2:
    case Intrinsic::log10:
    case Intrinsic::pow:
      return Ty->isDoubleTy();
    default:
      return false;
    }
  }
  // MUFU, inlined; the lossy F32 tier tolerates its ~2^-22 error.
  Intrinsic::ID approxSqrtF32() const override {
    return Intrinsic::nvvm_sqrt_approx_f;
  }

  bool hasInKernelMatrixRaise() const override { return true; }

  const ProbeABI &probes() const override { return deviceProbes(); }
  StringRef profilerBitcode() const override {
#ifdef POSEIDON_DEVICE_PROFILER
    return StringRef(reinterpret_cast<const char *>(kDeviceProfiler),
                     sizeof(kDeviceProfiler));
#else
    return "";
#endif
  }
  std::string profilerMismatch(const Function &) const override {
#ifdef POSEIDON_DEVICE_PROFILER
    return "";
#else
    return "this build of Poseidon has no NVPTX device profiler, so it cannot "
           "profile CUDA device code; configure it where CMake finds a CUDA "
           "compiler";
#endif
  }
};

} // namespace

const Target &nvptxTarget() {
  static const NVPTXTarget T{};
  return T;
}

} // namespace poseidon

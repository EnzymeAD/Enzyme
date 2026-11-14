// AMDGPU: HIP kernels, ROCm's ocml.
#include "MathFunctions.h"
#include "Peers.h"

#include "IR/Utils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/TargetParser/Triple.h"
#if LLVM_VERSION_MAJOR >= 24
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#endif

#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

namespace {

#ifdef POSEIDON_DEVICE_PROFILER_AMDGPU
// runtime/fpprofiler/FPProfilerDevice.cu, compiled as HIP to AMDGPU bitcode.
#include "FPProfilerDeviceAMDGPU.inc"
#endif

class AMDGPUTarget final : public Target {
public:
  bool isDevice() const override { return true; }
  StringRef name() const override { return "AMDGPU"; }
  // The triple names the processor (amdgpu9.42-amd-amdhsa), and a function
  // carries target-cpu only where it differs.
  StringRef processor(const Function &F) const override {
    StringRef cpu = Target::processor(F);
#if LLVM_VERSION_MAJOR >= 24
    const Triple T(F.getParent()->getTargetTriple());
    if (cpu.empty())
      return AMDGPU::getArchNameFromSubArch(T.getSubArch());
#endif
    return cpu;
  }
  std::string herbiePlatform(StringRef processor) const override {
    return ("hip-" + processor).str();
  }

  bool isKernel(const Function &F) const override {
    return F.getCallingConv() == CallingConv::AMDGPU_KERNEL;
  }
  // No boundProfiledLaunch: a profiled kernel has its bound as the
  // amdgpu-flat-work-group-size it shares with the original.
  unsigned globalAddressSpace() const override { return 1; }
  std::optional<unsigned> sharedAddressSpace() const override { return 3; }
  // The local data share a workgroup addresses (64 KiB on gfx942, AMDGPU.td
  // FeatureISAVersion9_4_2).
  uint64_t maxStaticSharedBytes() const override { return 64 * 1024; }
  bool allocatesAddressTakenShared() const override { return true; }

  // AMDGPU has no block-size register: HIP's blockDim reads the work-group
  // size out of the implicit kernel arguments, three i16 from byte 12 (code
  // object v5 and later).
  LaunchGeometry decodeLaunchValue(const Value *V) const override {
    if (const auto *L = dyn_cast<LoadInst>(V)) {
      if (!L->getType()->isIntegerTy(16))
        return {};
      const DataLayout &DL = L->getModule()->getDataLayout();
      APInt off(DL.getIndexTypeSizeInBits(L->getPointerOperandType()), 0);
      const auto *base = dyn_cast<IntrinsicInst>(
          L->getPointerOperand()->stripAndAccumulateConstantOffsets(
              DL, off, /*AllowNonInbounds=*/false));
      if (!base || base->getIntrinsicID() != Intrinsic::amdgcn_implicitarg_ptr)
        return {};
      int64_t o = off.getSExtValue();
      if (o == 12 || o == 14 || o == 16)
        return {LaunchValue::BlockDim, (int)(o - 12) / 2};
      return {};
    }
    const auto *II = dyn_cast<IntrinsicInst>(V);
    if (!II)
      return {};
    switch (II->getIntrinsicID()) {
    case Intrinsic::amdgcn_workitem_id_x:
      return {LaunchValue::ThreadIndex, 0};
    case Intrinsic::amdgcn_workitem_id_y:
      return {LaunchValue::ThreadIndex, 1};
    case Intrinsic::amdgcn_workitem_id_z:
      return {LaunchValue::ThreadIndex, 2};
    case Intrinsic::amdgcn_workgroup_id_x:
      return {LaunchValue::BlockIndex, 0};
    case Intrinsic::amdgcn_workgroup_id_y:
      return {LaunchValue::BlockIndex, 1};
    case Intrinsic::amdgcn_workgroup_id_z:
      return {LaunchValue::BlockIndex, 2};
    default:
      return {};
    }
  }

  StringRef mathLibrary() const override { return "ocml"; }
  std::string mathFunction(StringRef base, Type *Ty) const override {
    return ("__ocml_" + base + (Ty->isDoubleTy() ? "_f64" : "_f32")).str();
  }
  std::optional<StringRef> mathBase(StringRef name) const override {
    StringRef base = name;
    if (!base.consume_front("__ocml_"))
      return std::nullopt;
    return (base.consume_back("_f64") || base.consume_back("_f32")) &&
                   mathBases().count(base.str())
               ? base
               : StringRef();
  }
  std::optional<std::string> retypeMathFunction(StringRef name,
                                                Type *Ty) const override {
    StringRef ocml = name;
    if (!ocml.consume_front("__ocml_"))
      return std::nullopt;
    if (!(ocml.consume_back("_f64") || ocml.consume_back("_f32")) ||
        !libmFuncs().count(ocml.str()))
      return std::string();
    if (Ty->isDoubleTy())
      return ("__ocml_" + ocml + "_f64").str();
    if (Ty->isHalfTy() || Ty->isBFloatTy() || Ty->isFloatTy())
      return ("__ocml_" + ocml + "_f32").str();
    return std::string();
  }
  bool callsMathLibrary(StringRef op, Type *Ty) const override {
    return deviceCallsMathLibrary(op, Ty);
  }
  bool needsMathLibrary(Intrinsic::ID ID, Type *Ty) const override {
    switch (ID) {
    case Intrinsic::tan:
    case Intrinsic::asin:
    case Intrinsic::acos:
    case Intrinsic::atan:
    case Intrinsic::atan2:
    case Intrinsic::sinh:
    case Intrinsic::cosh:
    case Intrinsic::tanh:
      return Ty->isDoubleTy() || Ty->isFloatTy();
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
  // ocml's functions are not always-inline, and a build without the plugin
  // inlines them all the same.
  bool inlinesMathCall(const Function &) const override { return true; }

  const ProbeABI &probes() const override { return deviceProbes(); }
  StringRef profilerBitcode() const override {
#ifdef POSEIDON_DEVICE_PROFILER_AMDGPU
    return StringRef(reinterpret_cast<const char *>(kDeviceProfilerAMDGPU),
                     sizeof(kDeviceProfilerAMDGPU));
#else
    return "";
#endif
  }
  // The library carries the ROCm device libraries of one processor, so a
  // probe compiled for another cannot call into it.
  std::string profilerMismatch(const Function &F) const override {
#ifdef POSEIDON_DEVICE_PROFILER_AMDGPU
    if (processor(F) != POSEIDON_DEVICE_PROFILER_AMDGPU_ARCH)
      return (Twine("the AMDGPU device profiler was built for '") +
              POSEIDON_DEVICE_PROFILER_AMDGPU_ARCH
              "' (POSEIDON_AMDGPU_ARCH) but " +
              demangle(F.getName()) + " compiles for target-cpu '" +
              processor(F) +
              "'; build Poseidon for this processor, or compile the profiling "
              "build for that one")
          .str();
    return "";
#else
    return "this build of Poseidon has no AMDGPU device profiler, so it cannot "
           "profile HIP device code; configure it with POSEIDON_ROCM_PATH and "
           "an LLVM that builds the AMDGPU target";
#endif
  }
};

} // namespace

const Target &amdgpuTarget() {
  static const AMDGPUTarget T{};
  return T;
}

} // namespace poseidon

//===- NVPTX.cpp - the NVPTX target ---------------------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NVPTX: CUDA kernels, libdevice, tensor cores.
//
//===----------------------------------------------------------------------===//

#include "MathLibrary.h"
#include "Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Type.h"

#include <cassert>
#include <cmath>
#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

/// The special registers of the thread index, the block index and the block
/// dimension, by LaunchValue and axis.
static constexpr Intrinsic::ID kSreg[3][3] = {
    {Intrinsic::nvvm_read_ptx_sreg_tid_x, Intrinsic::nvvm_read_ptx_sreg_tid_y,
     Intrinsic::nvvm_read_ptx_sreg_tid_z},
    {Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
     Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
     Intrinsic::nvvm_read_ptx_sreg_ctaid_z},
    {Intrinsic::nvvm_read_ptx_sreg_ntid_x, Intrinsic::nvvm_read_ptx_sreg_ntid_y,
     Intrinsic::nvvm_read_ptx_sreg_ntid_z}};
static constexpr LaunchValue kSregValue[3] = {
    LaunchValue::ThreadIndex, LaunchValue::BlockIndex, LaunchValue::BlockDim};

namespace {

#ifdef POSEIDON_DEVICE_PROFILER
// runtime/profile/DeviceProfilerDevice.cu, compiled by the build to NVPTX
// bitcode.
#include "DeviceProfilerDevice.inc"
#endif

class NVPTXTarget final : public Target {
public:
  bool isDevice() const override { return true; }
  StringRef name() const override { return "NVPTX"; }
  std::string herbiePlatform(StringRef processor) const override {
    std::string Stripped = processor.str();
    llvm::erase(Stripped, '_');
    return "cuda-" + Stripped;
  }

  bool isKernel(const Function &F) const override {
    return F.getCallingConv() == CallingConv::PTX_Kernel;
  }
  /// Nothing at compile time bounds the block shape of a profiled kernel's
  /// launch below the device's limit; unbounded, ptxas sizes the reverse pass's
  /// registers for small blocks, and a larger launch the original makes runs
  /// out of registers.
  void boundProfiledLaunch(Function &K) const override {
    if (!K.hasFnAttribute("nvvm.maxntid"))
      K.addFnAttr("nvvm.maxntid", "1024");
  }
  unsigned globalAddressSpace() const override { return 1; }
  std::optional<unsigned> sharedAddressSpace() const override { return 3; }
  /// ptxas rejects the whole module above it.
  uint64_t maxStaticSharedBytes() const override { return 48 * 1024; }

  LaunchGeometry decodeLaunchValue(const Value *V) const override {
    const auto *II = dyn_cast<IntrinsicInst>(V);
    if (!II)
      return {};
    for (unsigned kind = 0; kind < 3; ++kind)
      for (unsigned axis = 0; axis < 3; ++axis)
        if (II->getIntrinsicID() == kSreg[kind][axis])
          return {kSregValue[kind], (int)axis};
    return {};
  }
  Value *emitLaunchValue(IRBuilderBase &B, LaunchValue Kind,
                         unsigned Axis) const override {
    const LaunchValue *Row = llvm::find(kSregValue, Kind);
    assert(Row != std::end(kSregValue) && "no register holds the value");
    return B.CreateCall(Intrinsic::getOrInsertDeclaration(
        B.GetInsertBlock()->getModule(), kSreg[Row - kSregValue][Axis]));
  }

  StringRef mathLibrary() const override { return "libdevice"; }
  std::string mathFunction(StringRef base, Type *Ty) const override {
    return ("__nv_" + base + (Ty->isDoubleTy() ? "" : "f")).str();
  }
  std::optional<StringRef> mathBase(StringRef name) const override {
    StringRef base = name;
    if (!base.consume_front("__nv_"))
      return std::nullopt;
    if (isDeviceMathBase(base))
      return base;
    if (base.ends_with("f") && isDeviceMathBase(base.drop_back(1)))
      return base.drop_back(1);
    return StringRef();
  }
  std::optional<std::string> retypeMathFunction(StringRef name,
                                                Type *Ty) const override {
    if (name.size() <= 5 || !name.starts_with("__nv_"))
      return std::nullopt;
    std::optional<std::string> libm =
        retypeLibmFunction(name.drop_front(5), Ty);
    return libm ? "__nv_" + *libm : std::string();
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
  /// One instruction where sqrt.rn.f32 is a sequence; the backend selects one
  /// divided by it as rsqrt.approx.f32.
  Intrinsic::ID approxSqrtF32() const override {
    return Intrinsic::nvvm_sqrt_approx_f;
  }
  /// PTX ISA 8.8, sections 9.7.3.15 and 9.7.3.16: over the positive finite
  /// range, sqrt.approx.f32 is within 2^-23 and rsqrt.approx.f32 within
  /// 2^-22.9 of the exact result. The model divides one by its root and rounds
  /// where rsqrt.approx.f32 does not, which can take up to 2^-24 off the error
  /// it carries.
  double approxSqrtF32Error() const override {
    return std::exp2(-22.9) + 0x1p-24;
  }

  bool hasInKernelMatrixRaise() const override { return true; }

  StringRef profilerBitcode() const override {
#ifdef POSEIDON_DEVICE_PROFILER
    return StringRef(reinterpret_cast<const char *>(kDeviceProfiler),
                     sizeof(kDeviceProfiler));
#else
    return "";
#endif
  }
  std::optional<std::string> profilerMismatch(const Function &) const override {
#ifdef POSEIDON_DEVICE_PROFILER
    return std::nullopt;
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

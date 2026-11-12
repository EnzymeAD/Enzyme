//===- Target.h - what the pass knows about the target of a module -------===//
//
// One Target per kind of module triple: the host CPU, NVPTX and AMDGPU. A
// module whose triple is neither GPU target compiles for the host. The three
// are stateless, so code asks the module's target directly; a question whose
// answer depends on the processor takes the function.
//===---------------------------------------------------------------------===//
#ifndef POSEIDON_TARGET_H
#define POSEIDON_TARGET_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Intrinsics.h"

#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class CallInst;
class Function;
class GlobalValue;
class Module;
class Type;
class Value;
} // namespace llvm

namespace poseidon {

// A value a kernel reads its launch geometry from, with its axis (0 to 2).
enum class LaunchValue { None, ThreadIndex, BlockIndex, BlockDim };
struct LaunchGeometry {
  LaunchValue Kind = LaunchValue::None;
  int Axis = -1;
};

// The profiler runtime entry points the probes of a target call
// (runtime/fpprofiler). Both GPU targets call the device profiler; a name the
// target's runtime does not have is empty.
struct ProbeABI {
  // (site name, slot, value, operand count, operands)
  llvm::StringRef LogValue;
  // (site name, slot, value, adjoint)
  llvm::StringRef LogGradient;
  // The condition-number perturbation applied in a probe's augmented forward.
  llvm::StringRef PerturbF64, PerturbF32;
  // The launch geometry of the function owning the probes.
  llvm::StringRef BlockDims;
  // A site's static profile header: published by the site's kernel on the
  // device, registered by a constructor on the host.
  llvm::StringRef PublishStatic, RegisterStatic;
  // The variable that links the host runtime's writer into the program.
  llvm::StringRef RuntimeAnchor;
  unsigned OperandCountBits;
  // (site name, argument, argument): stops the profiling run at a device call
  // site whose arguments reach overlapping memory the pass gave separate
  // shadows.
  llvm::StringRef OverlappingArguments = "";
};

class Target {
public:
  static const Target &of(const llvm::Module &M);
  static const Target &of(const llvm::Function &F);
  // The target of the processors named like `processor`, such as a cost
  // model's native_arch.
  static const Target &measuredOn(llvm::StringRef processor);

  // Identity.
  virtual bool isDevice() const = 0;
  virtual llvm::StringRef name() const = 0;
  virtual llvm::StringRef processor(const llvm::Function &F) const;
  // The Herbie platform of a cost model measured on `processor`.
  virtual std::string herbiePlatform(llvm::StringRef processor) const = 0;

  // Kernel ABI and memory.
  virtual bool isKernel(const llvm::Function &F) const = 0;
  // Bounds the block a profiled kernel `K` may be launched with to the
  // device's limit, where the target has no bound of its own on `K`.
  virtual void boundProfiledLaunch(llvm::Function &K) const {}
  virtual unsigned globalAddressSpace() const = 0;
  // A workgroup's shared memory; none on the host.
  virtual std::optional<unsigned> sharedAddressSpace() const = 0;
  bool isShared(const llvm::GlobalValue &G) const;
  // The most static shared memory one kernel can declare.
  virtual uint64_t maxStaticSharedBytes() const = 0;
  // Whether a kernel that may make an unknown call is allocated the static
  // shared memory of every address-taken function.
  virtual bool allocatesAddressTakenShared() const { return false; }

  // Launch geometry.
  virtual LaunchGeometry decodeLaunchValue(const llvm::Value *V) const = 0;

  // The math library.
  virtual llvm::StringRef mathLibrary() const = 0;
  // Its function for libm's `base`, at double or, for any narrower `Ty`,
  // single precision.
  virtual std::string mathFunction(llvm::StringRef base,
                                   llvm::Type *Ty) const = 0;
  // libm's name behind `name` if `name` is this library's, else nullopt;
  // empty if it is this library's and not a function Poseidon handles.
  virtual std::optional<llvm::StringRef> mathBase(llvm::StringRef name) const {
    return std::nullopt;
  }
  // `name`, this library's function, at `Ty`; nullopt if `name` is not this
  // library's, empty if it cannot be retyped.
  virtual std::optional<std::string>
  retypeMathFunction(llvm::StringRef name, llvm::Type *Ty) const = 0;
  // Whether a rewrite writes Herbie's operator `op` at `Ty` as a call of the
  // library rather than as LLVM's intrinsic or libm call.
  virtual bool callsMathLibrary(llvm::StringRef op, llvm::Type *Ty) const {
    return false;
  }
  // Whether the target selects the intrinsic `ID` at `Ty` only as a call of
  // the library's function.
  virtual bool needsMathLibrary(llvm::Intrinsic::ID ID, llvm::Type *Ty) const {
    return false;
  }
  // Whether a call of the library's `callee` is inlined where no inliner
  // reaches it any more, as a build without the plugin inlines it.
  virtual bool inlinesMathCall(const llvm::Function &callee) const;
  // The approximate single-precision square root, not_intrinsic where the
  // target has none.
  virtual llvm::Intrinsic::ID approxSqrtF32() const {
    return llvm::Intrinsic::not_intrinsic;
  }
  // The largest relative error, from the exact root, of what the code
  // generator makes of approxSqrtF32() and of one divided by it.
  virtual double approxSqrtF32Error() const { return 0; }

  // Whether a matrix product can be raised onto matrix instructions inside
  // its kernel (Targets/NVPTXTensorCore.h).
  virtual bool hasInKernelMatrixRaise() const { return false; }

  // The profiler runtime.
  virtual const ProbeABI &probes() const = 0;
  // The device profiler, as bitcode this build carries; empty if it carries
  // none for this target.
  virtual llvm::StringRef profilerBitcode() const { return ""; }
  // Why this build's device profiler cannot serve the probes of `F` (it
  // carries none for this target, or one for another processor), empty if it
  // can.
  virtual std::string profilerMismatch(const llvm::Function &F) const {
    return "";
  }

protected:
  Target() = default;
  Target(const Target &) = delete;
  Target &operator=(const Target &) = delete;
  ~Target() = default;
};

// Whether `ID` is a target's approximate single-precision square root, in any
// module.
bool isApproxSqrtF32(llvm::Intrinsic::ID ID);

// The approximate square root precision tuning writes the square root call
// `CI` as when it retypes the call to single precision, not_intrinsic if it
// calls the library's single-precision function instead.
llvm::Intrinsic::ID approxSqrtF32Of(const llvm::CallInst &CI);

// The launch geometry `V` reads, in the target of its module.
LaunchGeometry launchGeometryOf(const llvm::Value *V);

} // namespace poseidon

#endif // POSEIDON_TARGET_H

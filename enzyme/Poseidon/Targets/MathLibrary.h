//===- MathLibrary.h - the math functions the pass knows by name ----------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One row per libm function the pass knows, with a column for each question a
// part of the pass asks about it, so that every list of names is read off one
// table. include/poseidon/poseidon.h keeps its own literal lists, being a
// public header: the functions it links on NVPTX are the Device column, those
// on AMDGPU the Device column without ceil, copysign, fabs, floor, fma, fmax,
// fmin, powi, rint, round, sqrt and trunc.
//
// A device module whose sites are profiled or rewritten keeps the libdevice
// (NVPTX) or ocml (AMDGPU) math functions Poseidon models as calls through the
// inliner, so that both actions see a site's math as the same operations and
// Enzyme differentiates them by name, and keeps defined the ones poseidon.h
// links for a rewrite or a derivative to call. Every other device module drops
// what poseidon.h added and compiles as it does without the plugin.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATH_LIBRARY_H
#define POSEIDON_MATH_LIBRARY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace llvm {
class CallInst;
class Function;
class IRBuilderBase;
class Module;
class Type;
class Value;
} // namespace llvm

namespace poseidon {

struct MathFunction {
  llvm::StringLiteral Name;
  /// Priced from the cost model and retyped by a precision change.
  bool Libm;
  /// A call to it, or to its float form with an f suffix, is an operation
  /// isOptimizable accepts.
  bool Optimizable;
  /// Both GPU targets select its device math library function.
  bool Device;
  /// The C rendering of a rewrite writes its float form with an f suffix.
  bool FloatSuffixInC;
  /// A rewrite writes Herbie's operator of this name as a math-library call on
  /// either GPU target: deviceCallsMathLibrary.
  bool DeviceCall;
};

/// Name, Libm, Optimizable, Device, FloatSuffixInC, DeviceCall. Header-only so
/// that a tool reads the table without linking the pass.
inline constexpr MathFunction kMathFunctions[] = {
    {"sin", true, true, true, true, true},
    {"cos", true, true, true, true, true},
    {"tan", true, true, true, true, true},
    {"asin", true, true, true, true, true},
    {"acos", true, true, true, true, true},
    {"atan", true, true, true, true, true},
    {"atan2", true, true, true, true, true},
    {"sinh", true, true, true, true, true},
    {"cosh", true, true, true, true, true},
    {"tanh", true, true, true, true, true},
    {"asinh", true, true, true, true, true},
    {"acosh", true, true, true, true, true},
    {"atanh", true, true, true, true, true},
    {"exp", true, true, true, true, true},
    {"log", true, true, true, true, true},
    {"sqrt", true, true, true, true, false},
    {"cbrt", true, true, true, true, true},
    {"pow", true, true, true, true, true},
    {"powi", true, false, true, false, false},
    {"fabs", true, true, true, true, false},
    {"fma", true, true, true, true, false},
    {"hypot", true, true, true, true, false},
    {"expm1", true, true, true, true, true},
    {"log1p", true, true, true, true, true},
    {"ceil", true, true, true, true, false},
    {"floor", true, true, true, true, false},
    {"erf", true, true, true, true, false},
    {"exp2", true, true, true, true, true},
    {"lgamma", true, true, true, true, false},
    {"log10", true, true, true, true, true},
    {"log2", true, true, true, true, true},
    {"rint", true, true, true, true, false},
    {"round", true, true, true, true, false},
    {"tgamma", true, true, true, true, false},
    {"trunc", true, true, true, true, false},
    {"copysign", true, true, true, true, false},
    {"fdim", true, true, true, true, false},
    {"fmod", true, true, true, true, false},
    {"remainder", true, true, true, true, false},
    {"fmin", false, false, true, true, false},
    {"fmax", false, false, true, true, false},
    {"erfc", false, false, false, true, false},
    {"nearbyint", false, false, false, true, false},
};

inline llvm::ArrayRef<MathFunction> mathFunctions() { return kMathFunctions; }

/// The math function a call to `Name` computes: the libm function behind a
/// libdevice or ocml entry point Poseidon handles, else `Name` when it is a row
/// of mathFunctions(), else its float form's row without the f suffix, else
/// empty. "fmax" for fmax, fmaxf, __nv_fmax and __ocml_fmax_f32.
llvm::StringRef mathFunctionOf(llvm::StringRef Name);

/// Whether `Name` is a libm function Poseidon prices and can retype.
bool isLibmFunction(llvm::StringRef Name);

/// Whether a call to `Name` is one of a function isOptimizable accepts.
bool isOptimizableLibmCall(llvm::StringRef Name);

/// Whether the C library spells the single-precision form of `Name` with an
/// 'f' suffix.
bool hasFloatSuffixInC(llvm::StringRef Name);

/// Whether both GPU targets select the device math library's function for
/// libm's `Name`; true of fmin and fmax too, which both select as
/// llvm.minnum/llvm.maxnum.
bool isDeviceMathBase(llvm::StringRef Name);

/// libm's `name` at `Ty`; none if it is none Poseidon can retype.
std::optional<std::string> retypeLibmFunction(llvm::StringRef name,
                                              llvm::Type *Ty);

/// Herbie's operators a rewrite writes as math-library calls on either GPU
/// target: their intrinsics mostly have no instruction there and fail
/// instruction selection ("Cannot select: fNN = fsin").
bool deviceCallsMathLibrary(llvm::StringRef op, llvm::Type *Ty);

/// The libm name behind a CUDA libdevice or ROCm ocml entry point Poseidon
/// handles, for both the f64 and the f32 form ("__nv_sqrt", "__nv_sqrtf" and
/// "__ocml_sqrt_f64" all give "sqrt"); none for any other function.
std::optional<llvm::StringRef> deviceMathName(llvm::StringRef fnName);

/// `funcName`, a libm, libdevice or ocml function, at `newType`; empty if it is
/// none Poseidon can retype.
std::string getLibmFunctionForPrecision(llvm::StringRef funcName,
                                        llvm::Type *newType);

/// Keeps the modelled math functions of M out of line until releaseDeviceMath.
/// Runs before the inliner, in a device module with a site under profile
/// generation or use.
bool keepDeviceMath(llvm::Module &M);

/// A device module with a site under a profile action is an error unless it
/// includes poseidon/poseidon.h under POSEIDON_DEVICE_MATH.
llvm::Error requireDeviceMathMacro(llvm::Module &M);

/// In any other device module before the inliner, and in a kept one once its
/// sites are done: the math functions inline again where an inliner still runs,
/// and poseidon.h's references go, with what was linked only for them.
bool releaseDeviceMath(llvm::Module &M);

/// In a function a rewrite reached, once the kept math is released: its calls
/// of that math, which no inliner reaches any more, inlined.
void inlineDeviceMath(llvm::Function &F);

/// A call a rewrite adds to a libm or libdevice function, in that function's
/// calling convention: GlobalOpt moves a function only ever called directly to
/// fastcc, and a call in another convention is undefined. On the device a
/// callee the module does not define is marked for requireDeviceCallees.
llvm::CallInst *createMathCall(llvm::IRBuilderBase &B, llvm::FunctionCallee F,
                               llvm::ArrayRef<llvm::Value *> Args,
                               const llvm::Twine &Name = "");

/// Once the sites are done, as nothing links libdevice or ocml into a device
/// module later: a marked call to a function the module still does not define,
/// or under profile generation a transcendental intrinsic the target cannot
/// select (what Enzyme writes a derivative with when the library's function is
/// missing), is an error saying what to include.
llvm::Error requireDeviceCallees(llvm::Module &M, bool profileGenerate);

} // namespace poseidon

#endif // POSEIDON_MATH_LIBRARY_H

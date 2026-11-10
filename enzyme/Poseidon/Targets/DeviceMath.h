//===- DeviceMath.h - libdevice math of device modules with a site -------===//
//
// A device module whose sites are profiled or rewritten keeps the libdevice
// (NVPTX) or ocml (AMDGPU) math functions Poseidon models as calls through the
// inliner, so that both actions see a site's math as the same operations and
// Enzyme differentiates them by name, and keeps defined the ones poseidon.h
// links for a rewrite or a derivative to call. Every other device module drops
// what poseidon.h added and compiles as it does without the plugin.
//===---------------------------------------------------------------------===//
#ifndef POSEIDON_DEVICE_MATH_H
#define POSEIDON_DEVICE_MATH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"

namespace llvm {
class CallInst;
class Function;
class IRBuilderBase;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

// Before the inliner, in a device module with a site under profile generation
// or use.
bool keepDeviceMath(llvm::Module &M);

// In any other device module before the inliner, and in a kept one once its
// sites are done: the math functions inline again where an inliner still runs,
// and poseidon.h's references go, with what was linked only for them.
bool releaseDeviceMath(llvm::Module &M);

// In a function a rewrite reached, once the kept math is released: its calls
// of that math, which no inliner reaches any more, inlined.
void inlineDeviceMath(llvm::Function &F);

// A call a rewrite adds to a libm or libdevice function, in that function's
// calling convention: GlobalOpt moves a function only ever called directly to
// fastcc, and a call in another convention is undefined. On the device a
// callee the module does not define is marked for requireDeviceCallees.
llvm::CallInst *createMathCall(llvm::IRBuilderBase &B, llvm::FunctionCallee F,
                               llvm::ArrayRef<llvm::Value *> Args,
                               const llvm::Twine &Name = "");

// Once the sites are done, as nothing links libdevice or ocml into a device
// module later: a marked call to a function the module still does not define,
// or under profile generation a transcendental intrinsic the target cannot
// select (what Enzyme writes a derivative with when the library's function is
// missing), stops the compile with an error saying what to include.
void requireDeviceCallees(llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_DEVICE_MATH_H

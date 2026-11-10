#include "Target.h"
#include "Peers.h"

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace poseidon {

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

const ProbeABI &deviceProbes() {
  static const ProbeABI P{
      "poseidonLogValueCUDA",         "poseidonLogGradCUDA",
      "poseidonProbePerturbCUDA",     "poseidonProbePerturbCUDAf",
      "poseidonProfileBlockDimsCUDA", "poseidonProfileStaticCUDA",
      /*RegisterStatic=*/"",          /*RuntimeAnchor=*/"",
      /*OperandCountBits=*/64};
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

LaunchGeometry launchGeometryOf(const Value *V) {
  const auto *I = dyn_cast<Instruction>(V);
  return I ? Target::of(*I->getModule()).decodeLaunchValue(V)
           : LaunchGeometry();
}

} // namespace poseidon

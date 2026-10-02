//===- StageParam.h - df64 staging across a parameter array --------------===//
//
// Hoists the F64 -> df64 split out of the kernel for operands loaded from a
// pointer kernel parameter: `load double` becomes two `load float` at {+0,+4}.
//===---------------------------------------------------------------------===//
#ifndef POSEIDON_STAGE_PARAM_H
#define POSEIDON_STAGE_PARAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
class CallInst;
class Function;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

// Rewrite the df64 splits of F's pointer-parameter loads into direct limb
// loads; returns the number of parameters staged and their indices in
// `stagedParams`. `Proxy` is the function the site-argument mapping is recorded
// against (F itself at materialization, the original site function during
// pricing). Recognition needs that mapping before it changes any IR: the host
// substitutes a limb buffer for the wrapper's launch argument, which is only
// sound if the parameter reaches one and nothing else in the wrapper reads it.
unsigned stageParamArrayDS(llvm::Function &F, llvm::Function *Proxy,
                           llvm::SmallVectorImpl<unsigned> *stagedParams);

// Per-body stash (device cc1), mirroring noteGemmBody/getGemmBody.
// Descriptor file suffix for df64 parameter-array staging.
constexpr llvm::StringLiteral kStageScheme = ".dsstage";

struct StagedParamNote {
  llvm::SmallVector<unsigned, 4> bodyParams;
  bool valid = false;
};
void noteStagedBody(const llvm::Function *body, const StagedParamNote &n);
bool getStagedBody(const llvm::Function *body, StagedParamNote &out);

// Map body params to wrapper launch-arg indices via primalArgs and write
// <cacheDir>/<wrapper>.dsstage. Called unconditionally in the optimize phase:
// with nothing staged it removes any descriptor left by an earlier solve.
// `site` is the __poseidon_fp_optimize call.
void writeStageDescriptor(llvm::Function &wrapper,
                          llvm::ArrayRef<llvm::Value *> primalArgs,
                          const llvm::CallInst *site, const StagedParamNote &n,
                          llvm::StringRef cacheDir);

// Host sub-compilation: prepend the split call to every matching
// __device_stub__ launch. Must run on the host module BEFORE inlining.
bool rewriteStageStubBodies(llvm::Module &M, llvm::StringRef cacheDir);

} // namespace poseidon
#endif // POSEIDON_STAGE_PARAM_H

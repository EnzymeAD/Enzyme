#ifndef POSEIDON_CANONICALIZE_H
#define POSEIDON_CANONICALIZE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"

#include <string>

namespace llvm {
class Function;
class Value;
} // namespace llvm

namespace poseidon {

// Clone F as preprocess_<F> and bring the clone into the canonical form the
// profile was numbered against. Both phases go through here, so the slot
// indices setSlotMetadata assigns are the same ones the profile was
// recorded under.
llvm::Function *canonicalize(llvm::Function &F);

// A 16-hex-digit digest of the canonicalized clone's optimizable instruction
// sequence in slot order: opcode, result type, operand types, and the callee
// name of a call. The profile carries the digest of the form its slots were
// numbered against, and profile-use refuses a form that no longer matches.
std::string canonicalFormHash(const llvm::Function &F);

bool isOptimizable(const llvm::Value &V);

void setSlotMetadata(llvm::Function &F);

// The passes of the canonical prefix before and after its always-inliner (none
// before it at O0).
void addCanonicalPreInline(llvm::FunctionPassManager &FPM,
                           llvm::OptimizationLevel Level);
void addCanonicalPostInline(llvm::FunctionPassManager &FPM);

} // namespace poseidon

#endif // POSEIDON_CANONICALIZE_H

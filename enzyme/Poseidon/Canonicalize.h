#ifndef POSEIDON_CANONICALIZE_H
#define POSEIDON_CANONICALIZE_H

#include <string>

namespace llvm {
class Function;
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

} // namespace poseidon

#endif // POSEIDON_CANONICALIZE_H

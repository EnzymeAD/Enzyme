//===- Pipeline.h - the Poseidon passes and where they run ----------------===//
#ifndef POSEIDON_PIPELINE_H
#define POSEIDON_PIPELINE_H

namespace llvm {
class PassBuilder;
} // namespace llvm

namespace poseidon {

void registerPasses(llvm::PassBuilder &PB);

} // namespace poseidon

#endif // POSEIDON_PIPELINE_H

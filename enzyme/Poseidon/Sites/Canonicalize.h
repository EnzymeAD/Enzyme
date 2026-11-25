//===- Canonicalize.h - the canonical form of a site ----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CANONICALIZE_H
#define POSEIDON_CANONICALIZE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"

#include <optional>
#include <string>

namespace llvm {
class Function;
class Instruction;
class Value;
} // namespace llvm

namespace poseidon {

/// Clone F as preprocess_<F> and bring the clone into the canonical form the
/// profile was numbered against. Both phases go through here, so the slot
/// indices setSlotMetadata assigns are the same ones the profile was
/// recorded under.
llvm::Function *canonicalize(llvm::Function &F);

/// A 16-hex-digit digest of the canonicalized clone's optimizable instruction
/// sequence in slot order: opcode, result type, operand types, and the callee
/// name of a call. The profile carries the digest of the form its slots were
/// numbered against, and profile-use refuses a form that does not match it.
std::string canonicalFormHash(const llvm::Function &F);

/// Whether V is a floating-point operation the pass profiles and rewrites: an
/// instruction it can model, whose operands are float or double arguments and
/// literals, integer literals, or scalar floating-point or integer
/// instructions.
bool isOptimizable(const llvm::Value &V);

/// The metadata a canonical clone's optimizable instruction carries its slot
/// in: the rank among optimizable instructions, which the profile-generate and
/// profile-use compiles share.
constexpr llvm::StringLiteral kSlotMD = "poseidon.prof.idx";

/// Numbers the optimizable instructions of the canonical clone `F` in order,
/// the slots its profile records are indexed by.
void setSlotMetadata(llvm::Function &F);
/// The slot of `I`; none if it is null or has none.
std::optional<size_t> slotOf(const llvm::Instruction *I);
/// The slot of `I`, which has one.
size_t requireSlot(const llvm::Instruction *I);

/// The passes of the canonical prefix before and after its always-inliner (none
/// before it at O0).
void addCanonicalPreInline(llvm::FunctionPassManager &FPM,
                           llvm::OptimizationLevel Level);
/// Adds the passes of the canonical prefix that run after its always-inliner.
void addCanonicalPostInline(llvm::FunctionPassManager &FPM);

} // namespace poseidon

#endif // POSEIDON_CANONICALIZE_H

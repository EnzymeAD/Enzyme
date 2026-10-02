//===- Checkpointing.h - Scheme-driven checkpointing of time loops -------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// `__enzyme_checkpoint_for(step, start, n, enzyme_scheme, vt, data, ...)`
// is the loop `for (i = start; i < start + n; i++) step(i, args...)` whose
// reverse mode is driven by an external checkpointing scheme (see
// include/enzyme/checkpoint.h for the protocol).
//
// The marker is first lowered to a call to an internal function that runs
// the loop, marked with the `enzyme_checkpoint` attribute. The primal is then
// exact, and forward mode differentiates the loop like any other. Reverse
// mode does not differentiate the loop body: its augmented forward pass and
// its reverse pass are generated here, as calls into a driver that asks the
// scheme for actions and runs the augmented forward and reverse passes of one
// step at a time.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_CHECKPOINTING_H
#define ENZYME_CHECKPOINTING_H

#include "EnzymeLogic.h"

namespace llvm {
class Function;
class Module;
} // namespace llvm

/// Replace every call to `__enzyme_checkpoint_for` in `M` by a call to an
/// internal loop function carrying the `enzyme_checkpoint` attribute.
bool lowerCheckpointMarkers(llvm::Module &M);

/// Whether `F` is a loop function made by `lowerCheckpointMarkers`.
bool isCheckpointLoop(const llvm::Function *F);

/// The augmented forward pass of a checkpointed loop. It takes the loop's
/// arguments, each followed by its shadow if it is duplicated, and returns the
/// tape as an opaque pointer.
llvm::Function *createCheckpointAugmented(
    EnzymeLogic &Logic, RequestContext context, llvm::Function *loop,
    llvm::ArrayRef<DIFFE_TYPE> constant_args, TypeAnalysis &TA,
    const FnTypeInfo &typeInfo, bool runtimeActivity, bool strongZero,
    unsigned width, bool AtomicAdd);

/// The reverse pass (`ReverseModeGradient`, taking the tape last) or the
/// combined forward and reverse pass (`ReverseModeCombined`) of a
/// checkpointed loop.
/// The forward-mode derivative of a fixed-point loop: the tangent iterated at
/// the converged state. Null for other loops, which forward mode
/// differentiates through.
llvm::Function *
createCheckpointForward(EnzymeLogic &Logic, RequestContext context,
                        llvm::Function *loop, DIFFE_TYPE retType,
                        llvm::ArrayRef<DIFFE_TYPE> constant_args,
                        TypeAnalysis &TA, const FnTypeInfo &typeInfo,
                        bool runtimeActivity, bool strongZero, unsigned width);

llvm::Function *createCheckpointGradient(EnzymeLogic &Logic,
                                         RequestContext context,
                                         const ReverseCacheKey &key,
                                         TypeAnalysis &TA);

#endif // ENZYME_CHECKPOINTING_H

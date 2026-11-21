//=- Parallel.h - the threads the accuracy model runs on -----------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Independent pieces of a solve run concurrently. Each call computes into its
// own slot and the caller combines the slots in index order, so a result does
// not depend on the number of threads or on how the calls are scheduled.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_PARALLEL_H
#define POSEIDON_CORE_PARALLEL_H

#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstddef>

namespace poseidon {

// Runs Fn(I) for every I in [0, N) on up to `Threads` threads (0: every
// hardware thread) and returns once every call has returned. With one thread
// or one call, the calls run in order on the calling thread. Fn may itself
// call forEachIndex.
void forEachIndex(unsigned Threads, size_t N,
                  llvm::function_ref<void(size_t)> Fn);

// forEachIndex over the blocks of `Grain` consecutive indices of [0, N):
// Fn(Begin, End) for each.
void forEachBlock(unsigned Threads, size_t N, size_t Grain,
                  llvm::function_ref<void(size_t, size_t)> Fn);

} // namespace poseidon

#endif // POSEIDON_CORE_PARALLEL_H

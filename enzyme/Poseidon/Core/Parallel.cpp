//=- Parallel.cpp - the threads the accuracy model runs on ---------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pool is the plugin's own: a plugin calls only what its host exports, and
// opt does not export llvm::StdThreadPool.
//
//===----------------------------------------------------------------------===//

#include "Parallel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/thread.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

using namespace llvm;

namespace poseidon {

namespace {

// The indices of one forEachIndex call, claimed one at a time by every thread
// that takes part in it.
struct Batch {
  function_ref<void(size_t)> Fn;
  size_t N;
  std::atomic<size_t> Next{0};
  // Guarded by the pool's lock: the indices run so far, and the workers that
  // hold the batch.
  size_t Finished = 0;
  unsigned Users = 0;

  Batch(function_ref<void(size_t)> Fn, size_t N) : Fn(Fn), N(N) {}

  // Runs indices until none is left to claim; returns how many it ran.
  size_t runIndices() {
    size_t Ran = 0;
    for (size_t I; (I = Next.fetch_add(1)) < N; ++Ran)
      Fn(I);
    return Ran;
  }
};

class Pool {
public:
  // `Count` threads take part in a batch: the caller and Count - 1 workers.
  explicit Pool(unsigned Count) {
    for (unsigned I = 1; I < Count; ++I)
      Workers.emplace_back([this] { work(); });
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> Guard(Lock);
      Stop = true;
    }
    WorkAvailable.notify_all();
    for (llvm::thread &T : Workers)
      T.join();
  }

  // The caller runs indices too, so a batch started from inside another
  // finishes even when every worker is busy.
  void run(Batch &B) {
    {
      std::lock_guard<std::mutex> Guard(Lock);
      Open.push_back(&B);
    }
    WorkAvailable.notify_all();
    size_t Ran = B.runIndices();
    std::unique_lock<std::mutex> Guard(Lock);
    close(B);
    B.Finished += Ran;
    BatchDone.wait(Guard, [&] { return B.Finished == B.N && B.Users == 0; });
  }

private:
  void work() {
    std::unique_lock<std::mutex> Guard(Lock);
    while (true) {
      WorkAvailable.wait(Guard, [&] { return Stop || !Open.empty(); });
      if (Stop)
        return;
      Batch &B = *Open.front();
      ++B.Users;
      Guard.unlock();
      size_t Ran = B.runIndices();
      Guard.lock();
      close(B);
      B.Finished += Ran;
      --B.Users;
      if (B.Finished == B.N && B.Users == 0)
        BatchDone.notify_all();
    }
  }

  // A batch whose indices are all claimed takes no more workers.
  void close(Batch &B) {
    auto It = llvm::find(Open, &B);
    if (It != Open.end())
      Open.erase(It);
  }

  std::mutex Lock;
  std::condition_variable WorkAvailable, BatchDone;
  std::deque<Batch *> Open;
  bool Stop = false;
  std::vector<llvm::thread> Workers;
};

} // namespace

// The pool for a -poseidon-threads value, made on first use and kept for the
// process, as the solve asks for many small batches; null when one thread is
// all it allows.
static Pool *poolFor(unsigned Threads) {
  static std::mutex Lock;
  static std::map<unsigned, std::unique_ptr<Pool>> Pools;
  std::lock_guard<std::mutex> Guard(Lock);
  auto [It, Fresh] = Pools.try_emplace(Threads);
  if (Fresh) {
    unsigned Count = hardware_concurrency(Threads).compute_thread_count();
    if (Count > 1)
      It->second = std::make_unique<Pool>(Count);
  }
  return It->second.get();
}

void forEachIndex(unsigned Threads, size_t N, function_ref<void(size_t)> Fn) {
  Pool *P = N > 1 ? poolFor(Threads) : nullptr;
  if (!P) {
    for (size_t I = 0; I < N; ++I)
      Fn(I);
    return;
  }
  Batch B(Fn, N);
  P->run(B);
}

void forEachBlock(unsigned Threads, size_t N, size_t Grain,
                  function_ref<void(size_t, size_t)> Fn) {
  const size_t Blocks = (N + Grain - 1) / Grain;
  forEachIndex(Threads, Blocks,
               [&](size_t B) { Fn(B * Grain, std::min(N, (B + 1) * Grain)); });
}

} // namespace poseidon

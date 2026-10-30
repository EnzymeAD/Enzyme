//===- LoopCheckpointing.cpp - Generic loop checkpointing -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Implementations/LoopCheckpointing.h"

int64_t mlir::enzyme::binomialProgress(int64_t n, int64_t s) {
  if (n <= 0)
    return 0;
  if (n == 1)
    return 1;
  if (s <= 1)
    return n;
  int64_t t = 0, beta = 1; // beta == C(s + t, t)
  while (beta < n) {
    ++t;
    beta = beta * (s + t) / t;
  }
  int64_t lo = n - beta * s / (s + t);
  int64_t hi = beta * t / (s + t);
  if (lo < 1)
    lo = 1;
  if (hi > n - 1)
    hi = n - 1;
  int64_t m = (lo + hi) / 2;
  int64_t cap = n - (s - 1); // leave a step for each slot still to be placed
  if (m > cap)
    m = cap;
  return m < 1 ? 1 : m;
}

/*
 * checkpoint_schedule.h - The checkpointing schedules both backends share.
 *
 * Part of the Enzyme Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * What a schedule tag and a budget mean, as plain arithmetic with no
 * dependencies, so that Enzyme-MLIR (which evaluates it at compile time or
 * emits it as IR), the reference schemes of enzyme/checkpoint.h (which the
 * LLVM driver and Enzyme-MLIR's run-time schedules walk) and the frontends
 * all read a loop's annotation the same way.
 *
 * The budget is the number of checkpoints that live through the loop's
 * reverse sweep: the slots of a binomial schedule, the segments of a periodic
 * one. It is never the length of a segment. A budget of 0 or less asks for the
 * default, a segment length of floor(sqrt(n)) for the periodic schedule and
 * floor(sqrt(n)) slots (at least 2) for the binomial ones.
 */

#ifndef ENZYME_CHECKPOINT_SCHEDULE_H
#define ENZYME_CHECKPOINT_SCHEDULE_H

#include <stddef.h>
#include <stdint.h>

/* The built-in schedules. The values are those of the loop annotations
 * (__enzyme_set_checkpointing(schedule, budget), the enzyme.checkpoint loop
 * metadata) and of __enzyme_ckpt_schedule_begin. */
enum {
  ENZYME_CKPT_SCHEDULE_NONE = 0,
  /* Segments whose starts are kept; a segment is replayed to reverse it. */
  ENZYME_CKPT_SCHEDULE_PERIODIC = 1,
  /* Griewank and Walther's Revolve, as Checkpointing.jl schedules it. */
  ENZYME_CKPT_SCHEDULE_REVOLVE = 2,
  /* Every step's state; a periodic schedule of one-step segments. */
  ENZYME_CKPT_SCHEDULE_STORE_ALL = 3,
  /* Enzyme-MLIR's binomial schedule: a stack of slots, each advance chosen
   * by enzyme_ckpt_binomial_progress. Its static and run-time forms take the
   * same snapshots. */
  ENZYME_CKPT_SCHEDULE_BINOMIAL = 4,
};

/* The schedule a loop annotation names, the frontends' spelling of the tags
 * above: "binomial", "revolve", "periodic" (or "regular", its older name),
 * "store_all" or "none". -1 for any other name. `name` holds `len` characters
 * and need not be null-terminated. */
static inline int64_t enzyme_ckpt_schedule_from_name(const char *name,
                                                     size_t len) {
  static const struct {
    const char *name;
    int64_t schedule;
  } names[] = {
      {"none", ENZYME_CKPT_SCHEDULE_NONE},
      {"periodic", ENZYME_CKPT_SCHEDULE_PERIODIC},
      {"regular", ENZYME_CKPT_SCHEDULE_PERIODIC},
      {"revolve", ENZYME_CKPT_SCHEDULE_REVOLVE},
      {"store_all", ENZYME_CKPT_SCHEDULE_STORE_ALL},
      {"binomial", ENZYME_CKPT_SCHEDULE_BINOMIAL},
  };
  size_t i, k;
  for (i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
    for (k = 0; k < len && names[i].name[k] == name[k]; k++)
      ;
    if (k == len && names[i].name[k] == '\0')
      return names[i].schedule;
  }
  return -1;
}

/* What a schedule asks the driver to do next. Values match Checkpointing.jl's
 * ActionFlag. */
enum {
  ENZYME_CKPT_NONE = 0,
  ENZYME_CKPT_STORE = 1,
  ENZYME_CKPT_RESTORE = 2,
  ENZYME_CKPT_FORWARD = 3,
  ENZYME_CKPT_FIRSTUTURN = 4,
  ENZYME_CKPT_UTURN = 5,
  ENZYME_CKPT_ERROR = 6,
  ENZYME_CKPT_DONE = 7
};

/* floor(sqrt(n)), for n >= 0. */
static inline int64_t enzyme_ckpt_isqrt(int64_t n) {
  int64_t r = 0;
  if (n <= 0)
    return 0;
  while ((r + 1) <= n / (r + 1))
    r++;
  return r;
}

/* The number of steps to advance from a checkpoint with `n` steps left to
 * reverse and `s` slots left to place, among those that attain the minimal
 * number of repetitions (Griewank's beta(s, t) = C(s + t, t)): the window
 * [n - beta(s-1, t), beta(s, t-1)] for the smallest t with beta(s, t) >= n,
 * clamped to [1, n-1], at its midpoint. It leaves a step for each of the s-1
 * slots still to be placed, and with one slot left it advances all of n, so
 * the advances from `s` slots sum to exactly n. */
static inline int64_t enzyme_ckpt_binomial_progress(int64_t n, int64_t s) {
  int64_t t = 0, beta = 1, lo, hi, m, cap;
  if (n <= 0)
    return 0;
  if (n == 1)
    return 1;
  if (s <= 1)
    return n;
  while (beta < n) { /* beta == C(s + t, t) */
    ++t;
    beta = beta * (s + t) / t; /* exact: C(s+t,t) from C(s+t-1,t-1) */
  }
  lo = n - beta * s / (s + t); /* n - beta(s-1, t) */
  hi = beta * t / (s + t);     /* beta(s, t-1) */
  if (lo < 1)
    lo = 1;
  if (hi > n - 1)
    hi = n - 1;
  m = (lo + hi) / 2;
  cap = n - (s - 1);
  if (m > cap)
    m = cap;
  return m < 1 ? 1 : m;
}

/* The slots a binomial schedule over n steps uses for `budget`: never more
 * than there are steps. */
static inline int64_t enzyme_ckpt_binomial_slots(int64_t n, int64_t budget) {
  if (budget <= 0) {
    budget = enzyme_ckpt_isqrt(n);
    if (budget < 2)
      budget = 2;
  }
  if (budget > n)
    budget = n;
  return budget < 1 ? 1 : budget;
}

/* A periodic schedule over n steps: `outer` segments of `inner` steps, then
 * one of `trailing` steps when they do not divide n. The segment length is
 * rounded up from n / budget, so that there are at most `budget` segments. */
typedef struct EnzymeCkptPeriodicSplit {
  int64_t inner;
  int64_t outer;
  int64_t trailing;
} EnzymeCkptPeriodicSplit;

static inline EnzymeCkptPeriodicSplit
enzyme_ckpt_periodic_split(int64_t n, int64_t budget) {
  EnzymeCkptPeriodicSplit s;
  if (n <= 0) {
    s.inner = 1;
    s.outer = 0;
    s.trailing = 0;
    return s;
  }
  if (budget > 0)
    s.inner = (n + budget - 1) / budget;
  else
    s.inner = enzyme_ckpt_isqrt(n);
  if (s.inner < 1)
    s.inner = 1;
  s.outer = n / s.inner;
  s.trailing = n % s.inner;
  return s;
}

/* The number of segments, the trailing one included. */
static inline int64_t enzyme_ckpt_periodic_segments(EnzymeCkptPeriodicSplit s) {
  return s.outer + (s.trailing > 0);
}

/* The first step of segment k, and its number of steps. */
static inline int64_t enzyme_ckpt_periodic_start(EnzymeCkptPeriodicSplit s,
                                                 int64_t k) {
  return k * s.inner;
}

static inline int64_t enzyme_ckpt_periodic_length(EnzymeCkptPeriodicSplit s,
                                                  int64_t k) {
  return k < s.outer ? s.inner : s.trailing;
}

#endif /* ENZYME_CHECKPOINT_SCHEDULE_H */

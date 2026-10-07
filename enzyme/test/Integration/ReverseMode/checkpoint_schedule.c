// clang-format off
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 -I%enzyme_include %s -S -emit-llvm -o - | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 -I%enzyme_include %s -S -emit-llvm -o - | %lli - ; fi
// clang-format on

// The schedules of enzyme/checkpoint.h against the arithmetic of
// enzyme/checkpoint_schedule.h, which Enzyme-MLIR's compiled schedules use.
//
// EnzymeCkptBinomial is replayed on a model of the driver (which step the
// state is at, which step each slot holds) next to a transcription of
// Enzyme-MLIR's cacheBinomial and reverseBinomial (LoopCheckpointing.h), for
// every number of steps up to 60 and budgets 1 to 9: every turn must start
// from the state before its step, the slots must hold the steps Enzyme-MLIR's
// slots hold when it reverses that step, and the steps run forward must be
// Enzyme-MLIR's, less the one the driver's first turn runs. EnzymeCkptPeriodic
// must store the starts of Enzyme-MLIR's segments, and all schedules must
// reverse every step once, last first.

#include <enzyme/checkpoint.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c, ...)                                                          \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
      if (++fails > 20)                                                        \
        exit(1);                                                               \
    }                                                                          \
  } while (0)

// Enzyme-MLIR's binomial schedule: for each step reversed, last first, the
// steps the slots hold, and the steps run forward in all.
typedef struct {
  int64_t slotsAt[64][16];
  int64_t sp[64];
  int64_t forward;
} MLIRBinomial;

static void mlir_binomial(int64_t n, int64_t budget, MLIRBinomial *m) {
  int64_t B = budget < n ? budget : n, step = 0, k, slot[16];
  m->forward = 0;
  // cacheBinomial: slot k at stepCtr, then advance.
  for (k = 0; k < B; k++) {
    int64_t rem = n - step, left = B - k;
    slot[k] = step;
    int64_t split = enzyme_ckpt_binomial_progress(rem, left < rem ? left : rem);
    m->forward += split;
    step += split;
  }
  // reverseBinomial: from the top slot, replay to cur - 1, placing slots.
  int64_t sp = B;
  for (int64_t cur = n; cur >= 1; cur--) {
    int64_t capo = sp - 1, pos = slot[capo], a = capo;
    while (pos + 1 < cur) {
      int64_t rem = cur - pos, left = B - a;
      if (left > rem)
        left = rem;
      slot[a] = pos;
      int64_t np = pos + enzyme_ckpt_binomial_progress(rem, left);
      int64_t ub = np == cur ? cur - 1 : np;
      m->forward += ub - pos;
      pos = np;
      a++;
    }
    sp = a;
    m->sp[cur] = sp;
    for (k = 0; k < sp; k++)
      m->slotsAt[cur][k] = slot[k];
  }
}

// Replay a scheme's actions on a model of the driver.
static void replay(const EnzymeCheckpointScheme *vt, int64_t n, int64_t budget,
                   const MLIRBinomial *m, int binomial) {
  EnzymeCkptConfig config = {budget, 0, NULL, 0, NULL};
  void *st = vt->init(&config, n, 0);
  EnzymeCkptAction a;
  int64_t state = 0, slot[128], forward = 0, next = n, stored[128];
  int nstored = 0;
  for (int i = 0; i < 128; i++)
    slot[i] = -1;
  for (;;) {
    vt->next_action(st, &a);
    if (a.flag == ENZYME_CKPT_DONE)
      break;
    switch (a.flag) {
    case ENZYME_CKPT_STORE:
      CHECK(a.iteration == state, "n=%lld b=%lld: store of step %lld at %lld",
            (long long)n, (long long)budget, (long long)a.iteration,
            (long long)state);
      slot[a.cpnum] = state;
      if (nstored < 128)
        stored[nstored++] = state;
      break;
    case ENZYME_CKPT_RESTORE:
      CHECK(slot[a.cpnum] >= 0, "n=%lld b=%lld: restore of an empty slot",
            (long long)n, (long long)budget);
      state = slot[a.cpnum];
      break;
    case ENZYME_CKPT_FORWARD:
      CHECK(a.startiteration == state,
            "n=%lld b=%lld: forward from %lld at %lld", (long long)n,
            (long long)budget, (long long)a.startiteration, (long long)state);
      forward += a.iteration - a.startiteration;
      state = a.iteration;
      break;
    case ENZYME_CKPT_FIRSTUTURN:
    case ENZYME_CKPT_UTURN:
      CHECK(a.iteration == next && state == next - 1,
            "n=%lld b=%lld: turn of step %lld at %lld, expected %lld",
            (long long)n, (long long)budget, (long long)a.iteration - 1,
            (long long)state, (long long)next - 1);
      CHECK((a.flag == ENZYME_CKPT_FIRSTUTURN) == (next == n),
            "n=%lld b=%lld: first turn is not the last step", (long long)n,
            (long long)budget);
      if (binomial) {
        // The slots below the stack top hold Enzyme-MLIR's steps.
        for (int64_t k = 0; k < m->sp[next]; k++)
          CHECK(slot[k] == m->slotsAt[next][k],
                "n=%lld b=%lld: reversing step %lld, slot %lld holds %lld, "
                "Enzyme-MLIR's %lld",
                (long long)n, (long long)budget, (long long)next - 1,
                (long long)k, (long long)slot[k],
                (long long)m->slotsAt[next][k]);
      }
      // After a turn the state is that before the step: the driver's turn
      // runs the step's derivative from it.
      next--;
      break;
    default:
      CHECK(0, "n=%lld b=%lld: action %d", (long long)n, (long long)budget,
            (int)a.flag);
      next = 0;
      break;
    }
    if (fails > 20)
      break;
  }
  CHECK(next == 0, "n=%lld b=%lld: %lld steps not reversed", (long long)n,
        (long long)budget, (long long)next);
  if (binomial)
    CHECK(forward + 1 == m->forward,
          "n=%lld b=%lld: %lld steps forward, Enzyme-MLIR %lld", (long long)n,
          (long long)budget, (long long)forward, (long long)m->forward);
  else if (vt == &EnzymeCkptPeriodic && n > 0) {
    // The forward sweep stores the start of each of Enzyme-MLIR's segments
    // but the last, which it then reverses.
    EnzymeCkptPeriodicSplit s = enzyme_ckpt_periodic_split(n, budget);
    int64_t segs = enzyme_ckpt_periodic_segments(s);
    for (int64_t k = 0; k + 1 < segs && k < nstored; k++)
      CHECK(stored[k] == enzyme_ckpt_periodic_start(s, k),
            "n=%lld b=%lld: segment %lld starts at %lld, Enzyme-MLIR's %lld",
            (long long)n, (long long)budget, (long long)k, (long long)stored[k],
            (long long)enzyme_ckpt_periodic_start(s, k));
  }
  vt->finalize(st);
}

int main() {
  MLIRBinomial m;
  // Spot values of the advance, as Enzyme-MLIR folds them.
  CHECK(enzyme_ckpt_binomial_progress(10, 3) == 4, "bp(10,3) = %lld",
        (long long)enzyme_ckpt_binomial_progress(10, 3));
  CHECK(enzyme_ckpt_binomial_progress(1, 5) == 1, "bp(1,5)");
  CHECK(enzyme_ckpt_binomial_progress(7, 1) == 7, "bp(7,1)");
  CHECK(enzyme_ckpt_isqrt(37) == 6 && enzyme_ckpt_isqrt(36) == 6 &&
            enzyme_ckpt_isqrt(35) == 5,
        "isqrt");
  // The frontends' names of the schedules.
  CHECK(enzyme_ckpt_schedule_from_name("binomial", 8) ==
                ENZYME_CKPT_SCHEDULE_BINOMIAL &&
            enzyme_ckpt_schedule_from_name("revolve", 7) ==
                ENZYME_CKPT_SCHEDULE_REVOLVE &&
            enzyme_ckpt_schedule_from_name("periodic", 8) ==
                ENZYME_CKPT_SCHEDULE_PERIODIC &&
            enzyme_ckpt_schedule_from_name("regular", 7) ==
                ENZYME_CKPT_SCHEDULE_PERIODIC &&
            enzyme_ckpt_schedule_from_name("store_all", 9) ==
                ENZYME_CKPT_SCHEDULE_STORE_ALL &&
            enzyme_ckpt_schedule_from_name("none", 4) ==
                ENZYME_CKPT_SCHEDULE_NONE,
        "schedule names");
  // Not null-terminated, a prefix, or longer than a name.
  CHECK(enzyme_ckpt_schedule_from_name("revolve,4", 7) ==
                ENZYME_CKPT_SCHEDULE_REVOLVE &&
            enzyme_ckpt_schedule_from_name("revolve", 6) == -1 &&
            enzyme_ckpt_schedule_from_name("revolves", 8) == -1 &&
            enzyme_ckpt_schedule_from_name("", 0) == -1,
        "schedule name bounds");
  {
    EnzymeCkptPeriodicSplit s = enzyme_ckpt_periodic_split(10, 0);
    CHECK(s.inner == 3 && s.outer == 3 && s.trailing == 1,
          "split(10, default)");
    s = enzyme_ckpt_periodic_split(37, 3);
    CHECK(s.inner == 13 && s.outer == 2 && s.trailing == 11, "split(37, 3)");
  }
  for (int64_t n = 1; n <= 60; n++)
    for (int64_t b = 1; b <= 9; b++) {
      mlir_binomial(n, b, &m);
      replay(&EnzymeCkptBinomial, n, b, &m, 1);
      replay(&EnzymeCkptRevolve, n, b, NULL, 0);
      replay(&EnzymeCkptPeriodic, n, b, NULL, 0);
    }
  for (int64_t n = 1; n <= 60; n++)
    replay(&EnzymeCkptPeriodic, n, 0, NULL, 0);
  if (fails)
    return 1;
  printf("ok\n");
  return 0;
}

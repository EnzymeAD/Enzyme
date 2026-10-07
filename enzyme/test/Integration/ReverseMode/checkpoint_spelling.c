// clang-format off
// RUN: if [ %llvmver -ge 17 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -std=c2x -DATTR -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -std=c11 -DGNU -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -x c++ -std=c++17 -DATTR -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 17 ]; then %clang -x c++ -std=c++17 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// clang-format on

// The spellings of a checkpointed loop, for every schedule of
// enzyme/checkpoint_schedule.h: [[enzyme::checkpoint("revolve", 3)]] (C23,
// C++), __attribute__((enzyme_checkpoint("revolve", 3))), and, for C
// compilers without C23 attributes, `#pragma enzyme checkpoint("revolve", 3)`
// on the line before the loop. The gradient and the value must be those of
// the same loop without the annotation.

#include <enzyme/checkpoint.h>
#include <math.h>
#include <stdio.h>

#define N 3

double __enzyme_autodiff(void *, ...);

#if defined(ATTR)
#define LOOP(...) [[enzyme::checkpoint(__VA_ARGS__)]] for
#define LOOP0 [[enzyme::checkpoint]] for
#elif defined(GNU)
#define LOOP(...) __attribute__((enzyme_checkpoint(__VA_ARGS__))) for
#define LOOP0 __attribute__((enzyme_checkpoint)) for
#else
#define PRAGMA(x) _Pragma(#x)
#define LOOP(...) PRAGMA(enzyme checkpoint(__VA_ARGS__)) for
#define LOOP0 PRAGMA(enzyme checkpoint) for
#endif

double state[N];

#define BODY                                                                   \
  for (int k = 0; k < N; k++)                                                  \
    state[k] = sin(state[k]) * (1.0 + 0.1 * k) + 0.05 * state[(k + 1) % N];

#define DEF(name, ...)                                                         \
  double name(double x, long n) {                                              \
    for (int k = 0; k < N; k++)                                                \
      state[k] = x * (k + 1);                                                  \
    LOOP(__VA_ARGS__)(long i = 0; i < n; i++) { BODY }                         \
    return state[0] + state[1] * state[2];                                     \
  }

DEF(binomial, "binomial", 2)
DEF(revolve, "revolve", 2)
DEF(periodic, "periodic", 3)
DEF(store_all, "store_all")

double dflt(double x, long n) {
  for (int k = 0; k < N; k++)
    state[k] = x * (k + 1);
  LOOP0(long i = 0; i < n; i++) { BODY }
  return state[0] + state[1] * state[2];
}

double plain(double x, long n) {
  for (int k = 0; k < N; k++)
    state[k] = x * (k + 1);
  for (long i = 0; i < n; i++) {
    BODY
  }
  return state[0] + state[1] * state[2];
}

static int failures = 0;

#define CHECK(name, n, want)                                                   \
  do {                                                                         \
    if (name(0.4, n) != plain(0.4, n)) {                                       \
      printf(#name " n=%ld: primal differs\n", n);                             \
      failures++;                                                              \
    }                                                                          \
    double got = __enzyme_autodiff((void *)name, 0.4, n);                      \
    if (fabs(got - want) > 1e-12 * (1 + fabs(want))) {                         \
      printf(#name " n=%ld: gradient %.17g, expected %.17g\n", n, got, want);  \
      failures++;                                                              \
    }                                                                          \
  } while (0)

int main(void) {
  long steps[] = {1, 2, 9, 20};
  for (unsigned a = 0; a < 4; a++) {
    long n = steps[a];
    double want = __enzyme_autodiff((void *)plain, 0.4, n);
    CHECK(binomial, n, want);
    CHECK(revolve, n, want);
    CHECK(periodic, n, want);
    CHECK(store_all, n, want);
    CHECK(dflt, n, want);
  }
  if (failures)
    return 1;
  printf("ok\n");
  return 0;
}

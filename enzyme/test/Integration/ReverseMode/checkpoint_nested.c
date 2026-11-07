// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// Two-level checkpointing in the style of MITgcm's TAF adjoint: the outer
// loop runs over segments of `inner` steps and keeps its snapshots in files,
// each segment is itself a checkpointed loop with snapshots in memory. The
// gradient must equal that of the plain loop.

#define _GNU_SOURCE
#include <enzyme/checkpoint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 6

static double glob[N] = {0};

void __enzyme_autodiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

__attribute__((noinline)) static void step(int64_t i, double *u) {
  double tmp[N];
  for (int k = 0; k < N; k++)
    tmp[k] = u[k] + 0.2 * (u[(k + 1) % N] - u[k]) +
             0.05 * cos(glob[k] + 0.01 * (double)i);
  for (int k = 0; k < N; k++) {
    glob[k] = 0.8 * glob[k] + 0.2 * tmp[k] * u[k];
    u[k] = tmp[k];
  }
}

typedef struct {
  int64_t inner;
  EnzymeCkptConfig *inner_config;
} Levels;

__attribute__((noinline)) static void segment(int64_t k, double *u,
                                              Levels *levels) {
  __enzyme_checkpoint_for((void *)step, k * levels->inner, levels->inner,
                          enzyme_scheme, &EnzymeCkptPeriodic,
                          levels->inner_config, enzyme_checkpoint_region, u,
                          (int64_t)(N * sizeof(double)), u);
}

static double loss(const double *u) {
  double s = 0;
  for (int k = 0; k < N; k++)
    s += u[k] * u[k] * u[k] + glob[k];
  return s;
}

static double plain(double *u, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    step(i, u);
  return loss(u);
}

static double two_level(double *u, int64_t outer, Levels *levels,
                        EnzymeCkptConfig *outer_config) {
  __enzyme_checkpoint_for((void *)segment, 0, outer, enzyme_scheme,
                          &EnzymeCkptPeriodic, outer_config,
                          enzyme_checkpoint_region, u,
                          (int64_t)(N * sizeof(double)), u, levels);
  return loss(u);
}

static void init(double *u) {
  for (int k = 0; k < N; k++) {
    u[k] = 0.5 + 0.1 * k;
    glob[k] = 0.1 * k;
  }
}

typedef struct {
  double du[N];
  EnzymeCkptStats stats;
} Result;

// glob's shadow is a global, which keeps what earlier gradients accumulated
// in it; each gradient is computed in a child process.
static Result gradient(int64_t outer, int64_t inner, int64_t outer_snaps,
                       int64_t inner_snaps) {
  Result r;
  int fd[2];
  if (pipe(fd) != 0)
    abort();
  pid_t pid = fork();
  if (pid == 0) {
    double u[N];
    memset(&r, 0, sizeof(r));
    init(u);
    if (inner == 0) {
      __enzyme_autodiff((void *)plain, enzyme_dup, u, r.du, enzyme_const,
                        outer);
    } else {
      // Every outer snapshot goes to a file, as TAF's tapelev files do.
      EnzymeCkptConfig outer_config = {outer_snaps, 0, "/tmp", 0, &r.stats};
      EnzymeCkptConfig inner_config = {inner_snaps, 0, NULL, 0, NULL};
      Levels levels = {inner, &inner_config};
      __enzyme_autodiff((void *)two_level, enzyme_dup, u, r.du, enzyme_const,
                        outer, enzyme_const, &levels, enzyme_const,
                        &outer_config);
    }
    if (write(fd[1], &r, sizeof(r)) != sizeof(r))
      abort();
    _exit(0);
  }
  close(fd[1]);
  if (read(fd[0], &r, sizeof(r)) != sizeof(r))
    abort();
  close(fd[0]);
  waitpid(pid, NULL, 0);
  return r;
}

int main(void) {
  // (outer steps, inner steps, outer segments, inner segments)
  int64_t cases[][4] = {{1, 1, 1, 1}, {2, 3, 1, 2}, {3, 4, 2, 2},
                        {5, 6, 2, 3}, {6, 5, 6, 5}};
  int failures = 0;
  for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    int64_t outer = cases[c][0], inner = cases[c][1];
    Result ref = gradient(outer * inner, 0, 0, 0);
    Result r = gradient(outer, inner, cases[c][2], cases[c][3]);
    for (int k = 0; k < N; k++) {
      double err = fabs(r.du[k] - ref.du[k]) / (fabs(ref.du[k]) + 1e-300);
      if (!(err < 1e-12)) {
        printf("outer=%lld inner=%lld: du[%d] = %.17g, expected %.17g\n",
               (long long)outer, (long long)inner, k, r.du[k], ref.du[k]);
        failures++;
      }
    }
    if (r.stats.taped_steps != outer) {
      printf("outer=%lld inner=%lld: %lld outer steps taped\n",
             (long long)outer, (long long)inner,
             (long long)r.stats.taped_steps);
      failures++;
    }
  }
  if (failures) {
    printf("%d failures\n", failures);
    return 1;
  }
  printf("ok\n");
  return 0;
}

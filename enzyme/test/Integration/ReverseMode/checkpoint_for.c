// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// A time loop whose state lives both in an argument and in a global, as in
// Fortran codes that keep their state in COMMON blocks. Its gradient through
// __enzyme_checkpoint_for, for every reference scheme, number of steps and
// number of snapshots, must equal the gradient of the plain loop.

#define _GNU_SOURCE
#include <enzyme/checkpoint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 8

// Written by every step: a snapshot must hold it.
static double glob[N] = {0};
// Only read: a snapshot need not hold it.
static double kappa = 0.1;

void __enzyme_autodiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

__attribute__((noinline)) static void step(int64_t i, double *u) {
  double tmp[N];
  for (int k = 0; k < N; k++)
    tmp[k] = u[k] + kappa * (u[(k + 1) % N] - 2 * u[k] + u[(k + N - 1) % N]) +
             0.01 * sin(glob[k] + 0.1 * (double)i);
  for (int k = 0; k < N; k++) {
    glob[k] = 0.9 * glob[k] + 0.1 * tmp[k] * tmp[k];
    u[k] = tmp[k];
  }
}

static double loss(const double *u) {
  double s = 0;
  for (int k = 0; k < N; k++)
    s += u[k] * u[k] + glob[k];
  return s;
}

static double plain(double *u, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    step(i, u);
  return loss(u);
}

static double checkpointed(double *u, int64_t n,
                           const EnzymeCheckpointScheme *scheme,
                           EnzymeCkptConfig *config) {
  __enzyme_checkpoint_for((void *)step, 0, n, enzyme_scheme, scheme, config,
                          enzyme_checkpoint_region, u,
                          (int64_t)(N * sizeof(double)), u);
  return loss(u);
}

static void init(double *u) {
  for (int k = 0; k < N; k++) {
    u[k] = 1.0 + 0.5 * sin((double)k);
    glob[k] = 0.25 * k;
  }
}

static const char *names[] = {"revolve", "periodic", "store_all"};

typedef struct {
  double du[N];
  // The state after the gradient: the reverse pass must leave it as the
  // forward pass did.
  double u[N], glob[N];
  EnzymeCkptStats stats;
} Result;

// glob's shadow is a global too, and keeps what earlier gradients
// accumulated in it. Each gradient is computed in a child process, which
// starts with a zero shadow.
static Result gradient(int64_t n, const EnzymeCheckpointScheme *scheme,
                       int64_t snaps) {
  Result r;
  int fd[2];
  if (pipe(fd) != 0)
    abort();
  pid_t pid = fork();
  if (pid == 0) {
    double u[N];
    EnzymeCkptConfig config = {snaps, 0, NULL, 0, &r.stats};
    memset(&r, 0, sizeof(r));
    init(u);
    if (scheme)
      __enzyme_autodiff((void *)checkpointed, enzyme_dup, u, r.du,
                        enzyme_const, n, enzyme_const, scheme, enzyme_const,
                        &config);
    else
      __enzyme_autodiff((void *)plain, enzyme_dup, u, r.du, enzyme_const, n);
    memcpy(r.u, u, sizeof(u));
    memcpy(r.glob, glob, sizeof(glob));
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
  const EnzymeCheckpointScheme *schemes[] = {
      &EnzymeCkptRevolve, &EnzymeCkptPeriodic, &EnzymeCkptStoreAll};
  int64_t steps[] = {0, 1, 2, 7, 20, 100};
  int64_t snaps[] = {1, 2, 3, 5};
  int failures = 0;

  for (unsigned ns = 0; ns < sizeof(steps) / sizeof(steps[0]); ns++) {
    int64_t n = steps[ns];
    Result ref = gradient(n, NULL, 0);
    for (unsigned s = 0; s < 3; s++)
      for (unsigned c = 0; c < sizeof(snaps) / sizeof(snaps[0]); c++) {
        Result r = gradient(n, schemes[s], snaps[c]);
        for (int k = 0; k < N; k++) {
          double err =
              fabs(r.du[k] - ref.du[k]) / (fabs(ref.du[k]) + 1e-300);
          if (!(err < 1e-12)) {
            printf("%s n=%lld snaps=%lld: du[%d] = %.17g, expected %.17g\n",
                   names[s], (long long)n, (long long)snaps[c], k, r.du[k],
                   ref.du[k]);
            failures++;
          }
        }
        if (memcmp(r.u, ref.u, sizeof(r.u)) ||
            memcmp(r.glob, ref.glob, sizeof(r.glob))) {
          printf("%s n=%lld snaps=%lld: state after the gradient differs\n",
                 names[s], (long long)n, (long long)snaps[c]);
          failures++;
        }
        // Every step is taped exactly once.
        if (r.stats.taped_steps != n) {
          printf("%s n=%lld snaps=%lld: %lld taped steps\n", names[s],
                 (long long)n, (long long)snaps[c],
                 (long long)r.stats.taped_steps);
          failures++;
        }
        // Revolve keeps at most `snaps` snapshots.
        if (s == 0 && r.stats.max_slots > snaps[c]) {
          printf("revolve n=%lld snaps=%lld: %lld slots\n", (long long)n,
                 (long long)snaps[c], (long long)r.stats.max_slots);
          failures++;
        }
      }
  }
  if (failures) {
    printf("%d failures\n", failures);
    return 1;
  }
  printf("ok\n");
  return 0;
}

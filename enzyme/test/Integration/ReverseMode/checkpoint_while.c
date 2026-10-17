// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// A loop run until its state crosses a threshold, through
// __enzyme_checkpoint_while: the number of steps is only known when it ends.
// The gradient, and the state after it, must equal those of the plain loop.

#define _GNU_SOURCE
#include <enzyme/checkpoint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define N 4

void __enzyme_autodiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

// Grows u until its first entry passes `limit`: whether to go on.
__attribute__((noinline)) static int step(int64_t i, double *u,
                                          double *limit) {
  double tmp[N];
  for (int k = 0; k < N; k++)
    tmp[k] = u[k] + 0.1 * u[(k + 1) % N] + 0.01 * sin(u[k] + 0.1 * (double)i);
  for (int k = 0; k < N; k++)
    u[k] = tmp[k];
  return u[0] < *limit;
}

static double loss(const double *u) {
  double s = 0;
  for (int k = 0; k < N; k++)
    s += u[k] * u[k];
  return s;
}

static double plain(double *u, double *limit) {
  int64_t i = 0;
  while (step(i++, u, limit))
    ;
  return loss(u);
}

static double checkpointed(double *u, double *limit,
                           EnzymeCkptConfig *config) {
  __enzyme_checkpoint_while((void *)step, enzyme_scheme, &EnzymeCkptStoreAll,
                            config, enzyme_checkpoint_region, u,
                            (int64_t)(N * sizeof(double)), u, limit);
  return loss(u);
}

typedef struct {
  double du[N], u[N];
  int64_t steps;
} Result;

static void init(double *u) {
  for (int k = 0; k < N; k++)
    u[k] = 1.0 + 0.1 * k;
}

// Each gradient in a child process, as in checkpoint_for.c.
static Result gradient(double limit, int ckpt) {
  Result r;
  int fd[2];
  if (pipe(fd) != 0)
    abort();
  pid_t pid = fork();
  if (pid == 0) {
    double u[N];
    EnzymeCkptStats stats;
    EnzymeCkptConfig config = {1, 0, NULL, 0, &stats};
    memset(&r, 0, sizeof(r));
    memset(&stats, 0, sizeof(stats));
    init(u);
    if (ckpt) {
      __enzyme_autodiff((void *)checkpointed, enzyme_dup, u, r.du,
                        enzyme_const, &limit, enzyme_const, &config);
      r.steps = stats.taped_steps;
    } else {
      __enzyme_autodiff((void *)plain, enzyme_dup, u, r.du, enzyme_const,
                        &limit);
    }
    memcpy(r.u, u, sizeof(u));
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
  // The first entry starts at 1.0: these limits end the loop after the first
  // step, a few steps, and many.
  double limits[] = {0.5, 1.2, 2.0, 8.0};
  int failures = 0;
  for (unsigned l = 0; l < sizeof(limits) / sizeof(limits[0]); l++) {
    Result ref = gradient(limits[l], 0);
    Result r = gradient(limits[l], 1);
    for (int k = 0; k < N; k++) {
      double err = fabs(r.du[k] - ref.du[k]) / (fabs(ref.du[k]) + 1e-300);
      if (!(err < 1e-12)) {
        printf("limit %g: du[%d] = %.17g, expected %.17g\n", limits[l], k,
               r.du[k], ref.du[k]);
        failures++;
      }
    }
    if (memcmp(r.u, ref.u, sizeof(r.u))) {
      printf("limit %g: state after the gradient differs\n", limits[l]);
      failures++;
    }
    if (r.steps < 1) {
      printf("limit %g: no step taped\n", limits[l]);
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

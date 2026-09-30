// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// Vector mode through a checkpointed loop. Reverse mode with W adjoint seeds
// gives W rows of the Jacobian of the outputs in one sweep: one schedule, one
// set of snapshots and recomputations for all of them. Each row must equal the
// gradient of that output through the plain loop. Forward mode with W
// tangents must equal the plain loop's too.

#include "../test_utils.h"
#include <enzyme/checkpoint.h>
#include <math.h>
#include <string.h>

#define N 4
#define W 3

void __enzyme_autodiff(void *, ...);
void __enzyme_fwddiff(void *, ...);
extern int enzyme_dup, enzyme_const, enzyme_width;

__attribute__((noinline)) static void step(int64_t i, double *u) {
  double tmp[N];
  for (int k = 0; k < N; k++)
    tmp[k] = u[k] + 0.02 * u[(k + 1) % N] * u[k] +
             0.01 * sin(u[k] + 0.1 * (double)i);
  for (int k = 0; k < N; k++)
    u[k] = tmp[k];
}

// W outputs of the final state.
static void outputs(const double *u, double *out) {
  double s = 0, p = 1, c = 0;
  for (int k = 0; k < N; k++) {
    s += u[k] * u[k];
    p *= 1.0 + 0.1 * u[k];
    c += cos(u[k]) * (double)(k + 1);
  }
  out[0] = s;
  out[1] = p;
  out[2] = c;
}

static void plain(double *u, double *out, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    step(i, u);
  outputs(u, out);
}

static void ckpt(double *u, double *out, int64_t n, EnzymeCkptConfig *c) {
  __enzyme_checkpoint_for((void *)step, 0, n, enzyme_scheme, &EnzymeCkptRevolve,
                          c, enzyme_checkpoint_region, u,
                          (int64_t)(N * sizeof(double)), u);
  outputs(u, out);
}

// Relative comparison: the outputs grow with the number of steps.
static int close(double a, double b) {
  return fabs(a - b) <= 1e-10 * (fabs(b) + 1e-12);
}

static int failures = 0;
static void check(double got, double want, const char *what, int64_t n,
                  int j, int k) {
  if (!close(got, want)) {
    printf("n=%lld %s[%d][%d] = %.17g, expected %.17g\n", (long long)n, what,
           j, k, got, want);
    failures++;
  }
}

static void init(double *u) {
  for (int k = 0; k < N; k++)
    u[k] = 0.5 + 0.2 * k;
}

int main(void) {
  int64_t steps[] = {1, 2, 7, 20};
  int64_t snaps[] = {1, 2, 3};
  for (unsigned a = 0; a < sizeof(steps) / sizeof(steps[0]); a++) {
    int64_t n = steps[a];
    // Row j: the gradient of output j through the plain loop.
    double want[W][N];
    for (int j = 0; j < W; j++) {
      double u[N], du[N] = {0}, out[W], dout[W] = {0};
      init(u);
      dout[j] = 1.0;
      __enzyme_autodiff((void *)plain, enzyme_dup, u, du, enzyme_dup, out,
                        dout, enzyme_const, n);
      memcpy(want[j], du, sizeof(du));
    }
    for (unsigned b = 0; b < sizeof(snaps) / sizeof(snaps[0]); b++) {
      EnzymeCkptStats stats = {0};
      EnzymeCkptConfig config = {snaps[b], 0, NULL, 0, &stats};
      double u[N], out[W];
      double du[W][N] = {{0}}, dout[W][W] = {{0}};
      init(u);
      for (int j = 0; j < W; j++)
        dout[j][j] = 1.0;
      __enzyme_autodiff((void *)ckpt, enzyme_width, W, enzyme_dup, u, du[0],
                        du[1], du[2], enzyme_dup, out, dout[0], dout[1],
                        dout[2], enzyme_const, n, enzyme_const, &config);
      for (int j = 0; j < W; j++)
        for (int k = 0; k < N; k++)
          check(du[j][k], want[j][k], "du", n, j, k);
      // One sweep for all W rows: each step differentiated once, and never
      // more snapshots than asked for.
      if (stats.taped_steps != n || (n > 1 && stats.stores == 0)) {
        printf("n=%lld snaps=%lld: %lld steps differentiated, %lld stores\n",
               (long long)n, (long long)snaps[b], (long long)stats.taped_steps,
               (long long)stats.stores);
        failures++;
      }
      if (stats.max_slots > snaps[b]) {
        printf("n=%lld snaps=%lld: %lld slots\n", (long long)n,
               (long long)snaps[b], (long long)stats.max_slots);
        return 1;
      }
    }

    // Forward mode, W tangents at once.
    double v[W][N];
    for (int j = 0; j < W; j++)
      for (int k = 0; k < N; k++)
        v[j][k] = (j == k) ? 1.0 : 0.1 * (double)(j + k);
    double wantT[W][W];
    for (int j = 0; j < W; j++) {
      double u[N], du[N], out[W], dout[W] = {0};
      init(u);
      memcpy(du, v[j], sizeof(du));
      __enzyme_fwddiff((void *)plain, enzyme_dup, u, du, enzyme_dup, out, dout,
                       enzyme_const, n);
      memcpy(wantT[j], dout, sizeof(dout));
    }
    {
      EnzymeCkptConfig config = {2, 0, NULL, 0, NULL};
      double u[N], out[W], du[W][N], dout[W][W] = {{0}};
      init(u);
      memcpy(du, v, sizeof(du));
      __enzyme_fwddiff((void *)ckpt, enzyme_width, W, enzyme_dup, u, du[0],
                       du[1], du[2], enzyme_dup, out, dout[0], dout[1],
                       dout[2], enzyme_const, n, enzyme_const, &config);
      for (int j = 0; j < W; j++)
        for (int k = 0; k < W; k++)
          check(dout[j][k], wantT[j][k], "dout", n, j, k);
    }
  }
  if (failures)
    return 1;
  printf("ok\n");
  return 0;
}

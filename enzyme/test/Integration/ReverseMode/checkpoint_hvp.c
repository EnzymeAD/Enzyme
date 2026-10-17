// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O0 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O1 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O2 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi
// RUN: if [ %llvmver -ge 15 ]; then %clang -std=c11 -O3 %s -S -emit-llvm -o - %newLoadClangEnzyme | %lli - ; fi

// Hessian-vector products through a checkpointed loop, forward mode over
// reverse mode: the forward-mode derivative of a gradient that runs
// __enzyme_checkpoint_for (and __enzyme_checkpoint_while). The schedule is
// that of the gradient, run on the tangents of the steps, with snapshots that
// hold the state and its tangent. For every reference scheme, number of steps
// and number of snapshots, the product must equal forward over reverse of the
// plain loop, and central differences of its gradient. The state after it,
// and its tangent, must be those of the plain loop.

#include <enzyme/checkpoint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define N 4

void __enzyme_autodiff(void *, ...);
void __enzyme_fwddiff(void *, ...);
extern int enzyme_dup, enzyme_const, enzyme_dupv, enzyme_width;

// Written by every step, and read by it: a snapshot must hold it. It has no
// derivative, so its snapshot needs no tangent. (A global with a derivative
// would share its one shadow between the tangent and the adjoint, which
// Enzyme does not support in forward over reverse mode yet.)
static int64_t phase;

// u is the state, p parameters the step only reads.
__attribute__((noinline)) static void step(int64_t i, double *u,
                                           const double *p) {
  double tmp[N];
  phase = (7 * phase + 3) % 11;
  for (int k = 0; k < N; k++)
    tmp[k] = u[k] + 0.2 + 0.1 * p[k] * sin(u[(k + 1) % N] * u[k]) +
             0.01 * sin(u[k] * p[(k + 2) % N] + 0.1 * (double)(i + phase));
  for (int k = 0; k < N; k++)
    u[k] = tmp[k];
}

static double loss(const double *u, const double *p) {
  double s = 0;
  for (int k = 0; k < N; k++)
    s += u[k] * u[k] * u[k] + p[k] * u[k];
  return s;
}

static double plain(double *u, double *p, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    step(i, u, p);
  return loss(u, p);
}

static double checkpointed(double *u, double *p, int64_t n,
                           const EnzymeCheckpointScheme *scheme,
                           EnzymeCkptConfig *config) {
  __enzyme_checkpoint_for((void *)step, 0, n, enzyme_scheme, scheme, config,
                          enzyme_checkpoint_region, u,
                          (int64_t)(N * sizeof(double)), u, p);
  return loss(u, p);
}

// The while loop: steps until the first entry passes `limit`.
__attribute__((noinline)) static int wstep(int64_t i, double *u,
                                           const double *p,
                                           const double *limit) {
  step(i, u, p);
  return u[0] < *limit;
}

static double wplain(double *u, double *p, double *limit) {
  int64_t i = 0;
  while (wstep(i++, u, p, limit))
    ;
  return loss(u, p);
}

static double wcheckpointed(double *u, double *p, double *limit,
                            EnzymeCkptConfig *config) {
  __enzyme_checkpoint_while((void *)wstep, enzyme_scheme, &EnzymeCkptStoreAll,
                            config, enzyme_checkpoint_region, u,
                            (int64_t)(N * sizeof(double)), u, p, limit);
  return loss(u, p);
}

// The gradients with respect to u and p.
__attribute__((noinline)) static void
grad(double *u, double *du, double *p, double *dp, int64_t n,
     const EnzymeCheckpointScheme *scheme, EnzymeCkptConfig *config) {
  if (scheme)
    __enzyme_autodiff((void *)checkpointed, enzyme_dup, u, du, enzyme_dup, p,
                      dp, enzyme_const, n, enzyme_const, scheme, enzyme_const,
                      config);
  else
    __enzyme_autodiff((void *)plain, enzyme_dup, u, du, enzyme_dup, p, dp,
                      enzyme_const, n);
}

__attribute__((noinline)) static void wgrad(double *u, double *du, double *p,
                                            double *dp, double *limit,
                                            EnzymeCkptConfig *config) {
  if (config)
    __enzyme_autodiff((void *)wcheckpointed, enzyme_dup, u, du, enzyme_dup, p,
                      dp, enzyme_const, limit, enzyme_const, config);
  else
    __enzyme_autodiff((void *)wplain, enzyme_dup, u, du, enzyme_dup, p, dp,
                      enzyme_const, limit);
}

static const double x0[N] = {0.5, 0.7, 0.9, 1.1};
static const double p0[N] = {1.0, 0.8, 1.2, 0.9};
static const double vx[N] = {1, -1, 0.5, 2};
static const double vp[N] = {0.3, 0.2, -0.4, 0.1};

typedef struct {
  // The gradient and the Hessian-vector product, for u then p.
  double g[2 * N], hv[2 * N];
  // The state after, and its tangent.
  double u[N], du[N];
  int64_t phase;
  EnzymeCkptStats stats;
} Result;

// Forward over reverse; `limit` selects the while loop.
static Result hvp(int64_t n, double *limit,
                  const EnzymeCheckpointScheme *scheme, int64_t snaps) {
  Result r;
  memset(&r, 0, sizeof(r));
  double u[N], p[N], pd[N];
  EnzymeCkptConfig config = {snaps, 0, NULL, 0, &r.stats};
  for (int k = 0; k < N; k++) {
    u[k] = x0[k];
    r.du[k] = vx[k];
    p[k] = p0[k];
    pd[k] = vp[k];
  }
  phase = 0;
  if (limit)
    __enzyme_fwddiff((void *)wgrad, enzyme_dup, u, r.du, enzyme_dup, r.g,
                     r.hv, enzyme_dup, p, pd, enzyme_dup, r.g + N, r.hv + N,
                     enzyme_const, limit, enzyme_const,
                     scheme ? &config : NULL);
  else
    __enzyme_fwddiff((void *)grad, enzyme_dup, u, r.du, enzyme_dup, r.g, r.hv,
                     enzyme_dup, p, pd, enzyme_dup, r.g + N, r.hv + N,
                     enzyme_const, n, enzyme_const, scheme, enzyme_const,
                     &config);
  memcpy(r.u, u, sizeof(u));
  r.phase = phase;
  return r;
}

// Central differences of the gradient of the plain loop along (vx, vp).
static void fd(int64_t n, double *limit, double *out) {
  const double h = 1e-6;
  double g[2][2 * N];
  for (int s = 0; s < 2; s++) {
    double u[N], p[N];
    double sign = s ? -1 : 1;
    for (int k = 0; k < N; k++) {
      u[k] = x0[k] + sign * h * vx[k];
      p[k] = p0[k] + sign * h * vp[k];
    }
    memset(g[s], 0, sizeof(g[s]));
    phase = 0;
    if (limit)
      wgrad(u, g[s], p, g[s] + N, limit, NULL);
    else
      grad(u, g[s], p, g[s] + N, n, NULL, NULL);
  }
  for (int k = 0; k < 2 * N; k++)
    out[k] = (g[0][k] - g[1][k]) / (2 * h);
}

// Two products at once, in forward vector mode: along (vx, vp) and (vp, vx).
typedef struct {
  double g[2 * N], hv[2][2 * N], u[N];
  EnzymeCkptStats stats;
} Batch;

static Batch hvp2(int64_t n, const EnzymeCheckpointScheme *scheme,
                  int64_t snaps) {
  Batch r;
  memset(&r, 0, sizeof(r));
  double u[N], p[N], ud[2][N], pd[2][N], gu[2][N], gp[2][N];
  EnzymeCkptConfig config = {snaps, 0, NULL, 0, &r.stats};
  memset(gu, 0, sizeof(gu));
  memset(gp, 0, sizeof(gp));
  for (int k = 0; k < N; k++) {
    u[k] = x0[k];
    p[k] = p0[k];
    ud[0][k] = pd[1][k] = vx[k];
    pd[0][k] = ud[1][k] = vp[k];
  }
  phase = 0;
  __enzyme_fwddiff((void *)grad, enzyme_width, 2, enzyme_dupv,
                   sizeof(ud[0]), u, ud, enzyme_dupv, sizeof(gu[0]), r.g, gu,
                   enzyme_dupv, sizeof(pd[0]), p, pd, enzyme_dupv,
                   sizeof(gp[0]), r.g + N, gp, enzyme_const, n, enzyme_const,
                   scheme, enzyme_const, &config);
  for (int w = 0; w < 2; w++)
    for (int k = 0; k < N; k++) {
      r.hv[w][k] = gu[w][k];
      r.hv[w][N + k] = gp[w][k];
    }
  memcpy(r.u, u, sizeof(u));
  return r;
}

static int failures = 0;

static void check(const char *what, int64_t n, int64_t snaps, Result *r,
                  Result *ref, double *fdv) {
  for (int k = 0; k < 2 * N; k++) {
    double err = fabs(r->hv[k] - ref->hv[k]) / (fabs(ref->hv[k]) + 1e-300);
    double ferr = fabs(r->hv[k] - fdv[k]) / (fabs(fdv[k]) + 1e-3);
    double gerr = fabs(r->g[k] - ref->g[k]) / (fabs(ref->g[k]) + 1e-300);
    if (!(err < 1e-10) || !(gerr < 1e-12) || !(ferr < 1e-5)) {
      printf("%s n=%lld snaps=%lld: [%d] g %.17g (want %.17g) Hv %.17g "
             "(want %.17g, fd %.17g)\n",
             what, (long long)n, (long long)snaps, k, r->g[k], ref->g[k],
             r->hv[k], ref->hv[k], fdv[k]);
      failures++;
    }
  }
  if (memcmp(r->u, ref->u, sizeof(r->u)) || r->phase != ref->phase) {
    printf("%s n=%lld snaps=%lld: state after differs\n", what, (long long)n,
           (long long)snaps);
    failures++;
  }
  for (int k = 0; k < N; k++)
    if (!(fabs(r->du[k] - ref->du[k]) <= 1e-12 * fabs(ref->du[k]))) {
      printf("%s n=%lld snaps=%lld: tangent after differs\n", what,
             (long long)n, (long long)snaps);
      failures++;
      break;
    }
  if (r->stats.taped_steps != n) {
    printf("%s n=%lld snaps=%lld: %lld turns\n", what, (long long)n,
           (long long)snaps, (long long)r->stats.taped_steps);
    failures++;
  }
}

int main(void) {
  const EnzymeCheckpointScheme *schemes[] = {
      &EnzymeCkptRevolve, &EnzymeCkptPeriodic, &EnzymeCkptStoreAll};
  const char *names[] = {"revolve", "periodic", "store_all"};
  int64_t steps[] = {1, 2, 7, 20};
  int64_t snaps[] = {1, 2, 3, 5};

  for (unsigned ns = 0; ns < sizeof(steps) / sizeof(steps[0]); ns++) {
    int64_t n = steps[ns];
    Result ref = hvp(n, NULL, NULL, 0);
    double fdv[2 * N];
    fd(n, NULL, fdv);
    for (unsigned s = 0; s < 3; s++)
      for (unsigned c = 0; c < sizeof(snaps) / sizeof(snaps[0]); c++) {
        Result r = hvp(n, NULL, schemes[s], snaps[c]);
        check(names[s], n, snaps[c], &r, &ref, fdv);
        // Revolve keeps at most `snaps` snapshots.
        if (s == 0 && r.stats.max_slots > snaps[c]) {
          printf("revolve n=%lld snaps=%lld: %lld slots\n", (long long)n,
                 (long long)snaps[c], (long long)r.stats.max_slots);
          failures++;
        }
      }
  }

  for (unsigned ns = 0; ns < sizeof(steps) / sizeof(steps[0]); ns++) {
    int64_t n = steps[ns];
    Batch ref = hvp2(n, NULL, 0);
    Batch r = hvp2(n, &EnzymeCkptRevolve, 2);
    for (int w = 0; w < 2; w++)
      for (int k = 0; k < 2 * N; k++) {
        double err = fabs(r.hv[w][k] - ref.hv[w][k]) /
                     (fabs(ref.hv[w][k]) + 1e-300);
        if (!(err < 1e-10)) {
          printf("width 2 n=%lld: Hv[%d][%d] %.17g, expected %.17g\n",
                 (long long)n, w, k, r.hv[w][k], ref.hv[w][k]);
          failures++;
        }
      }
    if (memcmp(r.u, ref.u, sizeof(r.u)) || r.stats.taped_steps != n) {
      printf("width 2 n=%lld: state after differs, or %lld turns\n",
             (long long)n, (long long)r.stats.taped_steps);
      failures++;
    }
  }

  // The first entry starts at 0.5: these limits end the while loop after the
  // first step, a few steps, and many.
  double limits[] = {0.1, 1.0, 1.5, 4.0};
  for (unsigned l = 0; l < sizeof(limits) / sizeof(limits[0]); l++) {
    Result ref = hvp(0, &limits[l], NULL, 0);
    Result r = hvp(0, &limits[l], &EnzymeCkptStoreAll, 1);
    double fdv[2 * N];
    fd(0, &limits[l], fdv);
    check("while", r.stats.taped_steps, 1, &r, &ref, fdv);
    if (r.stats.taped_steps < 1) {
      printf("while limit %g: no step\n", limits[l]);
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

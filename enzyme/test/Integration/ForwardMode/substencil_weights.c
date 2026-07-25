// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Shock-capturing reconstruction: five stencil values in, one face value out,
// through nonlinear weights that are deliberately non-smooth in the data.
// WENO5-JS, WENO5-Z and TENO5 differ only in how they weight the same three
// fifth-order candidates, and between them they cover three separate hazards
// that a plain "AD vs finite difference" check would either miss or misreport.
//
// 1. WENO5-JS is smooth: its 1/(beta+eps)^2 denominators are held off zero by
//    eps = 1e-6, so AD and a central difference agree, and this is the control
//    case that says the harness works.
//
// 2. WENO5-Z has a kink.  Its tau5 = |beta0 - beta2| is exactly zero on any
//    symmetric stencil -- beta0 and beta2 are equal there as an algebraic
//    identity, not by luck -- so a symmetric stencil sits exactly on the
//    non-differentiable point.  The one-sided slopes really do differ (by
//    around 20% of the value here), AD returns one of them, and a central
//    difference returns their average, which is neither.  This is the case
//    where FD is wrong and AD is right, and the test pins down all three
//    numbers rather than just asserting a disagreement.
//
// 3. TENO5 is discontinuous.  Its sharp cutoff sets delta_k = 0 the moment a
//    candidate's normalised strength falls below C_T = 1e-5, so the final
//    weights delta_k * d_k are piecewise CONSTANT in the data.  Two things
//    follow, and both are gated: inside a branch the reconstruction is exactly
//    linear in the stencil, so the gradient is exactly the frozen-weight
//    stencil to machine precision; and across a branch the primal jumps, so a
//    central difference straddling the cutoff is meaningless while AD's
//    within-branch answer stays exact.  The crossing is located by bisection
//    rather than hardcoded, so the test keeps working if the constants move.
//
// The linear-profile check at the top is the one that ties the schemes
// together: when every candidate agrees, all three collapse to the optimal
// fifth-order stencil (1/30, -13/60, 47/60, 9/20, -1/20), and the |tau5| kink
// is annihilated because the weight perturbation multiplies (f_k - q) == 0.

#include "../frechet.h"
#include "../test_utils.h"

extern double __enzyme_fwddiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

#define NSTENCIL 5

// Sharp cutoff threshold, sharpness exponent and denominator floors.
#define TENO_CT 1.0e-5
#define EPS_TENO 1.0e-40
#define EPS_WENOZ 1.0e-40
#define EPS_WENOJS 1.0e-6

// Optimal sub-stencil combination weights.
static const double d_opt[3] = {0.1, 0.6, 0.3};

// The nine fifth-order Lagrange coefficients, candidate-major.
static const double sub_coeff[9] = {
    1.0 / 3.0,  -7.0 / 6.0, 11.0 / 6.0, // f0: v[0], v[1], v[2]
    -1.0 / 6.0, 5.0 / 6.0,  1.0 / 3.0,  // f1: v[1], v[2], v[3]
    1.0 / 3.0,  5.0 / 6.0,  -1.0 / 6.0, // f2: v[2], v[3], v[4]
};

// The optimal fifth-order stencil, which every scheme has to reduce to when
// its candidates agree: sum_k d_k * sub_coeff[k].
static const double optimal[NSTENCIL] = {1.0 / 30.0, -13.0 / 60.0, 47.0 / 60.0,
                                         9.0 / 20.0, -1.0 / 20.0};

// Jiang-Shu smoothness indicators.
void smoothness(const double *v, double *beta) {
  double a0 = v[0] - 2.0 * v[1] + v[2];
  double b0 = v[0] - 4.0 * v[1] + 3.0 * v[2];
  double a1 = v[1] - 2.0 * v[2] + v[3];
  double b1 = v[1] - v[3];
  double a2 = v[2] - 2.0 * v[3] + v[4];
  double b2 = 3.0 * v[2] - 4.0 * v[3] + v[4];

  beta[0] = (13.0 / 12.0) * a0 * a0 + 0.25 * b0 * b0;
  beta[1] = (13.0 / 12.0) * a1 * a1 + 0.25 * b1 * b1;
  beta[2] = (13.0 / 12.0) * a2 * a2 + 0.25 * b2 * b2;
}

// The three fifth-order candidates at the face i+1/2.
void candidates(const double *v, double *f) {
  f[0] = sub_coeff[0] * v[0] + sub_coeff[1] * v[1] + sub_coeff[2] * v[2];
  f[1] = sub_coeff[3] * v[1] + sub_coeff[4] * v[2] + sub_coeff[5] * v[3];
  f[2] = sub_coeff[6] * v[2] + sub_coeff[7] * v[3] + sub_coeff[8] * v[4];
}

double normalise(const double *w, const double *f) {
  return (w[0] * f[0] + w[1] * f[1] + w[2] * f[2]) / (w[0] + w[1] + w[2]);
}

// WENO5-JS: alpha_k = d_k / (beta_k + eps)^2.
double weno5js(const double *v) {
  double beta[3], f[3], w[3];
  smoothness(v, beta);
  candidates(v, f);
  for (int k = 0; k < 3; k++) {
    double den = beta[k] + EPS_WENOJS;
    w[k] = d_opt[k] / (den * den);
  }
  return normalise(w, f);
}

// WENO5-Z with tau5 supplied by the caller, so the two smooth branches of
// |beta0 - beta2| can be differentiated separately.  `sign` selects which one:
// on a symmetric stencil both give the same value and their derivatives are
// precisely the two one-sided limits of the real scheme.
double weno5z_branch(const double *v, double sign) {
  double beta[3], f[3], w[3];
  smoothness(v, beta);
  candidates(v, f);
  double tau5 = sign * (beta[0] - beta[2]);
  for (int k = 0; k < 3; k++)
    w[k] = d_opt[k] * (1.0 + tau5 / (beta[k] + EPS_WENOZ));
  return normalise(w, f);
}

// WENO5-Z proper: tau5 = |beta0 - beta2|.
double weno5z(const double *v) {
  double beta[3], f[3], w[3];
  smoothness(v, beta);
  candidates(v, f);
  double tau5 = fabs(beta[0] - beta[2]);
  for (int k = 0; k < 3; k++)
    w[k] = d_opt[k] * (1.0 + tau5 / (beta[k] + EPS_WENOZ));
  return normalise(w, f);
}

// TENO5's cutoff flags.  Kept separate from the reconstruction because they are
// exactly what the frozen-weight identity freezes.
void teno5_cutoff(const double *v, double *delta) {
  double beta[3], g[3];
  smoothness(v, beta);
  double tau5 = fabs(beta[0] - beta[2]);

  double sum = 0.0;
  for (int k = 0; k < 3; k++) {
    double x = 1.0 + tau5 / (beta[k] + EPS_TENO);
    double x3 = x * x * x;
    g[k] = x3 * x3;
    sum += g[k];
  }
  for (int k = 0; k < 3; k++)
    delta[k] = (g[k] / sum < TENO_CT) ? 0.0 : 1.0;
}

double teno5(const double *v) {
  double delta[3], f[3], w[3];
  teno5_cutoff(v, delta);
  candidates(v, f);
  for (int k = 0; k < 3; k++)
    w[k] = delta[k] * d_opt[k];

  double sum = w[0] + w[1] + w[2];
  if (sum < EPS_TENO)
    return f[1]; // every candidate cut off: fall back to the central one
  return (w[0] * f[0] + w[1] * f[1] + w[2] * f[2]) / sum;
}

#define DEFINE_GRADIENT(NAME, SCHEME)                                          \
  void NAME(const double *v, double *g) {                                      \
    double dv[NSTENCIL];                                                       \
    for (int col = 0; col < NSTENCIL; col++) {                                 \
      for (int i = 0; i < NSTENCIL; i++)                                       \
        dv[i] = 0.0;                                                           \
      dv[col] = 1.0;                                                           \
      g[col] = __enzyme_fwddiff((void *)SCHEME, enzyme_dup, v, dv);            \
    }                                                                          \
  }

DEFINE_GRADIENT(weno5js_gradient, weno5js)
DEFINE_GRADIENT(weno5z_gradient, weno5z)
DEFINE_GRADIENT(teno5_gradient, teno5)

// The gradient of one smooth branch of WENO5-Z.
void weno5z_branch_gradient(const double *v, double sign, double *g) {
  double dv[NSTENCIL];
  for (int col = 0; col < NSTENCIL; col++) {
    for (int i = 0; i < NSTENCIL; i++)
      dv[i] = 0.0;
    dv[col] = 1.0;
    g[col] = __enzyme_fwddiff((void *)weno5z_branch, enzyme_dup, v, dv,
                              enzyme_const, sign);
  }
}

// Central-difference gradient, for the comparisons where FD is the thing under
// examination rather than the reference.
typedef double (*Scheme)(const double *);

void fd_gradient(Scheme q, const double *v, double h, double *g) {
  double vp[NSTENCIL], vm[NSTENCIL];
  for (int col = 0; col < NSTENCIL; col++) {
    for (int i = 0; i < NSTENCIL; i++) {
      vp[i] = v[i];
      vm[i] = v[i];
    }
    vp[col] += h;
    vm[col] -= h;
    g[col] = (q(vp) - q(vm)) / (2.0 * h);
  }
}

// TENO5's exact gradient inside a branch: with delta frozen the reconstruction
// is linear in v, so the gradient is just the normalised candidate stencil.
void teno5_frozen_gradient(const double *v, double *g) {
  double delta[3];
  teno5_cutoff(v, delta);

  double w[3], sum = 0.0;
  for (int k = 0; k < 3; k++) {
    w[k] = delta[k] * d_opt[k];
    sum += w[k];
  }

  for (int j = 0; j < NSTENCIL; j++)
    g[j] = 0.0;
  if (sum < EPS_TENO) {
    g[1] = sub_coeff[3];
    g[2] = sub_coeff[4];
    g[3] = sub_coeff[5];
    return;
  }
  for (int k = 0; k < 3; k++)
    for (int c = 0; c < 3; c++)
      g[k + c] += (w[k] / sum) * sub_coeff[3 * k + c];
}

// A smooth asymmetric base with a step of amplitude s across the face, which
// drives the TENO cutoff through its flips as s grows.
void step_profile(double s, double *v) {
  for (int j = 0; j < NSTENCIL; j++)
    v[j] = sin(0.7 * (double)j) + (j >= 3 ? s : 0.0);
}

int main() {
  double g[NSTENCIL], g2[NSTENCIL], fd[NSTENCIL];

  // Every scheme reduces to the optimal fifth-order stencil when the
  // candidates agree, and the |tau5| kink cannot bite there because the weight
  // perturbation multiplies (f_k - q) == 0.
  {
    double linear[NSTENCIL];
    for (int j = 0; j < NSTENCIL; j++)
      linear[j] = 3.0 + 2.0 * (double)j;

    weno5js_gradient(linear, g);
    APPROX_EQ(frechet_rel_error(g, optimal, NSTENCIL), 0.0, 1e-14);
    weno5z_gradient(linear, g);
    APPROX_EQ(frechet_rel_error(g, optimal, NSTENCIL), 0.0, 1e-14);
    teno5_gradient(linear, g);
    APPROX_EQ(frechet_rel_error(g, optimal, NSTENCIL), 0.0, 1e-14);
    printf("linear profile: all three schemes give the optimal stencil\n");
  }

  // WENO5-JS is smooth, so AD and a central difference agree.  Control case.
  {
    double v[NSTENCIL];
    for (int j = 0; j < NSTENCIL; j++)
      v[j] = sin(0.7 * (double)j);

    weno5js_gradient(v, g);
    fd_gradient(weno5js, v, 1e-6, fd);
    double err = frechet_rel_error(fd, g, NSTENCIL);
    printf("weno5-js smooth: AD vs FD %.3e\n", err);
    APPROX_EQ(err, 0.0, 1e-8);
  }

  // WENO5-Z on a symmetric stencil sits exactly on the |tau5| kink.
  {
    double v[NSTENCIL] = {1.0, 0.0, 1.0, 0.0, 1.0};
    double beta[3];
    smoothness(v, beta);

    // beta0 == beta2 is an algebraic identity for a symmetric stencil, so the
    // kink is hit exactly rather than approached.
    APPROX_EQ(beta[0] - beta[2], 0.0, 0.0);

    double plus[NSTENCIL], minus[NSTENCIL], mean[NSTENCIL];
    weno5z_gradient(v, g);
    weno5z_branch_gradient(v, 1.0, plus);
    weno5z_branch_gradient(v, -1.0, minus);
    for (int j = 0; j < NSTENCIL; j++)
      mean[j] = 0.5 * (plus[j] + minus[j]);

    double jump = 0.0;
    for (int j = 0; j < NSTENCIL; j++) {
      double diff = fabs(plus[j] - minus[j]);
      if (diff > jump)
        jump = diff;
    }
    printf("weno5-z kink: one-sided slopes differ by %.3e\n", jump);

    // The kink is real, not a rounding artefact.
    if (jump < 1e-3) {
      fprintf(stderr, "one-sided slopes agree -- no kink to test\n");
      abort();
    }

    // AD lands on one of the two one-sided limits, exactly.
    double to_plus = frechet_rel_error(g, plus, NSTENCIL);
    double to_minus = frechet_rel_error(g, minus, NSTENCIL);
    printf("weno5-z kink: AD to (+) branch %.3e, to (-) branch %.3e\n", to_plus,
           to_minus);
    if (to_plus > 1e-14 && to_minus > 1e-14) {
      fprintf(stderr, "AD matched neither one-sided limit\n");
      abort();
    }

    // A central difference returns their average, which is neither -- this is
    // the case where finite differencing is simply the wrong instrument.  It
    // tracks the mean to O(h) rather than the usual O(h^2), since the second
    // order term is exactly what the kink destroys, so the tolerance here is
    // the step size and not its square.  That is still four orders below the
    // distance from AD, which is the comparison the test is making.
    fd_gradient(weno5z, v, 1e-6, fd);
    printf("weno5-z kink: FD to mean %.3e, FD to AD %.3e\n",
           frechet_rel_error(fd, mean, NSTENCIL),
           frechet_rel_error(fd, g, NSTENCIL));
    APPROX_EQ(frechet_rel_error(fd, mean, NSTENCIL), 0.0, 1e-5);
    if (frechet_rel_error(fd, g, NSTENCIL) < 1e-3) {
      fprintf(stderr, "FD agreed with AD at the kink -- expected it not to\n");
      abort();
    }
  }

  // TENO5's weights are piecewise constant, so inside a branch its gradient is
  // exactly the frozen-weight stencil, discontinuity or not.
  {
    const double amplitudes[] = {0.0, 0.2, 0.35, 0.8, 1.5, 3.0};
    const int n = sizeof(amplitudes) / sizeof(amplitudes[0]);

    for (int i = 0; i < n; i++) {
      double v[NSTENCIL], delta[3];
      step_profile(amplitudes[i], v);
      teno5_cutoff(v, delta);
      teno5_gradient(v, g);
      teno5_frozen_gradient(v, g2);

      double err = frechet_rel_error(g, g2, NSTENCIL);
      printf("teno5 s=%.2f delta=(%.0f,%.0f,%.0f): AD vs frozen weights %.3e\n",
             amplitudes[i], delta[0], delta[1], delta[2], err);
      APPROX_EQ(err, 0.0, 1e-14);
    }
  }

  // Bisect to a cutoff crossing and show what each instrument reports there.
  {
    double lo = 0.20, hi = 1.00; // delta_2 flips 1 -> 0 somewhere inside
    double v[NSTENCIL], delta[3];

    step_profile(lo, v);
    teno5_cutoff(v, delta);
    double delta2_lo = delta[2];
    step_profile(hi, v);
    teno5_cutoff(v, delta);
    if (delta2_lo == delta[2]) {
      fprintf(stderr, "no cutoff crossing bracketed\n");
      abort();
    }

    for (int it = 0; it < 200; it++) {
      double mid = 0.5 * (lo + hi);
      step_profile(mid, v);
      teno5_cutoff(v, delta);
      if (delta[2] == delta2_lo)
        lo = mid;
      else
        hi = mid;
    }
    printf("teno5 cutoff crossing bracketed to [%.17g, %.17g]\n", lo, hi);

    // The primal really is discontinuous across it.
    double vlo[NSTENCIL], vhi[NSTENCIL];
    step_profile(lo, vlo);
    step_profile(hi, vhi);
    double jump = fabs(teno5(vhi) - teno5(vlo));
    printf("teno5 primal jump across the cutoff: %.3e\n", jump);
    if (jump < 1e-3) {
      fprintf(stderr, "cutoff crossing is not discontinuous -- nothing to "
                      "distinguish AD from FD\n");
      abort();
    }

    // AD on the low side reports that side's exact linear stencil...
    teno5_gradient(vlo, g);
    teno5_frozen_gradient(vlo, g2);
    APPROX_EQ(frechet_rel_error(g, g2, NSTENCIL), 0.0, 1e-14);

    // ...while a central difference wide enough to straddle the crossing is
    // reporting the jump divided by the step, and agrees with nothing.
    fd_gradient(teno5, vlo, 0.05, fd);
    double err = frechet_rel_error(fd, g, NSTENCIL);
    printf("teno5 at the cutoff: straddling FD vs AD %.3e\n", err);
    if (err < 0.1) {
      fprintf(stderr, "straddling FD tracked AD -- expected it not to\n");
      abort();
    }
  }

  printf("done\n");
  return 0;
}

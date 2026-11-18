// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Forward-over-reverse on the mathematical entropy of the Euler equations,
// which is the one place in a compressible solver where second derivatives are
// load bearing.  Entropy-stable schemes exist because
//
//   eta(U) = -rho * s / (gamma - 1),   s = log(p) - gamma * log(rho)
//
// is convex, and every guarantee they offer follows from that -- so the Hessian
// this test builds is not an abstraction, it is the object whose definiteness
// makes the scheme provably stable.
//
// Nested AD is the fragile case in practice, and this file gates it four ways,
// none of which needs a hardcoded derivative:
//
//   1. The reverse-mode gradient must reproduce the entropy variables
//      w = (( gamma - s)/(gamma-1) - beta|u|^2, 2 beta u, 2 beta v, 2 beta w,
//      -2 beta), beta = rho/2p, in closed form.  This is the vector an
//      entropy-stable flux is actually written in terms of.
//   2. The Hessian must be symmetric.  It is a Hessian, so this is free, exact,
//      and impossible to satisfy by accident across 25 independently computed
//      entries.
//   3. The Hessian must be positive definite, which is the convexity that the
//      scheme's stability proof rests on.  Cholesky either completes or it does
//      not; no tolerance is involved.
//   4. Forward-over-reverse must agree with plain forward mode applied to the
//      closed-form gradient.  Two different AD compositions, one of them
//      nested, over the same mathematics.
//
// Together these say the second-order tape is right without ever writing down
// a second derivative by hand.

#include "../euler.h"
#include "../frechet.h"
#include "../test_utils.h"

extern double __enzyme_autodiff(void *, ...);
extern void __enzyme_fwddiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

// Harten's entropy for the compressible Euler equations.
double mathematical_entropy(const double *U, const EulerEos *eos) {
  double p = euler_pressure(U, eos);
  double s = log(p) - eos->gamma * log(U[RHO]);
  return -U[RHO] * s / (eos->gamma - 1.0);
}

// The entropy variables, in the form an entropy-stable flux consumes them.
void entropy_variables(const double *U, const EulerEos *eos, double *w) {
  double rho = U[RHO];
  double inv = 1.0 / rho;
  double u = U[RHOU] * inv;
  double v = U[RHOV] * inv;
  double z = U[RHOW] * inv;

  double p = euler_pressure(U, eos);
  double beta = rho / (2.0 * p);
  double s = log(p) - eos->gamma * log(rho);

  w[RHO] = (eos->gamma - s) / (eos->gamma - 1.0) - beta * (u * u + v * v + z * z);
  w[RHOU] = 2.0 * beta * u;
  w[RHOV] = 2.0 * beta * v;
  w[RHOW] = 2.0 * beta * z;
  w[RHOE] = -2.0 * beta;
}

void entropy_gradient(const double *U, const EulerEos *eos, double *g) {
  for (int i = 0; i < NVARS; i++)
    g[i] = 0.0;

  __enzyme_autodiff((void *)mathematical_entropy, enzyme_dup, U, g,
                    enzyme_const, eos);
}

// Forward over reverse: seed one input direction, differentiate the whole
// reverse sweep, and read off a column of the Hessian.
void entropy_hessian(const double *U, const EulerEos *eos, double *H) {
  double dU[NVARS], g[NVARS], dg[NVARS];

  for (int col = 0; col < NVARS; col++) {
    for (int i = 0; i < NVARS; i++) {
      dU[i] = 0.0;
      dg[i] = 0.0;
    }
    dU[col] = 1.0;

    __enzyme_fwddiff((void *)entropy_gradient, enzyme_dup, U, dU, enzyme_const,
                     eos, enzyme_dup, g, dg);

    for (int row = 0; row < NVARS; row++)
      H[row * NVARS + col] = dg[row];
  }
}

// Plain forward mode over the closed-form gradient, for comparison.
void entropy_variables_jacobian(const double *U, const EulerEos *eos,
                                double *J) {
  double dU[NVARS], w[NVARS], dw[NVARS];

  for (int col = 0; col < NVARS; col++) {
    for (int i = 0; i < NVARS; i++)
      dU[i] = 0.0;
    dU[col] = 1.0;

    __enzyme_fwddiff((void *)entropy_variables, enzyme_dup, U, dU, enzyme_const,
                     eos, enzyme_dup, w, dw);

    for (int row = 0; row < NVARS; row++)
      J[row * NVARS + col] = dw[row];
  }
}

// Cholesky without pivoting: succeeds exactly when A is positive definite.
// Returns the smallest pivot seen, or a negative value if the factorisation
// broke down.
double cholesky_min_pivot(const double *A, int n) {
  double L[NVARS * NVARS];
  double min_pivot = 0.0;

  for (int i = 0; i < n * n; i++)
    L[i] = 0.0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = A[i * n + j];
      for (int k = 0; k < j; k++)
        sum -= L[i * n + k] * L[j * n + k];

      if (i == j) {
        if (sum <= 0.0)
          return -1.0;
        if (min_pivot == 0.0 || sum < min_pivot)
          min_pivot = sum;
        L[i * n + j] = sqrt(sum);
      } else {
        L[i * n + j] = sum / L[j * n + j];
      }
    }
  }
  return min_pivot;
}

int main() {
  EulerEos eos = {1.4, 0.0};

  for (int s = 0; s < EULER_NSTATES; s++) {
    const double *q = euler_primitives[s];
    double U[NVARS], w[NVARS], g[NVARS];
    double H[NVARS * NVARS], J[NVARS * NVARS];

    euler_from_primitive(q[0], q[1], q[2], q[3], q[4], &eos, U);

    // 1. The reverse gradient is the entropy-variable vector.
    entropy_variables(U, &eos, w);
    entropy_gradient(U, &eos, g);
    double grad_err = frechet_rel_error(g, w, NVARS);
    APPROX_EQ(grad_err, 0.0, 1e-13);

    entropy_hessian(U, &eos, H);

    // 2. Symmetry, across 25 separately computed entries.
    double scale = 0.0;
    for (int i = 0; i < NVARS * NVARS; i++)
      if (fabs(H[i]) > scale)
        scale = fabs(H[i]);

    double asym = 0.0;
    for (int row = 0; row < NVARS; row++)
      for (int col = 0; col < NVARS; col++) {
        double d = fabs(H[row * NVARS + col] - H[col * NVARS + row]);
        if (d > asym)
          asym = d;
      }
    APPROX_EQ(asym / scale, 0.0, 1e-13);

    // 3. Convexity: Cholesky completes, so the Hessian is positive definite.
    double pivot = cholesky_min_pivot(H, NVARS);
    if (pivot < 0.0) {
      fprintf(stderr, "entropy Hessian is not positive definite at state %d -- "
                      "the convexity the scheme relies on is broken\n",
              s);
      abort();
    }

    // 4. Nested AD against single-level AD of the closed-form gradient.
    entropy_variables_jacobian(U, &eos, J);
    double nest_err = frechet_rel_error(H, J, NVARS * NVARS);
    APPROX_EQ(nest_err, 0.0, 1e-12);

    printf("state %d: grad %.2e  asym %.2e  min pivot %.3e  nested vs flat "
           "%.2e\n",
           s, grad_err, asym / scale, pivot, nest_err);
  }

  printf("done\n");
  return 0;
}

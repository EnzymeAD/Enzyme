// Frechet finite-difference gate for Enzyme-built Jacobians.
//
// Rather than compare a derivative against a hardcoded constant, compare the
// Jacobian's action against a central difference of the primal itself,
//
//   J*v  ==  (R(U + eps*v) - R(U - eps*v)) / (2*eps)  +  O(eps^2),
//
// so a kernel too involved to differentiate by hand can still be gated.  The
// probe direction v is the part that needs care: conserved states span many
// decades (rho ~ 1e-2 against rho*E ~ 1e5 for a hypersonic freestream), so an
// unscaled direction leaves the linear regime in the small components while
// barely perturbing the large ones.  frechet_direction gives every variable its
// own scale, taken from the state, and is deterministic in `seed` so a failure
// reproduces exactly.
//
// State vectors are interlaced: U[cell * nvars + var].

#include <math.h>
#include <stdlib.h>

// Residual R(U) evaluated into `out`; `ctx` carries whatever the kernel needs.
typedef void (*FrechetResidual)(const double *U, double *out, void *ctx);

// Deterministic per-variable-scaled probe direction.  A variable that is
// identically zero across the state falls back to unit scale so its columns are
// still probed.
void frechet_direction(int nvars, int ncells, const double *U, unsigned seed,
                       double *v) {
  for (int var = 0; var < nvars; var++) {
    double scale = 0.0;
    for (int cell = 0; cell < ncells; cell++) {
      double mag = fabs(U[cell * nvars + var]);
      if (mag > scale)
        scale = mag;
    }
    if (scale == 0.0)
      scale = 1.0;
    for (int cell = 0; cell < ncells; cell++) {
      int i = cell * nvars + var;
      v[i] = scale * sin(12.9898 * (double)(i + 1) + 78.233 * (double)seed);
    }
  }
}

// Central-difference directional derivative of R at U along v.
void frechet_apply(FrechetResidual R, void *ctx, const double *U, int n, int m,
                   const double *v, double eps, double *out) {
  double *Up = (double *)malloc(sizeof(double) * n);
  double *Um = (double *)malloc(sizeof(double) * n);
  double *Rp = (double *)malloc(sizeof(double) * m);
  double *Rm = (double *)malloc(sizeof(double) * m);

  for (int i = 0; i < n; i++) {
    Up[i] = U[i] + eps * v[i];
    Um[i] = U[i] - eps * v[i];
  }
  R(Up, Rp, ctx);
  R(Um, Rm, ctx);
  for (int i = 0; i < m; i++)
    out[i] = (Rp[i] - Rm[i]) / (2.0 * eps);

  free(Up);
  free(Um);
  free(Rp);
  free(Rm);
}

// Largest componentwise deviation of `a` from `b`, relative to the scale of b.
// Comparing against the vector scale rather than each component keeps a
// near-cancelling component from dominating the report.
double frechet_rel_error(const double *a, const double *b, int m) {
  double scale = 0.0, err = 0.0;
  for (int i = 0; i < m; i++) {
    double mag = fabs(b[i]);
    if (mag > scale)
      scale = mag;
  }
  if (scale == 0.0)
    scale = 1.0;
  for (int i = 0; i < m; i++) {
    double rel = fabs(a[i] - b[i]) / scale;
    if (rel > err)
      err = rel;
  }
  return err;
}

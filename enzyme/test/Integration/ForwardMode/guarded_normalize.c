// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Normalising a vector that can vanish, which is the geometric kernel under
// every rotated-hybrid Riemann solver and every surface-normal calculation.
// The primal hazard is well known and every code guards it; the derivative
// hazard is separate, survives the usual guards, and is what this file pins
// down.
//
// Away from the degeneracy the Jacobian is the closed-form projector
//
//   d(v/|v|)/dv = (I - n n^T) / |v|,   n = v/|v|,
//
// which is gated directly, along with three structural facts that need no
// reference values at all: the Jacobian is symmetric, it annihilates n on both
// sides, and -- for the full rotated frame below -- the orthonormality of the
// frame is preserved under differentiation.
//
// At the degeneracy the three ways of writing the guard stop agreeing, and the
// difference is invisible in the primal:
//
//   * Guarding after the sqrt still EVALUATES sqrt(0), whose own derivative is
//     0/0.  The branch nonetheless returns a constant, and Enzyme returns an
//     exactly zero tangent rather than letting the NaN escape.
//   * Guarding before the sqrt never reaches it.  Same answer, and this is the
//     version that is obviously correct by inspection.
//   * Flooring the length branchlessly with fmax(|v|, eps) is the version a
//     vectorising author reaches for, and it is equally NaN-free -- but its
//     derivative at the origin is I/eps, which is 1e12 here.  That is the right
//     derivative of what was actually written, and it is a landmine: it is
//     finite, so nothing traps, and it lands in an assembled Jacobian as a row
//     twelve orders of magnitude out of scale.
//
// The last point is the reason to gate this rather than just document it.  All
// three idioms look interchangeable, all three keep the primal finite, and only
// two of them keep the Jacobian usable.

#include "../frechet.h"
#include "../test_utils.h"

extern double __enzyme_fwddiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

#define GUARD_EPS 1.0e-12

// Guard AFTER the sqrt: sqrt(0) is evaluated at the origin.
void normalize_guard_after(const double *v, double *n) {
  double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len <= GUARD_EPS) {
    n[0] = 1.0;
    n[1] = 0.0;
    n[2] = 0.0;
    return;
  }
  double inv = 1.0 / len;
  n[0] = v[0] * inv;
  n[1] = v[1] * inv;
  n[2] = v[2] * inv;
}

// Guard BEFORE the sqrt: the sqrt is never reached at the origin.
void normalize_guard_before(const double *v, double *n) {
  double sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (sq <= GUARD_EPS * GUARD_EPS) {
    n[0] = 1.0;
    n[1] = 0.0;
    n[2] = 0.0;
    return;
  }
  double inv = 1.0 / sqrt(sq);
  n[0] = v[0] * inv;
  n[1] = v[1] * inv;
  n[2] = v[2] * inv;
}

// Branchless floor: no branch to mispredict, no NaN, and a derivative that
// blows up as 1/eps at the origin instead of vanishing.
void normalize_guard_floor(const double *v, double *n) {
  double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  double inv = 1.0 / fmax(len, GUARD_EPS);
  n[0] = v[0] * inv;
  n[1] = v[1] * inv;
  n[2] = v[2] * inv;
}

#define DEFINE_NORMALIZE_JACOBIAN(NAME, FN)                                    \
  void NAME(const double *v, double *J) {                                      \
    double dv[3], n[3], dn[3];                                                 \
    for (int col = 0; col < 3; col++) {                                        \
      for (int i = 0; i < 3; i++)                                              \
        dv[i] = 0.0;                                                           \
      dv[col] = 1.0;                                                           \
      __enzyme_fwddiff((void *)FN, enzyme_dup, v, dv, enzyme_dup, n, dn);      \
      for (int row = 0; row < 3; row++)                                        \
        J[row * 3 + col] = dn[row];                                            \
    }                                                                          \
  }

DEFINE_NORMALIZE_JACOBIAN(jacobian_guard_after, normalize_guard_after)
DEFINE_NORMALIZE_JACOBIAN(jacobian_guard_before, normalize_guard_before)
DEFINE_NORMALIZE_JACOBIAN(jacobian_guard_floor, normalize_guard_floor)

// (I - n n^T) / |v|
void projector(const double *v, double *J) {
  double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  double n[3] = {v[0] / len, v[1] / len, v[2] / len};
  for (int row = 0; row < 3; row++)
    for (int col = 0; col < 3; col++)
      J[row * 3 + col] = ((row == col ? 1.0 : 0.0) - n[row] * n[col]) / len;
}

int all_finite(const double *a, int n) {
  for (int i = 0; i < n; i++)
    if (!(a[i] == a[i]) || a[i] > 1e300 || a[i] < -1e300)
      return 0;
  return 1;
}

// The rotated frame of Nishikawa & Kitamura: the principal direction follows
// the velocity jump, the transverse direction is what is left of the face
// normal after projecting that out.  Two nested guarded normalisations and a
// sign flip, which is a fair sample of what real geometry code looks like.
//
// x = (uL, uR); out = (n1, n2, alpha1, alpha2).
void rotated_frame(const double *x, const double *nf, double *out) {
  double dq[3] = {x[3] - x[0], x[4] - x[1], x[5] - x[2]};
  double len = sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2]);

  if (len <= GUARD_EPS) {
    // Negligible velocity jump: fall back to the grid-aligned frame.
    for (int i = 0; i < 3; i++) {
      out[i] = nf[i];
      out[3 + i] = nf[i];
    }
    out[6] = 0.0;
    out[7] = 1.0;
    return;
  }

  double n1[3] = {dq[0] / len, dq[1] / len, dq[2] / len};
  double a1 = nf[0] * n1[0] + nf[1] * n1[1] + nf[2] * n1[2];

  // Reorient so the principal weight is non-negative.
  if (a1 < 0.0) {
    for (int i = 0; i < 3; i++)
      n1[i] = -n1[i];
    a1 = -a1;
  }

  double t[3] = {nf[0] - a1 * n1[0], nf[1] - a1 * n1[1], nf[2] - a1 * n1[2]};
  double tlen = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);

  if (tlen <= GUARD_EPS) {
    // Velocity jump parallel to the face normal: the transverse direction is
    // undefined and carries no weight.
    for (int i = 0; i < 3; i++) {
      out[i] = n1[i];
      out[3 + i] = nf[i];
    }
    out[6] = 1.0;
    out[7] = 0.0;
    return;
  }

  for (int i = 0; i < 3; i++) {
    out[i] = n1[i];
    out[3 + i] = t[i] / tlen;
  }
  out[6] = a1;
  out[7] = nf[0] * out[3] + nf[1] * out[4] + nf[2] * out[5];
}

// Three identities the frame satisfies for every input, hence three gradients
// that must vanish identically.
double frame_n1_normsq(const double *x, const double *nf) {
  double o[8];
  rotated_frame(x, nf, o);
  return o[0] * o[0] + o[1] * o[1] + o[2] * o[2];
}

double frame_n1_dot_n2(const double *x, const double *nf) {
  double o[8];
  rotated_frame(x, nf, o);
  return o[0] * o[3] + o[1] * o[4] + o[2] * o[5];
}

double frame_alpha_normsq(const double *x, const double *nf) {
  double o[8];
  rotated_frame(x, nf, o);
  return o[6] * o[6] + o[7] * o[7];
}

#define DEFINE_IDENTITY_GRADIENT(NAME, FN)                                     \
  void NAME(const double *x, const double *nf, double *g) {                    \
    double dx[6];                                                              \
    for (int col = 0; col < 6; col++) {                                        \
      for (int i = 0; i < 6; i++)                                              \
        dx[i] = 0.0;                                                           \
      dx[col] = 1.0;                                                           \
      g[col] = __enzyme_fwddiff((void *)FN, enzyme_dup, x, dx, enzyme_const,   \
                                nf);                                           \
    }                                                                          \
  }

DEFINE_IDENTITY_GRADIENT(grad_n1_normsq, frame_n1_normsq)
DEFINE_IDENTITY_GRADIENT(grad_n1_dot_n2, frame_n1_dot_n2)
DEFINE_IDENTITY_GRADIENT(grad_alpha_normsq, frame_alpha_normsq)

int main() {
  double J[9], K[9], want[9];

  // Away from the degeneracy every idiom gives the projector.
  {
    const double vectors[3][3] = {
        {3.0, -4.0, 12.0}, {1.0, 0.0, 0.0}, {-0.02, 0.005, 0.031}};

    for (int i = 0; i < 3; i++) {
      const double *v = vectors[i];
      projector(v, want);

      jacobian_guard_after(v, J);
      APPROX_EQ(frechet_rel_error(J, want, 9), 0.0, 1e-14);
      jacobian_guard_before(v, J);
      APPROX_EQ(frechet_rel_error(J, want, 9), 0.0, 1e-14);
      jacobian_guard_floor(v, J);
      APPROX_EQ(frechet_rel_error(J, want, 9), 0.0, 1e-14);

      // Structural: symmetric, and it annihilates n from both sides.
      double len = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      double n[3] = {v[0] / len, v[1] / len, v[2] / len};
      double scale = 1.0 / len;
      for (int r = 0; r < 3; r++) {
        double row = 0.0, col = 0.0;
        for (int c = 0; c < 3; c++) {
          APPROX_EQ(J[r * 3 + c], J[c * 3 + r], 1e-15 * scale);
          row += J[r * 3 + c] * n[c];
          col += J[c * 3 + r] * n[c];
        }
        APPROX_EQ(row, 0.0, 1e-14 * scale);
        APPROX_EQ(col, 0.0, 1e-14 * scale);
      }
      printf("normalize |v|=%.4g: projector, symmetry and null space all hold\n",
             len);
    }
  }

  // At the origin, the two branch idioms return an exactly zero tangent -- the
  // sqrt's 0/0 does not leak out of the untaken path.
  {
    double origin[3] = {0.0, 0.0, 0.0};

    jacobian_guard_after(origin, J);
    printf("origin, guard after sqrt : J[0][0] = %g\n", J[0]);
    if (!all_finite(J, 9)) {
      fprintf(stderr, "guard-after produced a non-finite tangent at the "
                      "origin\n");
      abort();
    }
    for (int i = 0; i < 9; i++)
      APPROX_EQ(J[i], 0.0, 0.0);

    jacobian_guard_before(origin, K);
    printf("origin, guard before sqrt: J[0][0] = %g\n", K[0]);
    if (!all_finite(K, 9)) {
      fprintf(stderr, "guard-before produced a non-finite tangent at the "
                      "origin\n");
      abort();
    }
    for (int i = 0; i < 9; i++)
      APPROX_EQ(K[i], 0.0, 0.0);

    // The branchless floor is finite too, and that is exactly the problem: it
    // is I/eps, a perfectly valid derivative of what was written and a row of
    // an assembled Jacobian that is 1e12 out of scale.
    jacobian_guard_floor(origin, K);
    printf("origin, branchless floor : J[0][0] = %g (1/eps = %g)\n", K[0],
           1.0 / GUARD_EPS);
    if (!all_finite(K, 9)) {
      fprintf(stderr, "branchless floor produced a non-finite tangent\n");
      abort();
    }
    for (int row = 0; row < 3; row++)
      for (int col = 0; col < 3; col++)
        APPROX_EQ(K[row * 3 + col],
                  (row == col ? 1.0 / GUARD_EPS : 0.0), 1e-3);
  }

  // Just inside the guard the branch idioms are still zero; just outside, the
  // stiffness the guard was hiding is fully present.  The guard relocates the
  // blow-up, it does not remove it.
  {
    double inside[3] = {0.5 * GUARD_EPS, 0.0, 0.0};
    double outside[3] = {2.0 * GUARD_EPS, 0.0, 0.0};

    jacobian_guard_after(inside, J);
    for (int i = 0; i < 9; i++)
      APPROX_EQ(J[i], 0.0, 0.0);

    jacobian_guard_after(outside, J);
    projector(outside, want);
    APPROX_EQ(frechet_rel_error(J, want, 9), 0.0, 1e-14);
    printf("just outside the guard: |J| ~ %.3g\n", fabs(J[4]));
    if (!(fabs(J[4]) > 1e11)) {
      fprintf(stderr, "expected the projector to be stiff just outside the "
                      "guard\n");
      abort();
    }
  }

  // The rotated frame: orthonormality is an identity in the inputs, so its
  // gradient has to vanish -- including on both degenerate branches, where the
  // frame is assembled from entirely different expressions.
  {
    const double nf[3] = {0.6, -0.8, 0.0};
    const double cases[4][6] = {
        {10.0, 3.0, -2.0, 40.0, -9.0, 5.0},   // generic
        {10.0, 3.0, -2.0, 10.0, 3.0, -2.0},   // no velocity jump
        {0.0, 0.0, 0.0, 0.6, -0.8, 0.0},      // jump parallel to the normal
        {1.0, 1.0, 1.0, 1.0 - 0.6, 1.0 + 0.8, 1.0}, // antiparallel, flips sign
    };
    const char *labels[4] = {"generic", "no jump", "parallel", "antiparallel"};

    for (int c = 0; c < 4; c++) {
      const double *x = cases[c];
      double o[8], g[6];

      rotated_frame(x, nf, o);
      APPROX_EQ(frame_n1_normsq(x, nf), 1.0, 1e-14);
      APPROX_EQ(frame_alpha_normsq(x, nf), 1.0, 1e-14);

      grad_n1_normsq(x, nf, g);
      if (!all_finite(g, 6)) {
        fprintf(stderr, "non-finite d|n1|^2 on the %s case\n", labels[c]);
        abort();
      }
      for (int i = 0; i < 6; i++)
        APPROX_EQ(g[i], 0.0, 1e-13);

      grad_n1_dot_n2(x, nf, g);
      if (!all_finite(g, 6)) {
        fprintf(stderr, "non-finite d(n1.n2) on the %s case\n", labels[c]);
        abort();
      }
      for (int i = 0; i < 6; i++)
        APPROX_EQ(g[i], 0.0, 1e-13);

      grad_alpha_normsq(x, nf, g);
      if (!all_finite(g, 6)) {
        fprintf(stderr, "non-finite d(alpha1^2+alpha2^2) on the %s case\n",
                labels[c]);
        abort();
      }
      for (int i = 0; i < 6; i++)
        APPROX_EQ(g[i], 0.0, 1e-13);

      printf("rotated frame %-13s: orthonormality gradients all vanish\n",
             labels[c]);
    }
  }

  printf("done\n");
  return 0;
}

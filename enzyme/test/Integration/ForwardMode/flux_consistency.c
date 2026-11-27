// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Differentiate the consistency property of an approximate Riemann solver.
//
// Every consistent numerical flux satisfies Fstar(U, U, n) == F(U, n) as an
// identity in U.  Differentiating both sides gives a Frechet gate on the two 
// face blocks a CFD Jacobian is assembled from:
//
//   dFstar/dUL + dFstar/dUR  ==  A(U).n   at UL == UR,
//
// with A the physical flux Jacobian.  It is exact, needs no finite difference,
// and holds independently of which branch the solver takes. The state space
// spans both subsonic and supersonic faces:
//
//   * Subsonic faces land in the HLL star region, where the flux is a rational
//     function of both states and both blocks are dense.
//   * Supersonic faces land in the fully upwind branch, where Fstar == FL and
//     the right block must come back EXACTLY zero -- a structural property, not
//     an approximate one, so it is gated at zero tolerance.
//
// Rusanov additionally exercises a tie in fmax: at UL == UR the two wave-speed
// estimates are equal, so the dissipation coefficient sits exactly on the
// non-differentiable point.  The identity survives anyway, because whatever
// one-sided value the branch picks multiplies (UR - UL) == 0; the surviving
// terms are -lambda/2 and +lambda/2, which cancel in the sum.  Enzyme is free
// to choose either side of the tie and still has to pass.

#include "../euler.h"
#include "../frechet.h"
#include "../test_utils.h"

extern void __enzyme_fwddiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

// Rusanov / local Lax-Friedrichs.
void rusanov_flux(const double *UL, const double *UR, const double *n,
                  const EulerEos *eos, double *F) {
  double FL[NVARS], FR[NVARS];
  euler_flux(UL, n, eos, FL);
  euler_flux(UR, n, eos, FR);

  double lam = fmax(fabs(euler_normal_velocity(UL, n)) +
                        euler_sound_speed(UL, eos),
                    fabs(euler_normal_velocity(UR, n)) +
                        euler_sound_speed(UR, eos));

  for (int i = 0; i < NVARS; i++)
    F[i] = 0.5 * (FL[i] + FR[i]) - 0.5 * lam * (UR[i] - UL[i]);
}

// HLL with Davis wave-speed estimates.
void hll_flux(const double *UL, const double *UR, const double *n,
              const EulerEos *eos, double *F) {
  double FL[NVARS], FR[NVARS];
  euler_flux(UL, n, eos, FL);
  euler_flux(UR, n, eos, FR);

  double unL = euler_normal_velocity(UL, n);
  double unR = euler_normal_velocity(UR, n);
  double cL = euler_sound_speed(UL, eos);
  double cR = euler_sound_speed(UR, eos);

  double sl = fmin(unL - cL, unR - cR);
  double sr = fmax(unL + cL, unR + cR);

  if (sl >= 0.0) {
    for (int i = 0; i < NVARS; i++)
      F[i] = FL[i];
  } else if (sr <= 0.0) {
    for (int i = 0; i < NVARS; i++)
      F[i] = FR[i];
  } else {
    double inv = 1.0 / (sr - sl);
    for (int i = 0; i < NVARS; i++)
      F[i] = (sr * FL[i] - sl * FR[i] + sl * sr * (UR[i] - UL[i])) * inv;
  }
}

// The physical flux Jacobian A(U).n, one seeded tangent per column.
void flux_jacobian(const double *U, const double *n, const EulerEos *eos,
                   double *A) {
  double dU[NVARS], F[NVARS], dF[NVARS];

  for (int col = 0; col < NVARS; col++) {
    for (int i = 0; i < NVARS; i++)
      dU[i] = 0.0;
    dU[col] = 1.0;

    __enzyme_fwddiff((void *)euler_flux, enzyme_dup, U, dU, enzyme_const, n,
                     enzyme_const, eos, enzyme_dup, F, dF);

    for (int row = 0; row < NVARS; row++)
      A[row * NVARS + col] = dF[row];
  }
}

// The two face blocks of a numerical flux.  Seeding one side at a time keeps
// the left and right contributions separable, which the upwinding check needs;
// the consistency identity then uses their sum.
#define DEFINE_FACE_BLOCKS(NAME, FLUX)                                         \
  void NAME(const double *UL, const double *UR, const double *n,               \
            const EulerEos *eos, double *AL, double *AR) {                     \
    double dUL[NVARS], dUR[NVARS], F[NVARS], dF[NVARS];                        \
                                                                               \
    for (int col = 0; col < NVARS; col++) {                                    \
      for (int i = 0; i < NVARS; i++) {                                        \
        dUL[i] = 0.0;                                                          \
        dUR[i] = 0.0;                                                          \
      }                                                                        \
      dUL[col] = 1.0;                                                          \
      __enzyme_fwddiff((void *)FLUX, enzyme_dup, UL, dUL, enzyme_dup, UR, dUR, \
                       enzyme_const, n, enzyme_const, eos, enzyme_dup, F, dF); \
      for (int row = 0; row < NVARS; row++)                                    \
        AL[row * NVARS + col] = dF[row];                                       \
    }                                                                          \
                                                                               \
    for (int col = 0; col < NVARS; col++) {                                    \
      for (int i = 0; i < NVARS; i++) {                                        \
        dUL[i] = 0.0;                                                          \
        dUR[i] = 0.0;                                                          \
      }                                                                        \
      dUR[col] = 1.0;                                                          \
      __enzyme_fwddiff((void *)FLUX, enzyme_dup, UL, dUL, enzyme_dup, UR, dUR, \
                       enzyme_const, n, enzyme_const, eos, enzyme_dup, F, dF); \
      for (int row = 0; row < NVARS; row++)                                    \
        AR[row * NVARS + col] = dF[row];                                       \
    }                                                                          \
  }

DEFINE_FACE_BLOCKS(rusanov_face_blocks, rusanov_flux)
DEFINE_FACE_BLOCKS(hll_face_blocks, hll_flux)

int main() {
  EulerEos eos = {1.4, 0.0};

  for (int s = 0; s < EULER_NSTATES; s++) {
    const double *q = euler_primitives[s];

    for (int f = 0; f < EULER_NNORMALS; f++) {
      const double *n = euler_normals[f];

      double U[NVARS], A[NVARS * NVARS];
      double AL[NVARS * NVARS], AR[NVARS * NVARS], sum[NVARS * NVARS];
      double Fstar[NVARS], F[NVARS];

      euler_from_primitive(q[0], q[1], q[2], q[3], q[4], &eos, U);
      euler_flux(U, n, &eos, F);
      flux_jacobian(U, n, &eos, A);

      // The primal is consistent to begin with -- if this fails, the
      // derivative identity below would be gating the wrong thing.
      rusanov_flux(U, U, n, &eos, Fstar);
      APPROX_EQ(frechet_rel_error(Fstar, F, NVARS), 0.0, 1e-14);
      hll_flux(U, U, n, &eos, Fstar);
      APPROX_EQ(frechet_rel_error(Fstar, F, NVARS), 0.0, 1e-14);

      rusanov_face_blocks(U, U, n, &eos, AL, AR);
      for (int i = 0; i < NVARS * NVARS; i++)
        sum[i] = AL[i] + AR[i];
      double err_rusanov = frechet_rel_error(sum, A, NVARS * NVARS);

      hll_face_blocks(U, U, n, &eos, AL, AR);
      for (int i = 0; i < NVARS * NVARS; i++)
        sum[i] = AL[i] + AR[i];
      double err_hll = frechet_rel_error(sum, A, NVARS * NVARS);

      printf("state %d normal %d: rusanov %g  hll %g\n", s, f, err_rusanov,
             err_hll);
      APPROX_EQ(err_rusanov, 0.0, 1e-13);
      APPROX_EQ(err_hll, 0.0, 1e-13);

      // On a supersonic face, HLL degenerates to pure upwinding, so the right
      // block must be structurally absent rather than merely small.
      double un = euler_normal_velocity(U, n);
      if (un - euler_sound_speed(U, &eos) > 0.0) {
        for (int i = 0; i < NVARS * NVARS; i++)
          APPROX_EQ(AR[i], 0.0, 0.0);
        printf("state %d normal %d: hll fully upwind, right block exact zero\n",
               s, f);
      }
    }
  }

  printf("done\n");
  return 0;
}

// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Gate a full 5x5 flux Jacobian, built column by column with forward mode,
// against an exact algebraic identity of the primal -- no hardcoded derivative
// values and no finite differencing.
//
// For an ideal gas, the Euler normal flux is homogeneous by degree one in the
// conserved state, so Euler's homogeneous-function theorem gives
//
//   A(U) * U == F(U),   A = dF/dU,
//
// which validates all 25 entries at once.  This is the identity underneath
// Steger-Warming flux-vector splitting, and it must hold to machine precision.
//
// The stiffened-gas branch is the matched counterexample.  Writing p = p_ideal
// - gamma*p_inf splits the flux as F = F_ideal + dF, where dF is
//
//   dF = (0, -gamma*p_inf*nx, -gamma*p_inf*ny, -gamma*p_inf*nz, -gamma*p_inf*un)
//
// and is homogeneous of degree ZERO (the momentum rows are constant in U, and
// un = (rho*u . n)/rho is a ratio of degree-one quantities).  Euler's theorem
// applied degree-wise then predicts the failure exactly:
//
//   A(U) * U - F(U) == -dF,
//
// so the identity breaks by a closed-form amount that Enzyme must reproduce to
// machine precision. A wrong Jacobian fails both halves.

#include "../euler.h"
#include "../frechet.h"
#include "../test_utils.h"

extern void __enzyme_fwddiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

// A[row * NVARS + col] = dF[row]/dU[col], one seeded tangent per column.  The
// blocks are 5x5, so tall-thin forward mode is the right shape here.
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

void matvec(const double *A, const double *x, double *y) {
  for (int row = 0; row < NVARS; row++) {
    y[row] = 0.0;
    for (int col = 0; col < NVARS; col++)
      y[row] += A[row * NVARS + col] * x[col];
  }
}

int main() {
  for (int s = 0; s < EULER_NSTATES; s++) {
    const double *q = euler_primitives[s];

    for (int f = 0; f < EULER_NNORMALS; f++) {
      const double *n = euler_normals[f];

      // Ideal gas: A(U) * U == F(U).
      {
        EulerEos eos = {1.4, 0.0};
        double U[NVARS], F[NVARS], A[NVARS * NVARS], AU[NVARS];

        euler_from_primitive(q[0], q[1], q[2], q[3], q[4], &eos, U);
        euler_flux(U, n, &eos, F);
        flux_jacobian(U, n, &eos, A);
        matvec(A, U, AU);

        double err = frechet_rel_error(AU, F, NVARS);
        printf("state %d normal %d ideal     : rel err %g\n", s, f, err);
        APPROX_EQ(err, 0.0, 1e-13);
      }

      // Stiffened gas: A(U) * U - F(U) == -dF, exactly.  p_inf is scaled to the
      // state's own pressure so every state stays physical.
      {
        EulerEos eos = {1.4, 0.5 * q[4]};
        double U[NVARS], F[NVARS], A[NVARS * NVARS], AU[NVARS];
        double deviation[NVARS], expected[NVARS];

        euler_from_primitive(q[0], q[1], q[2], q[3], q[4], &eos, U);
        euler_flux(U, n, &eos, F);
        flux_jacobian(U, n, &eos, A);
        matvec(A, U, AU);

        double gp = eos.gamma * eos.p_inf;
        double un = euler_normal_velocity(U, n);
        expected[RHO] = 0.0;
        expected[RHOU] = gp * n[0];
        expected[RHOV] = gp * n[1];
        expected[RHOW] = gp * n[2];
        expected[RHOE] = gp * un;

        for (int i = 0; i < NVARS; i++)
          deviation[i] = AU[i] - F[i];

        double err = frechet_rel_error(deviation, expected, NVARS);
        printf("state %d normal %d stiffened : rel err %g\n", s, f, err);
        APPROX_EQ(err, 0.0, 1e-13);
      }
    }
  }

  printf("done\n");
  return 0;
}

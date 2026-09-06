// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Reverse mode through the same data-dependent Newton solve that
// ForwardMode/newton_inversion.c covers, which means taping a loop whose trip
// count is not known until it runs: it breaks on a converged step, can break
// instead on a collapsed heat capacity, and clamps its iterate every pass.
//
// The whole thermodynamic state goes in as one vector so a single reverse sweep
// returns every sensitivity at once,
//
//   x = (e, Y_0, Y_1, T_guess),
//   dT/dx = (1/cv(T), -e_0(T)/cv(T), -e_1(T)/cv(T), 0),
//
// each entry known in closed form from the implicit function theorem.  The
// trailing zero is the interesting one: it says the taped adjoint of a
// converged iteration carries no memory of where the iteration started.
//
// The file also cross-checks reverse against forward on the identical kernel.
// That comparison needs no reference values and no finite differencing at all
// -- the two modes traverse the loop in opposite directions, over a tape in one
// case and not the other, so agreement to machine precision is a much tighter
// statement than either mode agreeing with a closed form.

#include "../frechet.h"
#include "../test_utils.h"
#include "../thermo.h"

extern double __enzyme_autodiff(void *, ...);
extern double __enzyme_fwddiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

#define NINPUT 4

// x = (e_target, Y_0, Y_1, T_guess).
double temperature_of(const double *x, int test_before_update, int polish) {
  return thermo_temperature(x[0], x + 1, x[3], test_before_update, polish);
}

void gradient_reverse(const double *x, int test_before_update, int polish,
                      double *g) {
  for (int i = 0; i < NINPUT; i++)
    g[i] = 0.0;

  __enzyme_autodiff((void *)temperature_of, enzyme_dup, x, g, enzyme_const,
                    test_before_update, enzyme_const, polish);
}

void gradient_forward(const double *x, int test_before_update, int polish,
                      double *g) {
  double dx[NINPUT];

  for (int col = 0; col < NINPUT; col++) {
    for (int i = 0; i < NINPUT; i++)
      dx[i] = 0.0;
    dx[col] = 1.0;

    g[col] = __enzyme_fwddiff((void *)temperature_of, enzyme_dup, x, dx,
                              enzyme_const, test_before_update, enzyme_const,
                              polish);
  }
}

double rel(double got, double want) { return fabs(got - want) / fabs(want); }

int main() {
  const double mixtures[2][NSPECIES] = {{0.767, 0.233}, {0.5, 0.5}};
  const double temperatures[] = {200.0, 300.0, 1000.0, 3000.0, 8000.0, 15000.0};
  const int ntemp = sizeof(temperatures) / sizeof(temperatures[0]);

  for (int m = 0; m < 2; m++) {
    for (int t = 0; t < ntemp; t++) {
      double T_ref = temperatures[t];
      double x[NINPUT];

      x[0] = thermo_energy(T_ref, mixtures[m]);
      x[1] = mixtures[m][0];
      x[2] = mixtures[m][1];
      x[3] = 500.0; // deliberately poor guess, so the loop iterates

      double T = temperature_of(x, THERMO_TEST_AFTER_UPDATE, 0);
      APPROX_EQ(rel(T, T_ref), 0.0, 1e-12);

      double cv = thermo_cv(T, x + 1);
      double want[NINPUT];
      want[0] = 1.0 / cv;
      want[1] = -thermo_energy_species(0, T) / cv;
      want[2] = -thermo_energy_species(1, T) / cv;
      want[3] = 0.0;

      double rev[NINPUT], fwd[NINPUT];
      gradient_reverse(x, THERMO_TEST_AFTER_UPDATE, 0, rev);
      gradient_forward(x, THERMO_TEST_AFTER_UPDATE, 0, fwd);

      printf("mix %d T %7.1f: dT/de %.6e dT/dY0 %.6e dT/dY1 %.6e "
             "dT/dTguess %.3e\n",
             m, T_ref, rev[0], rev[1], rev[2], rev[3]);

      // Against the implicit-function answer.
      for (int i = 0; i < 3; i++)
        APPROX_EQ(rel(rev[i], want[i]), 0.0, 1e-11);

      // The adjoint keeps no memory of the initial guess.
      APPROX_EQ(rev[3], 0.0, 1e-13);

      // Reverse and forward have to agree with each other, not just with the
      // closed form.  Scaled against the gradient as a whole, since one of its
      // entries is legitimately zero.
      APPROX_EQ(frechet_rel_error(rev, fwd, NINPUT), 0.0, 1e-13);

      // The polish leaves a converged adjoint alone rather than perturbing it.
      double polished[NINPUT];
      gradient_reverse(x, THERMO_TEST_AFTER_UPDATE, 1, polished);
      for (int i = 0; i < 3; i++)
        APPROX_EQ(rel(polished[i], want[i]), 0.0, 1e-11);
    }
  }

  // The clamped branches tape correctly too: no adjoint flows out of a
  // saturated fmax/fmin, on any input.
  {
    double x[NINPUT] = {-1.0e5, 0.767, 0.233, 500.0};
    double g[NINPUT];
    double T = temperature_of(x, THERMO_TEST_AFTER_UPDATE, 1);
    gradient_reverse(x, THERMO_TEST_AFTER_UPDATE, 1, g);
    printf("clamped low : T %g grad %g %g %g %g\n", T, g[0], g[1], g[2], g[3]);
    APPROX_EQ(T, THERMO_T_MIN, 0.0);
    for (int i = 0; i < NINPUT; i++)
      APPROX_EQ(g[i], 0.0, 0.0);
  }
  {
    double x[NINPUT] = {1.0e12, 0.767, 0.233, 500.0};
    double g[NINPUT];
    double T = temperature_of(x, THERMO_TEST_AFTER_UPDATE, 1);
    gradient_reverse(x, THERMO_TEST_AFTER_UPDATE, 1, g);
    printf("clamped high: T %g grad %g %g %g %g\n", T, g[0], g[1], g[2], g[3]);
    APPROX_EQ(T, THERMO_T_MAX, 0.0);
    for (int i = 0; i < NINPUT; i++)
      APPROX_EQ(g[i], 0.0, 0.0);
  }

  printf("done\n");
  return 0;
}

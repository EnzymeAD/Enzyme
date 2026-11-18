// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang -std=c11 -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -
// RUN: %clang -std=c11 -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -enzyme-inline=1 -S | %lli -

// Forward mode through a Newton solve whose trip count is data dependent.
//
// The loop runs up to 50 times, breaks early once the step is small, can break
// instead on a collapsed heat capacity, and clamps its iterate to a physical
// range every pass -- so the differentiated control flow is decided at runtime.
// The answer is fixed by the implicit function theorem and never mentions the
// iterates:
//
//   dT/de        =  1 / cv(T),
//   dT/dY_s      = -e_s(T) / cv(T),
//   dT/dT_guess ==  0.
//
// The last one is the sharpest gate in the file.  A converged root cannot
// depend on where the iteration started, so the tangent with respect to the
// initial guess has to vanish identically -- not to the solver tolerance, to
// machine precision.  It fails loudly for any AD that hands back the derivative
// of the final iterate in place of the derivative of the solution.
//
// The rest of the test measures what decides that.  A Newton step applied at an
// already-converged iterate discards whatever tangent the loop accumulated (see
// thermo.h), so everything hinges on whether the last step lands there:
//
//   1. test-after-update  -- quadratic convergence means the loop always runs
//      one full step past the point a 1e-6 test could trip, so it performs the
//      collapse itself and the tangent comes back at machine precision.
//   2. test-before-update -- that final step is skipped.  Both the primal and
//      the tangent are left at the step tolerance, the tangent roughly an order
//      of magnitude worse, so the derivative is the least accurate thing a
//      converged-looking solve returns.  Gated as an inequality, so the file
//      fails if it stops measuring that.
//   3. test-before-update + polish -- machine precision again, independent of
//      where the test was placed.
//
// Finally, on the temperature clamps the derivative is exactly zero: fmax/fmin
// have selected a constant branch and no tangent flows through it.

#include "../test_utils.h"
#include "../thermo.h"

extern double __enzyme_fwddiff(void *, ...);
extern int enzyme_dup;
extern int enzyme_const;

double dT_denergy(double e_target, const double *Y, double T_guess,
                  int test_before_update, int polish) {
  return __enzyme_fwddiff((void *)thermo_temperature, e_target, 1.0,
                          enzyme_const, Y, enzyme_const, T_guess, enzyme_const,
                          test_before_update, enzyme_const, polish);
}

double dT_dmassfraction(double e_target, const double *Y, double T_guess,
                        int test_before_update, int polish, int s) {
  double dY[NSPECIES];
  for (int i = 0; i < NSPECIES; i++)
    dY[i] = 0.0;
  dY[s] = 1.0;

  return __enzyme_fwddiff((void *)thermo_temperature, e_target, 0.0, enzyme_dup,
                          Y, dY, enzyme_const, T_guess, enzyme_const,
                          test_before_update, enzyme_const, polish);
}

double dT_dguess(double e_target, const double *Y, double T_guess,
                 int test_before_update, int polish) {
  return __enzyme_fwddiff((void *)thermo_temperature, e_target, 0.0,
                          enzyme_const, Y, T_guess, 1.0, enzyme_const,
                          test_before_update, enzyme_const, polish);
}

// Relative deviation of `got` from a nonzero reference.
double rel(double got, double want) { return fabs(got - want) / fabs(want); }

int main() {
  const double air[NSPECIES] = {0.767, 0.233};
  const double even[NSPECIES] = {0.5, 0.5};
  const double *mixtures[2] = {air, even};

  const double temperatures[] = {200.0, 300.0, 1000.0, 3000.0, 8000.0, 15000.0};
  const int ntemp = sizeof(temperatures) / sizeof(temperatures[0]);

  // Deliberately poor initial guess so the loop actually iterates.
  const double T_guess = 500.0;

  double worst_stale = 0.0;

  for (int m = 0; m < 2; m++) {
    const double *Y = mixtures[m];

    for (int t = 0; t < ntemp; t++) {
      double T_ref = temperatures[t];
      double e_target = thermo_energy(T_ref, Y);

      double T =
          thermo_temperature(e_target, Y, T_guess, THERMO_TEST_AFTER_UPDATE, 0);
      APPROX_EQ(rel(T, T_ref), 0.0, 1e-12);

      double cv = thermo_cv(T, Y);
      double want = 1.0 / cv;

      double after =
          dT_denergy(e_target, Y, T_guess, THERMO_TEST_AFTER_UPDATE, 0);
      double before =
          dT_denergy(e_target, Y, T_guess, THERMO_TEST_BEFORE_UPDATE, 0);
      double repaired =
          dT_denergy(e_target, Y, T_guess, THERMO_TEST_BEFORE_UPDATE, 1);

      double err_after = rel(after, want);
      double err_before = rel(before, want);
      double err_repaired = rel(repaired, want);

      // The primal the stale tangent sits next to, for comparison.
      double T_stale = thermo_temperature(e_target, Y, T_guess,
                                          THERMO_TEST_BEFORE_UPDATE, 0);
      double err_primal = rel(T_stale, T_ref);

      printf("mix %d T %7.1f: dT/de after %.2e before %.2e repaired %.2e | "
             "primal %.2e\n",
             m, T_ref, err_after, err_before, err_repaired, err_primal);

      APPROX_EQ(err_after, 0.0, 1e-13);
      APPROX_EQ(err_repaired, 0.0, 1e-13);
      APPROX_EQ(err_primal, 0.0, 1e-7);

      if (err_before > worst_stale)
        worst_stale = err_before;

      // dT/dY_s against -e_s(T)/cv(T), through the same iterative kernel.
      for (int s = 0; s < NSPECIES; s++) {
        double got = dT_dmassfraction(e_target, Y, T_guess,
                                      THERMO_TEST_AFTER_UPDATE, 0, s);
        APPROX_EQ(rel(got, -thermo_energy_species(s, T) / cv), 0.0, 1e-11);
      }

      // The converged root cannot depend on where the iteration started.
      double dguess =
          dT_dguess(e_target, Y, T_guess, THERMO_TEST_AFTER_UPDATE, 0);
      APPROX_EQ(dguess, 0.0, 1e-13);
    }
  }

  // The stale tangent has to be visibly stale somewhere in the sweep, or this
  // file has stopped testing what it claims to.  It is worst where cv varies
  // fastest, which is what the uncollapsed term is proportional to.
  printf("worst stale tangent: %.3e\n", worst_stale);
  if (worst_stale < 1.0e-9) {
    fprintf(stderr,
            "no measurably stale tangent anywhere in the sweep (worst %.3e) -- "
            "the loop is no longer stopping on its step tolerance\n",
            worst_stale);
    abort();
  }

  // On the clamps the derivative is structurally zero, not merely small.
  {
    double T =
        thermo_temperature(-1.0e5, air, T_guess, THERMO_TEST_AFTER_UPDATE, 1);
    double dT = dT_denergy(-1.0e5, air, T_guess, THERMO_TEST_AFTER_UPDATE, 1);
    printf("clamped low : T %g dT/de %g\n", T, dT);
    APPROX_EQ(T, THERMO_T_MIN, 0.0);
    APPROX_EQ(dT, 0.0, 0.0);
  }
  {
    double T =
        thermo_temperature(1.0e12, air, T_guess, THERMO_TEST_AFTER_UPDATE, 1);
    double dT = dT_denergy(1.0e12, air, T_guess, THERMO_TEST_AFTER_UPDATE, 1);
    printf("clamped high: T %g dT/de %g\n", T, dT);
    APPROX_EQ(T, THERMO_T_MAX, 0.0);
    APPROX_EQ(dT, 0.0, 0.0);
  }

  printf("done\n");
  return 0;
}

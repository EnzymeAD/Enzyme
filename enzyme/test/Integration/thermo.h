// Thermally perfect thermodynamics and the temperature inversion that goes with
// it, shared by the forward- and reverse-mode Newton-inversion tests.
//
// Each species carries a rigid-rotor translational-rotational energy plus one
// simple-harmonic-oscillator vibrational mode,
//
//   e_s(T) = R_s * (5/2 * T + theta_s / (exp(theta_s/T) - 1)),
//
// so the mixture energy is a genuinely nonlinear, non-invertible-in-closed-form
// function of temperature -- exactly the situation that forces a solver into a
// Newton loop, and the reason a CFD residual has an iterative kernel sitting in
// the middle of its differentiable path.
//
// The useful property for AD is that the loop has an exact answer that never
// mentions the iterates.  Implicit differentiation of e_mix(T, Y) == e gives
//
//   dT/de     =  1 / cv_mix(T, Y),
//   dT/dY_s   = -e_s(T) / cv_mix(T, Y),
//
// so the derivative can be gated with no finite differencing and no dependence
// on how the loop happened to terminate.

#include <math.h>

#define NSPECIES 2

#define THERMO_T_MIN 50.0
#define THERMO_T_MAX 50000.0
#define THERMO_MAX_ITER 50
#define THERMO_REL_TOL 1.0e-6
#define THERMO_CV_FLOOR 1.0e-8

// N2 and O2: gas constant [J/(kg K)] and characteristic vibrational
// temperature [K].
const double thermo_R[NSPECIES] = {296.8035, 259.8367};
const double thermo_theta[NSPECIES] = {3395.0, 2239.0};

// Species internal energy [J/kg].
double thermo_energy_species(int s, double T) {
  double x = thermo_theta[s] / T;
  return thermo_R[s] * (2.5 * T + thermo_theta[s] / (exp(x) - 1.0));
}

// Species specific heat at constant volume [J/(kg K)] -- d/dT of the above.
double thermo_cv_species(int s, double T) {
  double x = thermo_theta[s] / T;
  double ex = exp(x);
  double den = ex - 1.0;
  return thermo_R[s] * (2.5 + x * x * ex / (den * den));
}

double thermo_energy(double T, const double *Y) {
  double e = 0.0;
  for (int s = 0; s < NSPECIES; s++)
    e += Y[s] * thermo_energy_species(s, T);
  return e;
}

double thermo_cv(double T, const double *Y) {
  double cv = 0.0;
  for (int s = 0; s < NSPECIES; s++)
    cv += Y[s] * thermo_cv_species(s, T);
  return cv;
}

// Where the convergence test sits relative to the Newton update.  The two
// placements are equally common in solver code and indistinguishable for the
// primal, but they are NOT equivalent under differentiation -- see below.
#define THERMO_TEST_AFTER_UPDATE 0
#define THERMO_TEST_BEFORE_UPDATE 1

// Newton solve for T given the mixture internal energy, clamped to a physical
// range.  The trip count is data dependent: the loop breaks early on a
// converged step, and can break on a collapsed heat capacity instead.
//
// Differentiating the update
//
//   T' = T - (e_mix(T, Y) - e) / cv_mix(T, Y)
//
// gives dT' = 1/cv + (e_mix(T,Y) - e) * cv'/cv^2 * dT, so the incoming tangent
// survives only multiplied by the residual.  One Newton step applied at an
// already-converged iterate therefore DISCARDS whatever the loop accumulated
// and lands on the implicit-function answer.  Everything about this kernel's AD
// accuracy follows from where that last step falls:
//
//   * test-after-update runs one full iteration past convergence -- quadratic
//     convergence means the step that trips a 1e-6 test is itself around 1e-13
//     -- so the loop performs the collapse on its own and the tangent comes out
//     at machine precision.  `polish` is then redundant.
//   * test-before-update skips that final step and returns the previous
//     iterate.  Both the primal and the tangent are left at the step tolerance,
//     and the tangent is the worse of the two by roughly an order of magnitude,
//     because it carries the residual's error on top of the state's.
//   * `polish` restores both to machine precision, making the result
//     independent of where the test was placed.
//
// So the derivative is the least accurate thing a converged-looking solve
// returns, which is why an iterative kernel wants its tangent collapsed
// explicitly rather than by luck of loop structure.
double thermo_temperature(double e_target, const double *Y, double T_guess,
                          int test_before_update, int polish) {
  double T = fmax(T_guess, 100.0);

  for (int iter = 0; iter < THERMO_MAX_ITER; iter++) {
    double res = thermo_energy(T, Y) - e_target;
    double cv = thermo_cv(T, Y);
    if (cv < THERMO_CV_FLOOR)
      break;

    double dT = res / cv;
    if (test_before_update && fabs(dT) < THERMO_REL_TOL * T)
      break;

    T -= dT;
    T = fmax(T, THERMO_T_MIN);
    T = fmin(T, THERMO_T_MAX);

    if (!test_before_update && fabs(dT) < THERMO_REL_TOL * T)
      break;
  }

  if (polish) {
    double cv = thermo_cv(T, Y);
    if (cv >= THERMO_CV_FLOOR) {
      T -= (thermo_energy(T, Y) - e_target) / cv;
      T = fmax(T, THERMO_T_MIN);
      T = fmin(T, THERMO_T_MAX);
    }
  }

  return T;
}

// Compressible Euler physics shared by the CFD-derived integration tests.
//
// The conserved state is U = (rho, rho*u, rho*v, rho*w, rho*E) and the closure
// is the stiffened-gas (Tammann) equation of state
//
//   p = (gamma - 1) * (rho*E - 0.5*rho*|u|^2) - gamma*p_inf,
//
// which reduces to the ideal gas when p_inf == 0.  Both branches live in one
// closure deliberately: the ideal gas makes the normal flux homogeneous of
// degree one in U, and p_inf breaks that homogeneity by an exactly known
// amount.  ForwardMode/euler_homogeneity.c gates an Enzyme-built 5x5 flux
// Jacobian against both facts.

#include <math.h>

enum { RHO = 0, RHOU = 1, RHOV = 2, RHOW = 3, RHOE = 4, NVARS = 5 };

typedef struct {
  double gamma;
  double p_inf;
} EulerEos;

// Static pressure from the conserved state.
double euler_pressure(const double *U, const EulerEos *eos) {
  double inv = 1.0 / U[RHO];
  double ke =
      0.5 * inv * (U[RHOU] * U[RHOU] + U[RHOV] * U[RHOV] + U[RHOW] * U[RHOW]);
  return (eos->gamma - 1.0) * (U[RHOE] - ke) - eos->gamma * eos->p_inf;
}

// Frozen speed of sound.
double euler_sound_speed(const double *U, const EulerEos *eos) {
  double p = euler_pressure(U, eos);
  return sqrt(eos->gamma * (p + eos->p_inf) / U[RHO]);
}

// Velocity projected onto the face normal.
double euler_normal_velocity(const double *U, const double *n) {
  return (U[RHOU] * n[0] + U[RHOV] * n[1] + U[RHOW] * n[2]) / U[RHO];
}

// Normal flux F(U).n through a face with unit normal n.
void euler_flux(const double *U, const double *n, const EulerEos *eos,
                double *F) {
  double un = euler_normal_velocity(U, n);
  double p = euler_pressure(U, eos);
  F[RHO] = U[RHO] * un;
  F[RHOU] = U[RHOU] * un + p * n[0];
  F[RHOV] = U[RHOV] * un + p * n[1];
  F[RHOW] = U[RHOW] * un + p * n[2];
  F[RHOE] = (U[RHOE] + p) * un;
}

// Assemble a conserved state from the primitives (rho, u, v, w, p).
void euler_from_primitive(double rho, double u, double v, double w, double p,
                          const EulerEos *eos, double *U) {
  U[RHO] = rho;
  U[RHOU] = rho * u;
  U[RHOV] = rho * v;
  U[RHOW] = rho * w;
  U[RHOE] = (p + eos->gamma * eos->p_inf) / (eos->gamma - 1.0) +
            0.5 * rho * (u * u + v * v + w * w);
}

// Representative states, in primitive form (rho, u, v, w, p).  These are the
// real operating range of a hypersonic solver rather than round numbers: the
// Mach-8 freestream carries rho ~ 1e-2 against rho*E ~ 1e5, so a probe
// direction that ignores the per-variable scale leaves the linear regime in
// one component while barely moving another.
#define EULER_NSTATES 4
const double euler_primitives[EULER_NSTATES][5] = {
    {1.225, 50.0, 10.0, -5.0, 101325.0},   // sea-level subsonic
    {0.0184, 2400.0, 60.0, 0.0, 1197.0},   // Mach 8 at 30 km
    {0.1400, 300.0, -120.0, 45.0, 1.6e5},  // post-shock, hot and slow
    {1.0e-3, 1500.0, 0.0, -200.0, 25.0},   // strong expansion, near vacuum
};

// Face normals: axis-aligned, oblique, and fully three-dimensional.
#define EULER_NNORMALS 3
const double euler_normals[EULER_NNORMALS][3] = {
    {1.0, 0.0, 0.0},
    {0.6, -0.8, 0.0},
    {0.4242640687119285, 0.5656854249492380, -0.7071067811865476},
};

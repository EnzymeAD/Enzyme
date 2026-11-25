// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t/inc/poseidon && cp %S/../../../include/poseidon/poseidon.h %t/inc/poseidon/
// RUN: %clang -x c++ -O2 -c -DOTHER_TU %s -o %t/other.o
// RUN: %clang -x c++ -O2 -ffp-contract=on -I%t/inc %s -x none %t/other.o -lstdc++ -lm -o %t/stock
// RUN: %t/stock > %t/stock.out
//
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c -poseidon-print %s %t/other.o -o %t/gen 2>&1 | FileCheck --check-prefix=GEN %s
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > %t/gen.out
// RUN: diff %t/stock.out %t/gen.out
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/prof -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -poseidon-explain=%t/x -Rpass=poseidon %s %t/other.o -poseidon-print -o %t/use 2> %t/use.err
// RUN: FileCheck --check-prefix=USE %s < %t/use.err
// RUN: FileCheck --check-prefix=REMARK --implicit-check-not=remark: %s < %t/use.err
// RUN: %t/use > %t/use.out
// RUN: FileCheck --check-prefix=OUT %s < %t/use.out
// RUN: not diff %t/stock.out %t/use.out
//
// RUN: %poseidon_clangxx -O0 -poseidon-profile-generate -poseidon-cache=%t/c0 %s %t/other.o -o %t/gen0
// RUN: env POSEIDON_PROFILE_DIR=%t/prof0 %t/gen0 > %t/gen0.out
// RUN: diff %t/stock.out %t/gen0.out
// RUN: %poseidon_clangxx -O0 -poseidon-profile-use=%t/prof0 -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c0 %s %t/other.o -poseidon-print -o %t/use0 2>&1 | FileCheck --check-prefix=LEVEL %s
// RUN: %poseidon_clangxx -O3 -poseidon-profile-generate -poseidon-cache=%t/c3 %s %t/other.o -o %t/gen3
// RUN: env POSEIDON_PROFILE_DIR=%t/prof3 %t/gen3 > %t/gen3.out
// RUN: diff %t/stock.out %t/gen3.out
// RUN: %poseidon_clangxx -O3 -poseidon-profile-use=%t/prof3 -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c3 %s %t/other.o -poseidon-print -o %t/use3 2>&1 | FileCheck --check-prefix=LEVEL %s

// POSEIDON_OPTIMIZE on host functions: a free function called directly,
// through a pointer and from another translation unit, a member function, a
// template instantiation, a floating-point and an integer return. The
// profiling run computes what the program as written computes; the solve
// rewrites every call, at every optimization level, and reports at the source
// line and column inside the function.

#ifdef OTHER_TU
void wave(const double *x, double *y, int n);
void wave_elsewhere(const double *x, double *y, int n) { wave(x, y, n); }
#else
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <vector>

POSEIDON_OPTIMIZE void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    // REMARK: host_site_flow.cpp:[[@LINE+1]]:26: remark: Poseidon runs all 6 operations in float: (double)(expf((float)x[i] * ((float)x[i] * -0.5f)) * cosf((float)x[i] * 4.0f)); modelled error 7.1e-18 -> 1.2e-08, cost 1 -> 0.3125 (-69%) [PT0_0] [-Rpass=poseidon]
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

POSEIDON_OPTIMIZE_TAU(1e-6) double energy(const double *x, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i)
    s += 0.5 * x[i] * x[i] + std::sin(x[i]);
  return s;
}

struct Integrator {
  double dt;
  POSEIDON_OPTIMIZE void step(double *q, const double *p, int n);
};
void Integrator::step(double *q, const double *p, int n) {
  for (int i = 0; i < n; ++i)
    q[i] += dt * std::sqrt(p[i] * p[i] + 1.0);
}

template <typename T> POSEIDON_OPTIMIZE T poly(T x) { return std::exp(x) * x - 2 * x + 1; }

POSEIDON_OPTIMIZE int clip(const double *x, double *y, int n) {
  int c = 0;
  for (int i = 0; i < n; ++i) {
    y[i] = std::cos(x[i]) * x[i];
    c += y[i] > 0.25;
  }
  return c;
}

void wave_elsewhere(const double *x, double *y, int n);

int main(int argc, char **argv) {
  const int n = 512;
  std::vector<double> x(n), y(n), w(n), z(n), q(n, 1.0);
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 + 4.0 * i / (n - 1);
  void (*fp)(const double *, double *, int) = argc > 5 ? nullptr : wave;
  fp(x.data(), y.data(), n);
  wave_elsewhere(x.data(), w.data(), n);
  Integrator in{1e-3};
  in.step(q.data(), x.data(), n);
  double e = energy(x.data(), n);
  double p = poly(0.25) + poly(0.5);
  int c = clip(x.data(), z.data(), n);
  printf("%.10g %.10g %.10g %.10g %.10g %d\n", y[100], w[100], q[100], e, p, c);
}
#endif

// GEN: [poseidon] site energy(double const*, int): accuracy target 1e-06
// GEN-NEXT: [poseidon-profgen] wave(double const*, double*, int): site 0, 2 shadow buffer(s), output(s) 1, 1 call(s)
// GEN-NEXT: [poseidon-profgen] energy(double const*, int): site 1, 1 shadow buffer(s), output(s) return, 2 call(s)
// GEN-NEXT: [poseidon-profgen] Integrator::step(double*, double const*, int): site 2, 3 shadow buffer(s), output(s) 1, 2 call(s)
// GEN-NEXT: [poseidon-profgen] clip(double const*, double*, int): site 3, 2 shadow buffer(s), output(s) 1 return, 2 call(s)
// GEN-NEXT: [poseidon-profgen] double poly<double>(double): site 4, 0 shadow buffer(s), output(s) return, 3 call(s)
// GEN-NOT: {{.}}

// USE: [poseidon] preprocess__Z4wavePKdPdi_poseidon_body: tau=1.000000e-06
// USE: Applying solution for CS: All FP64(0%) + FP32(100%)
// USE: [poseidon] pin: _Z4wavePKdPdi:PT0=All FP64(0%) + FP32(100%)
// USE: [poseidon] preprocess__Z6energyPKdi_poseidon_body: the site's own accuracy target 1.000000e-06 overrides -poseidon-tau=1.000000e-06
// USE: Applying solution for CS: All FP64(0%) + FP32(100%)
// USE: [poseidon] pin: _Z6energyPKdi:PT0=All FP64(0%) + FP32(100%)
// USE: [poseidon] preprocess__ZN10Integrator4stepEPdPKdi_poseidon_body: tau=1.000000e-06
// USE: Applying solution for CS: All FP64(0%) + FP32(100%)
// USE: [poseidon] pin: _ZN10Integrator4stepEPdPKdi:PT0=All FP64(0%) + FP32(100%)
// USE: [poseidon] preprocess__Z4clipPKdPdi_poseidon_body: tau=1.000000e-06
// USE: Applying solution for CS: All FP64(0%) + FP32(100%)
// USE: [poseidon] pin: _Z4clipPKdPdi:PT0=All FP64(0%) + FP32(100%)
// USE: [poseidon] preprocess__Z4polyIdET_S0__poseidon_body: tau=1.000000e-06
// USE: Applying solution for CS: All FP64(0%) + FP32(100%)
// USE: [poseidon] pin: _Z4polyIdET_S0_:PT0=All FP64(0%) + FP32(100%)
// USE-NOT: Applying solution

// REMARK: host_site_flow.cpp:[[@LINE-68]]:14: remark: Poseidon runs all 4 operations in float: (double)((float)? + ((float)x[i] * 0.5f * (float)x[i] + sinf((float)x[i]))); modelled error 8.3e-19 -> 7.4e-08, cost 1 -> 0.3 (-70%) [PT0_0] [-Rpass=poseidon]
// REMARK: host_site_flow.cpp:[[@LINE-59]]:40: remark: Poseidon runs all 3 operations in float: (double)((float)? * sqrtf((float)p[i] * (float)p[i] + 1.0f) + (float)q[i]); modelled error 0 -> 2.9e-08, cost 1 -> 0.583333 (-42%) [PT0_0] [-Rpass=poseidon]
// REMARK: host_site_flow.cpp:[[@LINE-52]]:12: remark: Poseidon runs all 2 operations in float: (double)((float)x[i] * cosf((float)x[i])); modelled error 1.8e-17 -> 3.7e-08, cost 1 -> 0.416667 (-58%) [PT0_0] [-Rpass=poseidon]
// REMARK: host_site_flow.cpp:[[@LINE-58]]:62: remark: Poseidon runs all 4 operations in float: (double)(expf((float)x) * (float)x + (float)x * -2.0f + 1.0f); modelled error 1.1e-17 -> 8.2e-09, cost 1 -> 0.35 (-65%) [PT0_0] [-Rpass=poseidon]
// REMARK-NOT: remark:

// The call through the pointer and the one from the other translation unit
// run the same rewritten body.
// OUT: [[Y:[-0-9.e]+]] [[Y]]{{ }}

// At every level each site is rewritten; which mix of FP32 and FP64 a site
// gets depends on the profile of that level's program.
// LEVEL: Applying solution for CS:
// LEVEL: [poseidon] pin: _Z4wavePKdPdi:PT0=All FP64({{[0-9]+}}%) + FP32({{[0-9]+}}%)
// LEVEL: Applying solution for CS:
// LEVEL: [poseidon] pin: _Z6energyPKdi:PT0=All FP64({{[0-9]+}}%) + FP32({{[0-9]+}}%)
// LEVEL: Applying solution for CS:
// LEVEL: [poseidon] pin: _ZN10Integrator4stepEPdPKdi:PT0=All FP64({{[0-9]+}}%) + FP32({{[0-9]+}}%)
// LEVEL: Applying solution for CS:
// LEVEL: [poseidon] pin: _Z4clipPKdPdi:PT0=All FP64({{[0-9]+}}%) + FP32({{[0-9]+}}%)
// LEVEL: Applying solution for CS:
// LEVEL: [poseidon] pin: _Z4polyIdET_S0_:PT0=All FP64({{[0-9]+}}%) + FP32({{[0-9]+}}%)
// LEVEL-NOT: Applying solution

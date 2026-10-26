// RUN: rm -rf %t && mkdir -p %t/inc/poseidon && cp %S/../../include/poseidon/poseidon.h %t/inc/poseidon/
// RUN: %clang -x c++ -O2 -c -DOTHER_TU %s -o %t/other.o
// RUN: %clang -x c++ -O2 -ffp-contract=on -I%t/inc %s -x none %t/other.o -lstdc++ -lm -o %t/stock
// RUN: %t/stock > %t/stock.out
//
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c %s %t/other.o -o %t/gen 2> %t/gen.err
// RUN: FileCheck --check-prefix=GEN %s < %t/gen.err
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen > %t/gen.out
// RUN: diff %t/stock.out %t/gen.out
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/prof -poseidon-tau=1e-6 -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -poseidon-explain=%t/x -Rpass=poseidon %s %t/other.o -o %t/use 2> %t/use.err
// RUN: FileCheck --check-prefix=USE %s < %t/use.err
// RUN: FileCheck --check-prefix=REMARK %s < %t/use.err
// RUN: %t/use > %t/use.out
// RUN: FileCheck --check-prefix=OUT %s < %t/use.out
// RUN: not diff %t/stock.out %t/use.out
// RUN: %poseidon_explain %t/x --function wave --format source --source %s --clang %clang --clang-arg=-I%t/inc | FileCheck --check-prefix=SOURCE %s
//
// RUN: %poseidon_clangxx -O0 -poseidon-profile-generate -poseidon-cache=%t/c0 %s %t/other.o -o %t/gen0 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof0 %t/gen0 > %t/gen0.out
// RUN: diff %t/stock.out %t/gen0.out
// RUN: %poseidon_clangxx -O0 -poseidon-profile-use=%t/prof0 -poseidon-tau=1e-6 -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c0 %s %t/other.o -o %t/use0 2>&1 | FileCheck --check-prefix=LEVEL %s
// RUN: %poseidon_clangxx -O1 -poseidon-profile-generate -poseidon-cache=%t/c1 %s %t/other.o -o %t/gen1 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof1 %t/gen1 > %t/gen1.out
// RUN: %poseidon_clangxx -O1 -poseidon-profile-use=%t/prof1 -poseidon-tau=1e-6 -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c1 %s %t/other.o -o %t/use1 2>&1 | FileCheck --check-prefix=LEVEL %s
// RUN: %poseidon_clangxx -O3 -poseidon-profile-generate -poseidon-cache=%t/c3 %s %t/other.o -o %t/gen3 2>/dev/null
// RUN: env POSEIDON_PROFILE_DIR=%t/prof3 %t/gen3 > %t/gen3.out
// RUN: %poseidon_clangxx -O3 -poseidon-profile-use=%t/prof3 -poseidon-tau=1e-6 -poseidon-cost-model=%S/Inputs/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c3 %s %t/other.o -o %t/use3 2>&1 | FileCheck --check-prefix=LEVEL %s
// REQUIRES: poseidon, enzyme

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
    // REMARK: host_site_flow.cpp:[[@LINE+1]]:{{[0-9]+}}: remark: Poseidon runs
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

// GEN-DAG: [poseidon-profgen] wave(double const*, double*, int): site {{[0-9]}}, 2 shadow buffer(s), output(s) 1, 1 call(s)
// GEN-DAG: [poseidon-profgen] energy(double const*, int): site {{[0-9]}}, 1 shadow buffer(s), output(s) return
// GEN-DAG: [poseidon-profgen] Integrator::step(double*, double const*, int): site {{[0-9]}}, 3 shadow buffer(s), output(s) 1
// GEN-DAG: [poseidon-profgen] double poly<double>(double): site {{[0-9]}}, 0 shadow buffer(s), output(s) return
// GEN-DAG: [poseidon-profgen] clip(double const*, double*, int): site {{[0-9]}}, 2 shadow buffer(s), output(s) 1 return

// USE-DAG: preprocess__Z4wavePKdPdi_poseidon_body: tau=1.000000e-06
// USE-DAG: preprocess__Z6energyPKdi_poseidon_body: the site's own accuracy target 1.000000e-06 overrides -poseidon-tau=1.000000e-06
// USE-DAG: preprocess__ZN10Integrator4stepEPdPKdi_poseidon_body: tau=1.000000e-06
// USE-DAG: preprocess__Z4polyIdET_S0__poseidon_body: tau=1.000000e-06
// USE-DAG: preprocess__Z4clipPKdPdi_poseidon_body: tau=1.000000e-06

// The call through the pointer and the one from the other translation unit
// run the same rewritten body.
// OUT: [[Y:[-0-9.e]+]] [[Y]]{{ }}

// SOURCE: (float)

// LEVEL-COUNT-5: Applying solution for CS:

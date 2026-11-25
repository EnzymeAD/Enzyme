// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -std=c++20 -O2 -DA=1e-6 -DB=1e-9 -poseidon-profile-generate -poseidon-cache=%t/cg %s -o %t/gen
// RUN: env POSEIDON_PROFILE_DIR=%t/prof %t/gen
// RUN: ls %t/prof | FileCheck --check-prefix=PROFILES %s
// RUN: %poseidon_clangxx -std=c++20 -O2 -DA=1e-9 -DB=1e-6 -poseidon-profile-use=%t/prof -poseidon-tau=1e-3 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c %s -o %t/use 2>&1 | FileCheck --check-prefix=USE --implicit-check-not='did not contain' %s

// A target that __poseidon_optimize_tau or a marker's poseidon_tau gives,
// taken by the caller as a template argument, is left out of the name of the
// call's copy: the
// profile of a call is named after its position, and a compile with other
// targets finds it.

#include "poseidon/poseidon.h"
#include <cstdio>

int poseidon_tau;
template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);

void pressure(const double *d, double *p) {
  double J = (1 + d[0]) * (1 + d[3]) - d[1] * d[2];
  *p = -J * (J - 1);
}

template <auto t> void viaCall(const double *d, double *p) {
  __poseidon_optimize_tau(t, pressure, d, p);
}

template <auto t> void viaMarker(const double *d, double *p) {
  __poseidon_fp_optimize<void>((void *)pressure, poseidon_tau, t, d, p);
}

int main() {
  double d[4] = {1e-3, 2e-4, -3e-4, 5e-4}, p[4];
  viaCall<A>(d, p);
  viaCall<B>(d, p + 1);
  viaMarker<A>(d, p + 2);
  viaMarker<B>(d, p + 3);
  printf("%g %g %g %g\n", p[0], p[1], p[2], p[3]);
}

// PROFILES: preprocess__Z8pressurePKdPd.poseidon._Z7viaCallITnDaL_Z3tauEEvPKdPd.0.fpprofile
// PROFILES-NEXT: preprocess__Z8pressurePKdPd.poseidon._Z7viaCallITnDaL_Z3tauEEvPKdPd.1.fpprofile
// PROFILES-NEXT: preprocess__Z8pressurePKdPd.poseidon._Z9viaMarkerITnDaL_Z3tauEEvPKdPd.0.fpprofile
// PROFILES-NEXT: preprocess__Z8pressurePKdPd.poseidon._Z9viaMarkerITnDaL_Z3tauEEvPKdPd.1.fpprofile

// USE-DAG: preprocess__Z8pressurePKdPd.poseidon._Z7viaCallITnDaL_Z3tauEEvPKdPd.0: the site's own accuracy target 1.000000e-09 overrides
// USE-DAG: preprocess__Z8pressurePKdPd.poseidon._Z7viaCallITnDaL_Z3tauEEvPKdPd.1: the site's own accuracy target 1.000000e-06 overrides
// USE-DAG: preprocess__Z8pressurePKdPd.poseidon._Z9viaMarkerITnDaL_Z3tauEEvPKdPd.0: the site's own accuracy target 1.000000e-09 overrides
// USE-DAG: preprocess__Z8pressurePKdPd.poseidon._Z9viaMarkerITnDaL_Z3tauEEvPKdPd.1: the site's own accuracy target 1.000000e-06 overrides

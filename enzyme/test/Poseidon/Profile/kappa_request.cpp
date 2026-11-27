// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen
// RUN: env POSEIDON_KAPPA_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/plain %t/gen %t/plain.runs > /dev/null 2> %t/plain.err
// RUN: FileCheck --check-prefix=ONCE %s < %t/plain.runs
// RUN: FileCheck --check-prefix=NOTICE %s < %t/plain.err
// RUN: FileCheck --check-prefix=UNMEASURED %s < %t/plain/_Z5gaussPKdPdi.fpprofile
// RUN: FileCheck --check-prefix=UNMEASURED %s < %t/plain/_Z4wavePKdPdi.fpprofile
// RUN: env POSEIDON_KAPPA_LADDER=1e-6,1e-5 POSEIDON_PROFILE_DIR=%t/env %poseidon_kappa -- %t/gen %t/env.runs > /dev/null 2> %t/env.err
// RUN: FileCheck --check-prefix=SIX %s < %t/env.runs
// RUN: FileCheck --check-prefix=MEASURE -DRUNS=5 %s < %t/env.err
// RUN: FileCheck --check-prefix=MEASURED %s < %t/env/_Z5gaussPKdPdi.fpprofile
// RUN: FileCheck --check-prefix=MEASURED %s < %t/env/_Z4wavePKdPdi.fpprofile
// RUN: ls -A %t/env | FileCheck --check-prefix=FILES --implicit-check-not=tmp %s
// RUN: not env POSEIDON_KAPPA_LADDER=1e-6,x POSEIDON_PROFILE_DIR=%t/badladder %poseidon_kappa -- %t/gen %t/badladder.runs 2>&1 | FileCheck --check-prefix=LADDER %s
// RUN: not ls %t/badladder.runs
// RUN: mkdir %t/scratch
// RUN: env TMPDIR=%t/scratch POSEIDON_KAPPA_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/tool %poseidon_kappa -- %t/gen %t/tool.runs > /dev/null 2> %t/tool.err
// RUN: ls -A %t/scratch | count 0
// RUN: FileCheck --check-prefix=FOUR %s < %t/tool.runs
// RUN: FileCheck --check-prefix=MEASURE -DRUNS=3 %s < %t/tool.err
// RUN: FileCheck --check-prefix=MEASURED %s < %t/tool/_Z4wavePKdPdi.fpprofile
// RUN: not %poseidon_kappa 2>&1 | FileCheck --check-prefix=USAGE %s
// RUN: printf '#!/bin/sh\n[ "$POSEIDON_KAPPA" = child ] && echo "no second run" && exit 3\nexec %t/gen "$@"\n' > %t/once.sh && chmod +x %t/once.sh
// RUN: not env TMPDIR=%t/scratch POSEIDON_KAPPA_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/failed %poseidon_kappa -- %t/once.sh %t/failed.runs > /dev/null 2> %t/failed.err
// RUN: FileCheck --check-prefix=FAILED %s < %t/failed.err
// RUN: ls -A %t/scratch | count 0
// RUN: env POSEIDON_KAPPA=child POSEIDON_PROFILE_DIR=%t/child %t/gen %t/child.runs > /dev/null
// RUN: ls %t/child | FileCheck --check-prefix=CHILD %s
// RUN: not %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/plain -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-algebraic=0 -poseidon-cache=%t/c -c %s -o %t/use.o 2>&1 | FileCheck --check-prefix=USE %s
// RUN: env POSEIDON_KAPPA=0 POSEIDON_PROFILE_DIR=%t/declined %t/gen %t/declined.runs > /dev/null 2> %t/declined.err
// RUN: FileCheck --check-prefix=ONCE %s < %t/declined.runs
// RUN: FileCheck --check-prefix=DECLINED %s < %t/declined.err
// RUN: FileCheck --check-prefix=NOKAPPA -DID=0 %s < %t/declined/_Z5gaussPKdPdi.fpprofile
// RUN: FileCheck --check-prefix=NOKAPPA -DID=1 %s < %t/declined/_Z4wavePKdPdi.fpprofile
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/declined -poseidon-early-prune -poseidon-tau=1e-6 -poseidon-cost-model=%host_cost_model -poseidon-enable-algebraic=0 -poseidon-cache=%t/c -c %s -poseidon-print -o %t/declined.o 2>&1 \
// RUN:   | FileCheck --check-prefix=DECLINED-USE --implicit-check-not=error: %s

// Condition numbers of a metric: poseidon-kappa measures them with more runs
// and says so first; the program run alone runs once, says what measuring
// takes and marks the profiles, which profile use then refuses; POSEIDON_KAPPA=0
// declines and each site is solved against its own target. A measurement run
// writes the metric only; a bad ladder or failing runs are errors and leave no
// scratch directory. The metric's name holds a space and outgrows inline storage.

#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>
#include <vector>

POSEIDON_OPTIMIZE void gauss(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::exp(-0.5 * x[i] * x[i]) * std::cos(4.0 * x[i]);
}

POSEIDON_OPTIMIZE void wave(const double *x, double *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = std::sin(x[i]) * x[i] + 1.0;
}

int main(int argc, char **argv) {
  if (FILE *f = fopen(argv[1], "a")) {
    fputs("run\n", f);
    fclose(f);
  }
  const int n = 256;
  std::vector<double> x(n), y(n), z(n);
  for (int i = 0; i < n; ++i)
    x[i] = -2.0 + 4.0 * i / (n - 1);
  gauss(x.data(), y.data(), n);
  wave(x.data(), z.data(), n);
  poseidon_metric("sum of y and z at a third", y[n / 3] + z[n / 3]);
  printf("%.17g\n", y[n / 3] + z[n / 3]);
}

// ONCE: run
// ONCE-NOT: run
// FOUR-COUNT-4: run
// FOUR-NOT: run
// SIX-COUNT-6: run
// SIX-NOT: run
// NOTICE: [poseidon] metric "sum of y and z at a third" was declared but its condition numbers at 2 sites were not measured, and profile use needs them (POSEIDON_KAPPA=0 on this run solves each site against its own target instead); measuring takes 3 more runs of this program: poseidon-kappa -- {{.*}}/gen {{.*}}/plain.runs
// NOTICE-NOT: [poseidon]
// DECLINED: [poseidon] metric "sum of y and z at a third" was declared and its condition numbers at 2 sites are not measured (POSEIDON_KAPPA=0): each site is solved against its own target
// DECLINED-NOT: [poseidon]
// NOKAPPA: SiteId = [[ID]]
// NOKAPPA-NEXT: CanonicalHash = {{[0-9a-f]+}}
// NOKAPPA-NOT: Kappa

// Each site is solved at the compile's target, with no condition number.
// DECLINED-USE-NOT: measured condition number
// DECLINED-USE: [poseidon] preprocess__Z5gaussPKdPdi_poseidon_body: tau=1.000000e-06 S=
// DECLINED-USE-NOT: measured condition number
// DECLINED-USE: [poseidon] preprocess__Z4wavePKdPdi_poseidon_body: tau=1.000000e-06 S=
// DECLINED-USE-NOT: measured condition number
// UNMEASURED: CanonicalHash =
// UNMEASURED-NEXT: KappaMark = unmeasured
// UNMEASURED-NOT: Kappa
// MEASURE-NOT: [poseidon]
// MEASURE: [poseidon] measuring the condition numbers of metric "sum of y and z at a third" at 2 sites: [[RUNS]] more runs of this program (run it without poseidon-kappa to skip)
// MEASURE-NEXT: [poseidon] baseline run 2 -> noise floor
// MEASURE-NOT: not measured
// MEASURE: [poseidon] profile run: {{[0-9.]+}} s
// MEASURE-NEXT: [poseidon] condition-number measurement: [[RUNS]] runs, {{[0-9.]+}} s
// CHILD-NOT: fpprofile
// CHILD: metric.txt
// CHILD-NOT: fpprofile
// MEASURED: Kappa = {{[1-9]|0\.0*[1-9]}}
// MEASURED-NEXT: KappaMark = {{ok|nonlinear}}

// Every profile is written whole through a file renamed over it, and the
// measurement rewrites them the same way.
// FILES-DAG: metric.txt
// FILES-DAG: _Z4wavePKdPdi.fpprofile
// FILES-DAG: _Z5gaussPKdPdi.fpprofile
// LADDER: [poseidon] POSEIDON_KAPPA_LADDER='1e-6,x' is not a comma-separated list of positive numbers.
// USAGE: usage: poseidon-kappa -- <program> [args...]
// FAILED: [poseidon] baseline run 2 -> noise floor
// FAILED-NEXT: poseidon-kappa: error: the second unperturbed run (exit 3) wrote no metric; its output follows
// FAILED-NEXT: no second run
// USE: kappa_request.cpp:[[#@LINE-70]]:1: error: Poseidon: the profiling run declared a metric with poseidon_metric and did not measure the condition number of preprocess__Z5gaussPKdPdi_poseidon_body against it ({{.*}}/plain/_Z5gaussPKdPdi.fpprofile); run the profiling build again under 'poseidon-kappa -- <program> [args...]', or with POSEIDON_KAPPA=0 to solve each site against its own target
// USE-NEXT: POSEIDON_OPTIMIZE void gauss(const double *x, double *y, int n) {
// USAGE: That takes 1 + <sites> x
// USAGE-NEXT: <steps> more runs, at the perturbation steps
// POSEIDON_KAPPA_LADDER lists

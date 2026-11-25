// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen

// RUN: env POSEIDON_PROBE_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/plain %t/gen %t/plain.runs > /dev/null 2> %t/plain.err
// RUN: FileCheck --check-prefix=ONCE %s < %t/plain.runs
// RUN: FileCheck --check-prefix=NOTICE %s < %t/plain.err
// RUN: FileCheck --check-prefix=UNMEASURED %s < %t/plain/preprocess__Z5gaussPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=UNMEASURED %s < %t/plain/preprocess__Z4wavePKdPdi_poseidon_body.fpprofile

// RUN: env POSEIDON_PROBE_LADDER=1e-6,1e-5 POSEIDON_PROFILE_DIR=%t/env %poseidon_probe -- %t/gen %t/env.runs > /dev/null 2> %t/env.err
// RUN: FileCheck --check-prefix=SIX %s < %t/env.runs
// RUN: FileCheck --check-prefix=MEASURE -DRUNS=5 %s < %t/env.err
// RUN: FileCheck --check-prefix=MEASURED %s < %t/env/preprocess__Z5gaussPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=MEASURED %s < %t/env/preprocess__Z4wavePKdPdi_poseidon_body.fpprofile
// RUN: ls -A %t/env | FileCheck --check-prefix=FILES %s
// RUN: not env POSEIDON_PROBE_LADDER=1e-6,x POSEIDON_PROFILE_DIR=%t/badladder %poseidon_probe -- %t/gen %t/badladder.runs 2>&1 | FileCheck --check-prefix=LADDER %s
// RUN: not ls %t/badladder.runs

// RUN: env POSEIDON_PROBE_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/tool %poseidon_probe -- %t/gen %t/tool.runs > /dev/null 2> %t/tool.err
// RUN: FileCheck --check-prefix=FOUR %s < %t/tool.runs
// RUN: FileCheck --check-prefix=MEASURE -DRUNS=3 %s < %t/tool.err
// RUN: FileCheck --check-prefix=MEASURED %s < %t/tool/preprocess__Z4wavePKdPdi_poseidon_body.fpprofile
// RUN: not %poseidon_probe 2>&1 | FileCheck --check-prefix=USAGE %s

// RUN: env POSEIDON_PROBE=child POSEIDON_PROFILE_DIR=%t/child %t/gen %t/child.runs > /dev/null
// RUN: ls %t/child | FileCheck --check-prefix=CHILD %s

// RUN: not %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/plain -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -c %s -o %t/use.o 2>&1 | FileCheck --check-prefix=USE %s

// RUN: env POSEIDON_PROBE=0 POSEIDON_PROFILE_DIR=%t/declined %t/gen %t/declined.runs > /dev/null 2> %t/declined.err
// RUN: FileCheck --check-prefix=ONCE %s < %t/declined.runs
// RUN: FileCheck --check-prefix=DECLINED %s < %t/declined.err
// RUN: FileCheck --check-prefix=NOKAPPA -DID=0 %s < %t/declined/preprocess__Z5gaussPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=NOKAPPA -DID=1 %s < %t/declined/preprocess__Z4wavePKdPdi_poseidon_body.fpprofile
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/declined -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -c %s -poseidon-print -o %t/declined.o 2>&1 \
// RUN:   | FileCheck --check-prefix=DECLINED-USE --implicit-check-not=error: %s

// The condition numbers of a program that declares a metric are measured by
// poseidon-probe, which runs it again and says so before the first extra run.
// Run alone, the program runs once, says what measuring would take, and marks
// the profiles so that profile use stops on them; POSEIDON_PROBE=0 declines,
// and profile use then solves each site against its own target. The program
// counts its own runs. A run of the measurement writes the metric and no
// profile. A ladder that does not parse stops poseidon-probe before the
// program runs.
// Both sites move the metric well above the noise of its runs, so each gets a
// nonzero condition number. The metric's name holds a space, which the
// measurement runs have to read back, and is longer than a string keeps
// inline, so that the exit handler reads it from memory it still owns.

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

// NOTICE: [probe] metric "sum of y and z at a third" was declared but its condition numbers at 2 sites were not measured, and profile use needs them (POSEIDON_PROBE=0 on this run solves each site against its own target instead); measuring takes 3 more runs of this program: poseidon-probe -- {{.*}}/gen {{.*}}/plain.runs
// NOTICE-NOT: [probe]

// DECLINED: [probe] metric "sum of y and z at a third" was declared and its condition numbers at 2 sites are not measured (POSEIDON_PROBE=0): each site is solved against its own target
// DECLINED-NOT: [probe]

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

// MEASURE-NOT: [probe]
// MEASURE: [probe] measuring the condition numbers of metric "sum of y and z at a third" at 2 sites: [[RUNS]] more runs of this program (run it without poseidon-probe to skip)
// MEASURE-NEXT: [probe] baseline run 2 -> noise floor
// MEASURE-NOT: not measured
// MEASURE: [probe] profile run: {{[0-9.]+}} s
// MEASURE-NEXT: [probe] condition-number measurement: [[RUNS]] runs, {{[0-9.]+}} s

// CHILD-NOT: fpprofile
// CHILD: metric.txt
// CHILD-NOT: fpprofile

// MEASURED: Kappa = {{[1-9]|0\.0*[1-9]}}
// MEASURED-NEXT: KappaMark = {{ok|nonlinear}}

// Every profile is written whole through a file renamed over it, and the
// measurement rewrites them the same way.
// FILES-NOT: tmp
// FILES: metric.txt
// FILES-NEXT: preprocess__Z4wavePKdPdi_poseidon_body.fpprofile
// FILES-NEXT: preprocess__Z5gaussPKdPdi_poseidon_body.fpprofile
// FILES-NOT: tmp

// LADDER: Poseidon FPProfiler: POSEIDON_PROBE_LADDER='1e-6,x' is not a comma-separated list of positive numbers.

// USAGE: usage: poseidon-probe -- <program> [args...]

// USE: probe_request.cpp:[[#@LINE-81]]:1: error: Poseidon: the profiling run declared a quantity of interest with poseidon_metric and did not measure the condition number of preprocess__Z5gaussPKdPdi_poseidon_body against it ({{.*}}/plain/preprocess__Z5gaussPKdPdi_poseidon_body.fpprofile). Run the profiling build again as 'poseidon-probe -- <program> [args...]', the command the profiling run printed, to measure it, or with POSEIDON_PROBE=0 in its environment to solve each site against its own target.
// USE-NEXT: POSEIDON_OPTIMIZE void gauss(const double *x, double *y, int n) {

// USAGE: That takes 1 + <sites> x
// USAGE-NEXT: <steps> more runs, at the perturbation steps
// POSEIDON_PROBE_LADDER lists

// REQUIRES: enzyme
// RUN: rm -rf %t && mkdir -p %t
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/c %s -o %t/gen 2>/dev/null

// RUN: env POSEIDON_PROBE_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/plain %t/gen %t/plain.runs > /dev/null 2> %t/plain.err
// RUN: FileCheck --check-prefix=ONCE %s < %t/plain.runs
// RUN: FileCheck --check-prefix=NOTICE %s < %t/plain.err
// RUN: FileCheck --check-prefix=UNMEASURED %s < %t/plain/preprocess__Z5gaussPKdPdi_poseidon_body.fpprofile
// RUN: FileCheck --check-prefix=UNMEASURED %s < %t/plain/preprocess__Z4wavePKdPdi_poseidon_body.fpprofile

// RUN: env POSEIDON_PROBE=1 POSEIDON_PROBE_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/env %t/gen %t/env.runs > /dev/null 2> %t/env.err
// RUN: FileCheck --check-prefix=FOUR %s < %t/env.runs
// RUN: FileCheck --check-prefix=MEASURE %s < %t/env.err
// RUN: FileCheck --check-prefix=MEASURED %s < %t/env/preprocess__Z5gaussPKdPdi_poseidon_body.fpprofile

// RUN: env POSEIDON_PROBE_LADDER=1e-6 POSEIDON_PROFILE_DIR=%t/tool %poseidon_probe -- %t/gen %t/tool.runs > /dev/null 2> %t/tool.err
// RUN: FileCheck --check-prefix=FOUR %s < %t/tool.runs
// RUN: FileCheck --check-prefix=MEASURE %s < %t/tool.err
// RUN: FileCheck --check-prefix=MEASURED %s < %t/tool/preprocess__Z4wavePKdPdi_poseidon_body.fpprofile
// RUN: not %poseidon_probe 2>&1 | FileCheck --check-prefix=USAGE %s

// RUN: not %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/plain -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -c %s -o %t/use.o 2> %t/use.err
// RUN: FileCheck --check-prefix=USE %s < %t/use.err

// RUN: env POSEIDON_PROBE=0 POSEIDON_PROFILE_DIR=%t/declined %t/gen %t/declined.runs > /dev/null 2> %t/declined.err
// RUN: FileCheck --check-prefix=ONCE %s < %t/declined.runs
// RUN: FileCheck --check-prefix=DECLINED %s < %t/declined.err
// RUN: FileCheck --check-prefix=NOKAPPA %s < %t/declined/preprocess__Z5gaussPKdPdi_poseidon_body.fpprofile
// RUN: %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/declined -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-enable-herbie=0 -poseidon-cache=%t/c -c %s -o %t/declined.o 2> %t/declined.use.err

// A profiling run whose program declares a metric re-executes itself to
// measure the condition numbers only when asked, by POSEIDON_PROBE=1 or by
// running it under poseidon-probe, and says so before the first extra run.
// Not asked, it runs once, says in one line what measuring would take, and
// marks the profiles so that profile use stops on them rather than solving
// them as if no metric had been declared. Declining with POSEIDON_PROBE=0 is
// the way to have exactly that. The program counts its own runs.
// The host target has no perturbation hook, so what a measurement finds here
// is not checked, only that it ran. The metric's name holds a space, which
// the measurement runs have to read back.

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
  poseidon_metric("y sum", y[n / 3] + z[n / 3]);
  printf("%.17g\n", y[n / 3] + z[n / 3]);
}

// ONCE: run
// ONCE-NOT: run

// FOUR-COUNT-4: run
// FOUR-NOT: run

// NOTICE: [probe] metric "y sum" was declared but its condition numbers at 2 sites were not measured, and profile use needs them (POSEIDON_PROBE=0 on this run solves each site against its own target instead); measuring takes 3 more runs of this program: poseidon-probe -- {{.*}}/gen {{.*}}/plain.runs
// NOTICE-NOT: [probe]

// DECLINED: [probe] metric "y sum" was declared and its condition numbers at 2 sites are not measured (POSEIDON_PROBE=0): each site is solved against its own target
// DECLINED-NOT: [probe]

// NOKAPPA-NOT: Kappa

// UNMEASURED: CanonicalHash =
// UNMEASURED-NEXT: KappaMark = unmeasured
// UNMEASURED-NOT: Kappa

// MEASURE-NOT: [probe]
// MEASURE: [probe] measuring the condition numbers of metric "y sum" at 2 sites: 3 more runs of this program (run it without POSEIDON_PROBE=1 to skip)
// MEASURE-NEXT: [probe] baseline run 2 -> noise floor
// MEASURE-NOT: not measured

// MEASURED: Kappa =
// MEASURED-NOT: unmeasured

// USAGE: usage: poseidon-probe -- <program> [args...]

// USE: probe_request.cpp:{{[0-9]+}}:1: error: {{.*}}Poseidon: the profiling run declared a quantity of interest with poseidon_metric and did not measure the condition number of {{.*}} against it ({{.*}}/plain/preprocess__Z{{.*}}.fpprofile). Run the profiling build again as 'poseidon-probe -- <program> [args...]', the command the profiling run printed, to measure it, or with POSEIDON_PROBE=0 in its environment to solve each site against its own target.
// USE-NEXT: POSEIDON_OPTIMIZE void {{gauss|wave}}(

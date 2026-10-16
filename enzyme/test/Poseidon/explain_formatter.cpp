// Poseidon profiles the call wrapped in __poseidon_fp_optimize, then rewrites
// it to the cheapest form whose modelled error meets -poseidon-tau. In the
// assembly, kernel is the function as written and the rewritten call is inlined
// into main; the solver's decisions are in the compiler output.
namespace std { double cos(double); double pow(double, double); }
extern int poseidon_unused;


template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;

// (1 - cos x) / x^2 loses its digits to cancellation as x -> 0.
__attribute__((noinline)) void kernel(const double *x, double *out, int n) {
  for (int i = 0; i < n; ++i)
    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
}

int main() {
  const int n = 1024;
  static double x[n], out[n], dx[n], dout[n], ref[n];
  for (int i = 0; i < n; ++i) {
    x[i] = std::pow(10.0, -8.0 + 4.0 * i / (n - 1));
    dout[i] = 1.0;
  }
  kernel(x, ref, n);
  __poseidon_fp_optimize<void>((void *)kernel, enzyme_dup, x, dx, enzyme_dup, out, dout, n);
}

// RUN: %poseidon_explain %S/Inputs/explain/formatter/cancellation.explain.json --format report | FileCheck --check-prefix=REPORT %s
// RUN: %poseidon_explain %S/Inputs/explain/formatter/cancellation.explain.json --format remarks --file-name '<source>' | FileCheck --check-prefix=REMARKS %s
// RUN: %poseidon_explain %S/Inputs/explain/formatter/cancellation.explain.json --format fpcore | FileCheck --check-prefix=FPCORE %s
// RUN: %poseidon_explain %S/Inputs/explain/formatter/cancellation.explain.json --format json | FileCheck --check-prefix=JSON %s
// RUN: %poseidon_explain %S/Inputs/explain/formatter/cancellation.explain.json --format source --source %s --clang %clang | FileCheck --check-prefix=SOURCE %s
// RUN: %poseidon_explain %S/Inputs/explain/formatter/cancellation.explain.json --format diff --source %s --clang %clang | FileCheck --check-prefix=DIFF %s
// RUN: %poseidon_explain %S/Inputs/explain/formatter/cancellation.explain.json --format source --source %s --clang /nonexistent/clang | FileCheck --check-prefix=NOAST %s
// REQUIRES: poseidon

// REPORT: Poseidon site kernel(double const*, double*, int)  [_Z6kernelPKdPdi]
// REPORT-NEXT: called at explain_formatter.cpp:26:3
// REPORT-NEXT: accuracy target tau = 1e-06; cost model cm_x86-64_explorer.csv; selector dp
// REPORT-NEXT: as written:   modelled error 2.8e-10, cost 3576832
// REPORT-NEXT: with Poseidon: modelled error 1.2e-16, cost 0 (-100%); applied R0_1
// REPORT: explain_formatter.cpp:15:37  S0: (1.0 - cos(x[i])) / (x[i] * x[i])
// REPORT-NEXT: profile: x[i] in [1e-08, 0.0001]; 1024 executions; 4 operations; cost 3576832
// REPORT-NEXT: applied R0_1 (algebraic, binary64): 0.5
// REPORT-NEXT: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]), valid on the profiled range x[i] in [1e-08, 0.0001]; modelled error 1.2e-16; cost change -3576832
// REPORT: R0_0 pareto algebraic 2.1e-25 -3453952 fma(-0.041666666666666664, x[i] * x[i], 0.5)
// REPORT-NEXT: on the Pareto frontier; -poseidon-apply-rewrites=R0_0 applies it
// REPORT: R1_0 rejected algebraic 1.2e-16 -3576832 (double)0.5f
// REPORT-NEXT: the selected R0_1 rewrites the same output
// REPORT: Pareto frontier (cost, modelled error; * = selected):
// REPORT: * 0 1.2e-16 R0_1 -poseidon-tau=0 -poseidon-comp-cost-budget=-3576832
// REPORT-NEXT: 122880 2.1e-25 R0_0 -poseidon-tau=0 -poseidon-comp-cost-budget=-3453952

// REMARKS: <source>:26:3: note: Poseidon site kernel: accuracy target tau = 1e-06; modelled error as written 2.8e-10, after 1.2e-16; applied R0_1; cost 3576832 -> 0
// REMARKS-NEXT: <source>:15:37: note: Poseidon rewrote (1.0 - cos(x[i])) / (x[i] * x[i]) as 0.5 (series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]), valid on the profiled range x[i] in [1e-08, 0.0001]); modelled error 1.2e-16 [R0_1]
// REMARKS-NEXT: <source>:15:37: note: Poseidon did not select R0_0 fma(-0.041666666666666664, x[i] * x[i], 0.5): on the Pareto frontier
// REMARKS-NEXT: <source>:15:37: note: Poseidon rejected R1_0 (double)0.5f: the selected R0_1 rewrites the same output

// FPCORE: (FPCore (v0) :name "S0 output 0 as written" :precision binary64 :pre (<= 1e-08 v0 0.0001)
// FPCORE-NEXT: (/ (- 1 (cos v0)) (* v0 v0)))
// FPCORE-NEXT: (FPCore (v0) :name "S0 R0_1 (applied)" :precision binary64 :pre (<= 1e-08 v0 0.0001)
// FPCORE-NEXT: 1/2)
// FPCORE-NEXT: (FPCore (v0) :name "S0 R0_0 (pareto)" :precision binary64 :pre (<= 1e-08 v0 0.0001)
// FPCORE-NEXT: (fma -1/24 (* v0 v0) 1/2))

// JSON: "schema": "poseidon-explain/1"

// SOURCE: {{^}}    out[i] = 0.5;  // Poseidon: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]) [R0_1], modelled error 1.2e-16{{$}}

// DIFF: --- a/explain_formatter.cpp
// DIFF-NEXT: +++ b/explain_formatter.cpp
// DIFF: {{^}}-    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
// DIFF-NEXT: {{^}}+    out[i] = 0.5;  // Poseidon

// NOAST: {{^}}    // Poseidon: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]): (1.0 - cos(x[i])) / (x[i] * x[i]) -> 0.5 [R0_1], modelled error 1.2e-16
// NOAST-NEXT: {{^}}    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);

// poseidon-explain's formats on recorded explanations, and its errors on
// overlapping rewrites, ASTs and explanations it cannot read.
// DEFINE: %{explain} = %poseidon_explain %t/formatter/cancellation.explain.json
// RUN: rm -rf %t && split-file %s %t
// RUN: %{explain} --format report | FileCheck --check-prefix=REPORT %s
// RUN: %{explain} --format fpcore | FileCheck --check-prefix=FPCORE %s
// RUN: %{explain} --format budgets | FileCheck --check-prefix=BUDGETS %s
// RUN: %{explain} --format source --source %t/formatter.cpp --clang %clang | FileCheck --check-prefix=SOURCE %s
// RUN: %{explain} --format diff --source %t/formatter.cpp --clang %clang | FileCheck --check-prefix=DIFF %s
// RUN: %{explain} --format source --source %t/formatter.cpp --clang /nonexistent/clang 2>&1 | FileCheck --check-prefix=NOAST %s
// RUN: not %poseidon_explain %t/formatter --function no_such_site 2>&1 | FileCheck --check-prefix=NOMATCH %s
// RUN: %poseidon_explain %t/formatter/rotation.explain.json --format source --source %t/formatter_rotation.cpp --clang %clang | FileCheck --check-prefix=ROT-SOURCE %s
// RUN: %poseidon_explain %t/formatter/rotation.explain.json --format report | FileCheck --check-prefix=ROT-REPORT %s
// RUN: %{explain} %t/formatter/overlap.explain.json --format source --source %t/formatter.cpp --clang %clang 2>&1 | FileCheck --check-prefix=OVERLAP %s
// RUN: %poseidon_explain %t/formatter/mixed.explain.json --format source --source %t/formatter_rotation.cpp --clang %clang | FileCheck --check-prefix=MIXED %s
// RUN: not %{explain} --format source --source %t/formatter.cpp --clang echo 2>&1 | FileCheck --check-prefix=BADAST --implicit-check-not=Traceback %s
// RUN: mkdir %t/utf8 && sed "15s|^    out\[i\] = |/*$(printf '\303\251')*/out[i]=|" %t/formatter.cpp > %t/utf8/formatter.cpp
// RUN: %{explain} --format source --source %t/utf8/formatter.cpp --clang %clang | FileCheck --check-prefix=UTF8 %s
// RUN: printf '{"schema": ' > %t/cut.explain.json
// RUN: not %poseidon_explain %t/cut.explain.json 2>&1 | FileCheck --check-prefix=BADJSON --implicit-check-not=Traceback %s
// RUN: not %poseidon_explain %t/missing.explain.json 2>&1 | FileCheck --check-prefix=MISSING --implicit-check-not=Traceback %s

// REPORT: Poseidon site kernel(double const*, double*, int)  [_Z6kernelPKdPdi]
// REPORT-NEXT: called at formatter.cpp:26:3
// REPORT-NEXT: accuracy target tau = 1e-06; cost model cm_x86-64_explorer.csv; selector dp
// REPORT-NEXT: as written:   modelled error 2.8e-10, cost 1
// REPORT-NEXT: with Poseidon: modelled error 1.2e-16, cost 0 (-100%); applied A0_1
// REPORT: formatter.cpp:15:37  S0: (1.0 - cos(x[i])) / (x[i] * x[i])
// REPORT-NEXT: profile: x[i] in [1e-08, 0.0001]; 1024 executions; 4 operations; cost 1
// REPORT-NEXT: applied A0_1 (algebraic, binary64): 0.5
// REPORT-NEXT: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]), valid on the profiled range x[i] in [1e-08, 0.0001]; modelled error 1.2e-16; cost change -1
// REPORT: A0_0 pareto algebraic 2.1e-25 -0.965646 fma(-0.041666666666666664, x[i] * x[i], 0.5)
// REPORT-NEXT: on the Pareto frontier; -poseidon-pin=A0_0 applies it
// REPORT: A1_0 rejected algebraic 1.2e-16 -1 (double)0.5f
// REPORT-NEXT: the selected A0_1 rewrites the same output
// REPORT: Pareto frontier (cost, modelled error; * = selected):
// REPORT: * 0 1.2e-16 A0_1 -poseidon-tau=0 -poseidon-cost-budget=-1
// REPORT-NEXT: 0.0343544 2.1e-25 A0_0 -poseidon-tau=0 -poseidon-cost-budget=-0.9656455768680218

// FPCORE: (FPCore (v0) :name "S0 output 0 as written" :precision binary64 :pre (<= 1e-08 v0 0.0001)
// FPCORE-NEXT: (/ (- 1 (cos v0)) (* v0 v0)))
// FPCORE-NEXT: (FPCore (v0) :name "S0 A0_1 (applied)" :precision binary64 :pre (<= 1e-08 v0 0.0001)
// FPCORE-NEXT: 1/2)
// FPCORE-NEXT: (FPCore (v0) :name "S0 A0_0 (pareto)" :precision binary64 :pre (<= 1e-08 v0 0.0001)
// FPCORE-NEXT: (fma -1/24 (* v0 v0) 1/2))

// BUDGETS: {{^}}-1,-0.96564557686802177{{$}}

// SOURCE: {{^}}    out[i] = 0.5;  // Poseidon: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]) [A0_1], modelled error 1.2e-16{{$}}

// DIFF: --- a/formatter.cpp
// DIFF-NEXT: +++ b/formatter.cpp
// DIFF: {{^}}-    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);
// DIFF-NEXT: {{^}}+    out[i] = 0.5;  // Poseidon

// NOAST: poseidon-explain: no AST of kernel ({{.+}}); its rewrites are shown as comments
// NOAST: {{^}}    // Poseidon: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]): (1.0 - cos(x[i])) / (x[i] * x[i]) -> 0.5 [A0_1], modelled error 1.2e-16
// NOAST-NEXT: {{^}}    out[i] = (1.0 - std::cos(x[i])) / (x[i] * x[i]);

// NOMATCH: poseidon-explain: no explanation in {{.*}}formatter is of a site named like 'no_such_site'

// The two library calls of one FP subgraph move to float, each where it is
// written, with its operands cast.
// ROT-SOURCE: {{^}}    double angle = std::sqrt(x * x + y * y + z * z);{{$}}
// ROT-SOURCE-NEXT: {{^}}    double s = (double)sinf((float)(0.5 * angle)) / angle;  // Poseidon: sin in float [P0_10]{{$}}
// ROT-SOURCE-NEXT: {{^}}    q[4 * i] = (double)cosf((float)(0.5 * angle));  // Poseidon: cos in float [P0_10]{{$}}
// ROT-SOURCE-NEXT: {{^}}    q[4 * i + 1] = s * x;{{$}}

// ROT-REPORT: formatter_rotation.cpp:15:22  S0: x * (sin(sqrt(z * z + (x * x + y * y)) * 0.5) / sqrt(z * z + (x * x + y * y)))
// ROT-REPORT-NEXT: output 1 (formatter_rotation.cpp:16:22): y * (sin(sqrt(z * z + (x * x + y * y)) * 0.5) / sqrt(z * z + (x * x + y * y)))
// ROT-REPORT: profile: y in [-0.5, -0.05], x in [0.001, 1], z in [0.02, 0.06]; 1000 executions; 11 operations; cost 1
// ROT-REPORT-NEXT: applied P0_10 (precision): Funcs FP64(0%) + FP32(100%): sin (13:16) and cos (14:16) in float (2 of 11 operations)

// A second explanation of kernel applies A1_1 to the expression A0_1 rewrote.
// OVERLAP: poseidon-explain: A1_1 overlaps an earlier rewrite; shown as a comment
// OVERLAP: {{^}}    // Poseidon: algebraic rewrite at binary64: not applied, it overlaps an earlier rewrite [A1_1], modelled error 1.2e-16{{$}}
// OVERLAP-NEXT: {{^}}    out[i] = 0.5;  // Poseidon: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]) [A0_1], modelled error 1.2e-16{{$}}

// Changes to different formats render each operation in its own type.
// MIXED: {{^}}    double s = (double)sinf((float)(0.5 * angle)) / angle;  // Poseidon: sin in float [P0_10]{{$}}
// MIXED-NEXT: {{^}}    q[4 * i] = (double)cosf((_Float16)(0.5 * angle));  // Poseidon: cos in half [P0_10]{{$}}

// BADAST: poseidon-explain: clang's AST of kernel is not JSON: {{.+}}

// Line 15 with a two-byte character in a comment, keeping the byte columns.
// UTF8: {{^}}/*{{.+}}*/out[i]=0.5;  // Poseidon: series approximation

// BADJSON: poseidon-explain: {{.*}}cut.explain.json is not JSON: {{.+}}

// MISSING: poseidon-explain: cannot read {{.*}}missing.explain.json: No such file or directory

//--- formatter.cpp
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
//--- formatter_rotation.cpp
// One step of attitude integration: four quaternion components solved as one FP subgraph.
namespace std { double sqrt(double); double sin(double); double cos(double); }
extern int poseidon_unused;


template <typename R, typename... T> R __poseidon_fp_optimize(void *, T...);
extern int enzyme_dup;

__attribute__((noinline)) void rotation(const double *w, double *q, int n) {
  for (int i = 0; i < n; ++i) {
    double x = w[3 * i], y = w[3 * i + 1], z = w[3 * i + 2];
    double angle = std::sqrt(x * x + y * y + z * z);
    double s = std::sin(0.5 * angle) / angle;
    q[4 * i] = std::cos(0.5 * angle);
    q[4 * i + 1] = s * x;
    q[4 * i + 2] = s * y;
    q[4 * i + 3] = s * z;
  }
}
//--- formatter/cancellation.explain.json
{"schema": "poseidon-explain/1",
 "function": {"name": "kernel(double const*, double*, int)", "symbol": "_Z6kernelPKdPdi", "site": {"file": "formatter.cpp", "line": 26, "col": 3}},
 "target": {"cost_model": "cm_x86-64_explorer.csv", "selector": "dp", "tau": 1e-06, "tau_from": "flag"},
 "result": {"applied": ["A0_1"], "cost_after": 0, "cost_before": 1.0, "modelled_rel_error": 1.1607706226323944e-16, "modelled_rel_error_as_written": 2.770306794452491e-10, "status": "rewritten"},
 "pareto": [{"applied": ["A0_1"], "budget": -1.0, "cost": 0, "modelled_rel_error": 1.1607706226323944e-16, "selected": true},
            {"applied": ["A0_0"], "budget": -0.9656455768680218, "cost": 0.03435442313197824, "modelled_rel_error": 2.1273352908378104e-25}],
 "subgraphs": [{"id": "S0", "kind": "subgraph", "loc": {"file": "formatter.cpp", "line": 15, "col": 37}, "cost": 1.0, "executions": 1024,
  "inputs": [{"loc": {"file": "formatter.cpp", "line": 15, "col": 30}, "min": 1e-08, "max": 0.0001, "name": "x[i]", "var": "v0"}],
  "ops": [{"loc": {"file": "formatter.cpp", "line": 15, "col": 21}, "opcode": "cos"}, {"loc": {"file": "formatter.cpp", "line": 15, "col": 19}, "opcode": "-"},
          {"loc": {"file": "formatter.cpp", "line": 15, "col": 45}, "opcode": "*"}, {"loc": {"file": "formatter.cpp", "line": 15, "col": 37}, "opcode": "/"}],
  "original": {"c": "(1.0 - cos(x[i])) / (x[i] * x[i])"},
  "outputs": [{"c": "(1.0 - cos(x[i])) / (x[i] * x[i])", "fpcore": "(/ (- 1 (cos v0)) (* v0 v0))", "loc": {"file": "formatter.cpp", "line": 15, "col": 37}, "ops": [0, 1, 2, 3]}],
  "alternatives": [
   {"id": "A0_0", "kind": "algebraic", "status": "pareto", "precision": "binary64", "outputs": [0], "c": "fma(-0.041666666666666664, x[i] * x[i], 0.5)",
    "fpcore": "#s(approx (/ (- 1 (cos v0)) (* v0 v0)) (fma.f64 #s(literal -1/24 binary64) (*.f64 v0 v0) #s(literal 1/2 binary64)))",
    "approx_note": "series approximation of (1.0 - cos(x[i])) / (x[i] * x[i])", "modelled_error": 2.1273352908378104e-25, "cost_delta": -0.9656455768680218, "reason": "on the Pareto frontier; -poseidon-pin=A0_0 applies it"},
   {"id": "A0_1", "kind": "algebraic", "status": "applied", "precision": "binary64", "outputs": [0], "c": "0.5", "fpcore": "#s(approx (/ (- 1 (cos v0)) (* v0 v0)) #s(literal 1/2 binary64))",
    "approx_note": "series approximation of (1.0 - cos(x[i])) / (x[i] * x[i])", "modelled_error": 1.1607706226323944e-16, "cost_delta": -1.0, "reason": "part of the cheapest solution that meets the target"},
   {"id": "A1_0", "kind": "algebraic", "status": "rejected", "precision": "binary32", "outputs": [0], "c": "(double)0.5f", "fpcore": "#s(approx (/ (- 1 (cos v0)) (* v0 v0)) #s(literal 1/2 binary32))",
    "approx_note": "series approximation of (1.0 - cos(x[i])) / (x[i] * x[i])", "modelled_error": 1.1607706226323944e-16, "cost_delta": -1.0, "reason": "the selected A0_1 rewrites the same output"}]}]}
//--- formatter/overlap.explain.json
{"schema": "poseidon-explain/1",
 "function": {"name": "kernel(double const*, double*, int)", "symbol": "_Z6kernelPKdPdi", "site": {"file": "formatter.cpp", "line": 26, "col": 3}},
 "subgraphs": [{"id": "S1", "kind": "subgraph", "loc": {"file": "formatter.cpp", "line": 15, "col": 37},
  "ops": [{"loc": {"file": "formatter.cpp", "line": 15, "col": 21}, "opcode": "cos"}, {"loc": {"file": "formatter.cpp", "line": 15, "col": 19}, "opcode": "-"},
          {"loc": {"file": "formatter.cpp", "line": 15, "col": 45}, "opcode": "*"}, {"loc": {"file": "formatter.cpp", "line": 15, "col": 37}, "opcode": "/"}],
  "outputs": [{"c": "(1.0 - cos(x[i])) / (x[i] * x[i])", "loc": {"file": "formatter.cpp", "line": 15, "col": 37}, "ops": [0, 1, 2, 3]}],
  "alternatives": [{"id": "A1_1", "kind": "algebraic", "status": "applied", "precision": "binary64", "outputs": [0], "c": "0.25", "modelled_error": 1.1607706226323944e-16}]}]}
//--- formatter/rotation.explain.json
{"schema": "poseidon-explain/1",
 "function": {"name": "rotation(double const*, double*, int)", "symbol": "_Z8rotationPKdPdi"},
 "target": {"cost_model": "cm_x86-64_explorer.csv", "selector": "dp", "tau": 1e-06, "tau_from": "flag"},
 "result": {"applied": ["P0_10"], "cost_after": 0.2558692006707658, "cost_before": 1.0, "modelled_rel_error": 1.3241097083711176e-08, "modelled_rel_error_as_written": 1.0087777130359326e-17, "status": "rewritten"},
 "subgraphs": [{"id": "S0", "kind": "subgraph", "loc": {"file": "formatter_rotation.cpp", "line": 15, "col": 22}, "cost": 1.0, "executions": 1000,
  "inputs": [{"min": -0.5, "max": -0.05, "name": "y", "var": "v0"}, {"min": 0.001, "max": 1, "name": "x", "var": "v1"}, {"min": 0.02, "max": 0.06, "name": "z", "var": "v2"}],
  "ops": [{"opcode": "*"}, {"opcode": "fma"}, {"opcode": "fma"}, {"opcode": "sqrt"}, {"opcode": "*"},
          {"loc": {"file": "formatter_rotation.cpp", "line": 13, "col": 16}, "opcode": "sin", "float_call": "sinf"}, {"opcode": "/"},
          {"loc": {"file": "formatter_rotation.cpp", "line": 14, "col": 16}, "opcode": "cos", "float_call": "cosf"}, {"opcode": "*"}, {"opcode": "*"}, {"opcode": "*"}],
  "original": {"c": ["x * (sin(sqrt(z * z + (x * x + y * y)) * 0.5) / sqrt(z * z + (x * x + y * y)))", "y * (sin(sqrt(z * z + (x * x + y * y)) * 0.5) / sqrt(z * z + (x * x + y * y)))"]},
  "outputs": [{"loc": {"file": "formatter_rotation.cpp", "line": 15, "col": 22}, "c": "x * (sin(sqrt(z * z + (x * x + y * y)) * 0.5) / sqrt(z * z + (x * x + y * y)))"},
              {"loc": {"file": "formatter_rotation.cpp", "line": 16, "col": 22}, "c": "y * (sin(sqrt(z * z + (x * x + y * y)) * 0.5) / sqrt(z * z + (x * x + y * y)))"}],
  "alternatives": [{"id": "P0_10", "kind": "precision", "status": "applied", "changes": [{"ops": [5, 7], "to": "FP32"}], "modelled_error": 1.3241097083711176e-08,
   "cost_delta": -0.7441307993292342, "summary": "Funcs FP64(0%) + FP32(100%): sin (13:16) and cos (14:16) in float (2 of 11 operations)"}]}]}
//--- formatter/mixed.explain.json
{"schema": "poseidon-explain/1",
 "function": {"name": "rotation(double const*, double*, int)", "symbol": "_Z8rotationPKdPdi"},
 "subgraphs": [{"id": "S0", "kind": "subgraph", "loc": {"file": "formatter_rotation.cpp", "line": 15, "col": 22},
  "ops": [{"opcode": "*"}, {"opcode": "fma"}, {"opcode": "fma"}, {"opcode": "sqrt"}, {"opcode": "*"},
          {"loc": {"file": "formatter_rotation.cpp", "line": 13, "col": 16}, "opcode": "sin", "float_call": "sinf"}, {"opcode": "/"},
          {"loc": {"file": "formatter_rotation.cpp", "line": 14, "col": 16}, "opcode": "cos", "float_call": "cosf"}],
  "alternatives": [{"id": "P0_10", "kind": "precision", "status": "applied", "changes": [{"ops": [5], "to": "FP32"}, {"ops": [7], "to": "FP16"}],
   "summary": "sin (13:16) in float and cos (14:16) in half"}]}]}

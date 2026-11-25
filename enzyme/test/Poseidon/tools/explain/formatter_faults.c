// poseidon-explain on explanations and ASTs its formats cannot take as they
// are: a rewrite that overlaps an earlier one is reported, a precision
// alternative whose changes go to different formats renders each operation in
// its own type, and an AST or an explanation that is not JSON, or that cannot
// be read, is an error. Columns count bytes, as clang counts them.
//
// RUN: rm -rf %t && split-file %s %t
// RUN: %poseidon_explain %S/../../Inputs/explain/formatter/overlap.explain.json --format source --source %S/formatter.cpp --clang %clang 2>&1 | FileCheck --check-prefix=OVERLAP %s
// RUN: %poseidon_explain %S/../../Inputs/explain/formatter/mixed.explain.json --format source --source %S/formatter_rotation.cpp --clang %clang | FileCheck --check-prefix=MIXED %s
// RUN: not %poseidon_explain %S/../../Inputs/explain/formatter/cancellation.explain.json --format source --source %S/formatter.cpp --clang %S/../../Inputs/explain/bad-ast-clang.sh 2>&1 | FileCheck --check-prefix=BADAST --implicit-check-not=Traceback %s
// RUN: %poseidon_explain %S/../../Inputs/explain/formatter/cancellation.explain.json --format source --source %t/formatter.cpp --clang %clang | FileCheck --check-prefix=UTF8 %s
// RUN: printf '{"schema": ' > %t/cut.explain.json
// RUN: not %poseidon_explain %t/cut.explain.json 2>&1 | FileCheck --check-prefix=BADJSON --implicit-check-not=Traceback %s
// RUN: not %poseidon_explain %t/missing.explain.json 2>&1 | FileCheck --check-prefix=MISSING --implicit-check-not=Traceback %s

// OVERLAP: poseidon-explain: R1_1 overlaps an earlier rewrite; shown as a comment
// OVERLAP: {{^}}    // Poseidon: algebraic rewrite at binary64: not applied, it overlaps an earlier rewrite [R1_1], modelled error 1.2e-16{{$}}
// OVERLAP-NEXT: {{^}}    out[i] = 0.5;  // Poseidon: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]) [R0_1], modelled error 1.2e-16{{$}}

// MIXED: {{^}}    double s = (double)sinf((float)(0.5 * angle)) / angle;  // Poseidon: sin in float [P0_10]{{$}}
// MIXED-NEXT: {{^}}    q[4 * i] = (double)cosf((_Float16)(0.5 * angle));  // Poseidon: cos in half [P0_10]{{$}}

// The source of cancellation.explain.json with a two-byte character in a
// comment before the rewritten expression on line 15, which keeps the byte
// columns of the original by dropping the indentation and the spaces around =.
// UTF8: {{^}}/*{{.+}}*/out[i]=0.5;  // Poseidon: series approximation

// BADAST: poseidon-explain: clang's AST of kernel is not JSON: {{.+}}

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
/*é*/out[i]=(1.0 - std::cos(x[i])) / (x[i] * x[i]);
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

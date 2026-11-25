// poseidon-explain on explanations and ASTs its formats cannot take as they
// are: a rewrite that overlaps an earlier one is reported, a precision
// alternative whose changes go to different formats renders each operation in
// its own type, and an AST or an explanation that is not JSON, or that cannot
// be read, is an error. Columns count bytes, as clang counts them.
//
// RUN: rm -rf %t && mkdir %t
// RUN: %poseidon_explain %S/../Inputs/explain/formatter/overlap.explain.json --format source --source %S/explain_formatter.cpp --clang %clang 2>&1 | FileCheck --check-prefix=OVERLAP %s
// RUN: %poseidon_explain %S/../Inputs/explain/formatter/mixed.explain.json --format source --source %S/explain_formatter_rotation.cpp --clang %clang | FileCheck --check-prefix=MIXED %s
// RUN: not %poseidon_explain %S/../Inputs/explain/formatter/cancellation.explain.json --format source --source %S/explain_formatter.cpp --clang %S/../Inputs/explain/bad-ast-clang.sh 2>&1 | FileCheck --check-prefix=BADAST --implicit-check-not=Traceback %s
// RUN: %python -c "import sys; s = open(sys.argv[1]).read().split('\n'); s[14] = '/*\u00e9*/' + s[14]; open(sys.argv[2], 'w', encoding='utf-8').write('\n'.join(s))" %S/explain_formatter.cpp %t/explain_formatter.cpp
// RUN: %poseidon_explain %S/../Inputs/explain/formatter/utf8.explain.json --format source --source %t/explain_formatter.cpp --clang %clang | FileCheck --check-prefix=UTF8 %s
// RUN: printf '{"schema": ' > %t/cut.explain.json
// RUN: not %poseidon_explain %t/cut.explain.json 2>&1 | FileCheck --check-prefix=BADJSON --implicit-check-not=Traceback %s
// RUN: not %poseidon_explain %t/missing.explain.json 2>&1 | FileCheck --check-prefix=MISSING --implicit-check-not=Traceback %s

// OVERLAP: poseidon-explain: R1_1 overlaps an earlier rewrite; shown as a comment
// OVERLAP: {{^}}    // Poseidon: algebraic rewrite at binary64: not applied, it overlaps an earlier rewrite [R1_1], modelled error 1.2e-16{{$}}
// OVERLAP-NEXT: {{^}}    out[i] = 0.5;  // Poseidon: series approximation of (1.0 - cos(x[i])) / (x[i] * x[i]) [R0_1], modelled error 1.2e-16{{$}}

// MIXED: {{^}}    double s = (double)sinf((float)(0.5 * angle)) / angle;  // Poseidon: sin in float [PT0_10]{{$}}
// MIXED-NEXT: {{^}}    q[4 * i] = (double)cosf((_Float16)(0.5 * angle));  // Poseidon: cos in half [PT0_10]{{$}}

// UTF8: {{^}}/*{{.+}}*/    out[i] = 0.5;  // Poseidon: series approximation

// BADAST: poseidon-explain: clang's AST of kernel is not JSON: {{.+}}

// BADJSON: poseidon-explain: {{.*}}cut.explain.json is not JSON: {{.+}}

// MISSING: poseidon-explain: cannot read {{.*}}missing.explain.json: No such file or directory

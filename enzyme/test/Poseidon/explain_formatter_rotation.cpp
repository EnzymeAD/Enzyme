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

int main() {
  const int n = 1000;
  static double w[3 * n], q[4 * n], dw[3 * n], dq[4 * n], ref[4 * n];
  for (int i = 0; i < n; ++i) {
    w[3 * i] = 1e-3 * (i + 1);
    w[3 * i + 1] = -0.05 * (i % 10 + 1);
    w[3 * i + 2] = 0.02 * (i % 3 + 1);
  }
  for (int i = 0; i < 4 * n; ++i)
    dq[i] = 1.0;
  rotation(w, ref, n);
  __poseidon_fp_optimize<void>((void *)rotation, enzyme_dup, w, dw, enzyme_dup, q, dq, n);
}

// RUN: %poseidon_explain %S/Inputs/explain/formatter/rotation.explain.json --format source --source %s --clang %clang | FileCheck --check-prefix=SOURCE %s
// RUN: %poseidon_explain %S/Inputs/explain/formatter/rotation.explain.json --format remarks | FileCheck --check-prefix=REMARKS %s
// RUN: %poseidon_explain %S/Inputs/explain/formatter/rotation.explain.json --format report | FileCheck --check-prefix=REPORT %s
// REQUIRES: poseidon

// The applied candidate moves the two library calls of one FP subgraph to
// float; each is rewritten where it is written, with its operands cast.

// SOURCE: {{^}}    double angle = std::sqrt(x * x + y * y + z * z);{{$}}
// SOURCE-NEXT: {{^}}    double s = (double)sinf((float)(0.5 * angle)) / angle;  // Poseidon: sin in float [PT0_10]{{$}}
// SOURCE-NEXT: {{^}}    q[4 * i] = (double)cosf((float)(0.5 * angle));  // Poseidon: cos in float [PT0_10]{{$}}
// SOURCE-NEXT: {{^}}    q[4 * i + 1] = s * x;{{$}}

// REMARKS: explain_formatter_rotation.cpp:32:3: note: Poseidon site rotation: accuracy target tau = 1e-06; modelled error as written 1e-17, after 1.3e-08; applied PT0_10; cost 7156000 -> 1831000
// REMARKS-NEXT: explain_formatter_rotation.cpp:13:16: note: Poseidon: Funcs FP64(0%) + FP32(100%): sin (13:16) and cos (14:16) in float (2 of 11 operations); modelled error 1.3e-08 [PT0_10]

// REPORT: explain_formatter_rotation.cpp:15:22  S0: x * (sin(sqrt(z * z + (x * x + y * y)) * 0.5) / sqrt(z * z + (x * x + y * y)))
// REPORT-NEXT: output 1 (explain_formatter_rotation.cpp:16:22): y * (sin(sqrt(z * z + (x * x + y * y)) * 0.5) / sqrt(z * z + (x * x + y * y)))
// REPORT: profile: y in [-0.5, -0.05], x in [0.001, 1], z in [0.02, 0.06]; 1000 executions; 11 operations; cost 7156000
// REPORT-NEXT: applied PT0_10 (precision): Funcs FP64(0%) + FP32(100%): sin (13:16) and cos (14:16) in float (2 of 11 operations)

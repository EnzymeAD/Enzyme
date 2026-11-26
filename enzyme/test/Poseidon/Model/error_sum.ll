; DEFINE: %{opt} = %opt %t/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-tau=1e-12 -poseidon-print -S
; RUN: rm -rf %t && split-file %s %t && mkdir %t/alternating
; RUN: sed 's/SumGrad = .*/SumGrad = 0.0/' %S/../Inputs/fma_site_profile/tester.fpprofile > %t/alternating/tester.fpprofile
; RUN: %{opt} -poseidon-profile-use=%t/alternating -poseidon-cache=%t/c -poseidon-explain=%t/x -o %t/out.ll 2>&1 | FileCheck %s
; RUN: FileCheck --check-prefix=IR %s < %t/out.ll
; RUN: FileCheck --check-prefix=JSON %s < %t/x/tester.explain.json
; DEFINE: %{dot} = %poseidon_clangxx -O2 -g -poseidon-profile-use=%t/prof -poseidon-tau=5e-11 \
; DEFINE:   -poseidon-enable-algebraic=0 -poseidon-cost-model=%host_cost_model -poseidon-cache=%t/dc \
; DEFINE:   -poseidon-print -DTAU=5e-11 %t/dot.cpp
; RUN: %if enzyme %{ %poseidon_clangxx -O2 -g -poseidon-profile-generate -poseidon-cache=%t/dc %t/dot.cpp -o %t/gen %}
; RUN: %if enzyme %{ env POSEIDON_PROFILE_DIR=%t/prof %t/gen %}
; RUN: %if enzyme %{ FileCheck --check-prefix=PROF %s < %t/prof/_Z3dotPKdS0_Pdii.fpprofile %}
; RUN: %if enzyme %{ %{dot} -o %t/opt 2>&1 | FileCheck --check-prefix=SOLVE %s %}
; RUN: %if enzyme %{ %t/opt | FileCheck --check-prefix=ERR %s %}

; An operation's accuracy weight is its summed |gradient|: with gradients that
; cancel in sign (SumGrad 0), narrowing is priced, and refused at 1e-12.

; CHECK: Initial AccuracyCost: 8.094653e-14
; CHECK: {{^}}---------------------------{{$}}
; CHECK-NEXT: 9.483896e-05		-2.000000e+02		All FP64(0%) + FP32(100%)
; CHECK-NEXT: 6.665215e-05		0.000000e+00		All FP64(50%) + FP32(50%)
; CHECK-NOT: Applying solution
; CHECK: [poseidon] preprocess_tester: tau=1.000000e-12 S=1.500000e+02 A0=8.094653e-14 selected cost=0.000000e+00 accCost=0.000000e+00 rel=5.396436e-16
; CHECK-NOT: Applying solution
; CHECK: [poseidon] pin: {{$}}
; CHECK-NOT: Applying solution
; CHECK: [poseidon] no rewrite applied for preprocess_tester; calling original body tester (bit-transparent no-op)
; IR-LABEL: define double @tester(double %x, double %y, double %z)
; IR-NEXT: entry:
; IR-NEXT: %add = fadd fast double %x, %y
; IR-NEXT: %mul = fmul fast double %add, %z
; IR-NEXT: ret double %mul
; IR-LABEL: define double @test_opt(
; IR-NEXT: entry:
; IR-NEXT: [[R:%.*]] = call double @tester(double %x, double %y, double %z)
; IR-NEXT: ret double [[R]]
; JSON: "left_as_written":
; JSON-NEXT: "ops": [
; JSON-NEXT: {
; JSON-NEXT: "executions": 100,
; JSON-NEXT: "gradient": 25,
; JSON-NEXT: "max": 15,
; JSON-NEXT: "min": 3,
; JSON-NEXT: "opcode": "+",
; JSON: "executions": 100,
; JSON-NEXT: "gradient": 50,
; JSON-NEXT: "max": 150,
; JSON-NEXT: "min": 6,
; JSON-NEXT: "opcode": "*",

; dot.cpp: each output is a 64-term dot product plus a product p of four
; inputs. Narrowing p's multiplies (FP64 30% + FP32 70%) rounds one value per
; output, the dot product 64 independent partial sums. A profile with the
; squared sums sums squared errors, and the narrowing, which measures 4.4e-10,
; is refused at 5e-11.
; PROF: {{^}}0{{$}}
; PROF: SumSqGrad = 1.0434558885{{[0-9]+}}e+02
; PROF-NEXT: SumSqSens = 3.6372672733{{[0-9]+}}e+01
; PROF-NEXT: Exec = 512
; PROF: {{^}}4{{$}}
; PROF: SumSqGrad = 3.5990405341{{[0-9]+}}e+04
; PROF-NEXT: SumSqSens = 1.5421053491{{[0-9]+}}e+07
; PROF-NEXT: Exec = 32768
; PROF-NOT: {{^}}5{{$}}
; SOLVE: Initial AccuracyCost: 5.545200e-29
; SOLVE: {{^}}---------------------------{{$}}
; SOLVE-NEXT: 1.412603e-09		-3.072000e+03		All FP64(0%) + FP32(100%)
; SOLVE-NEXT: 1.516270e-13		-2.048000e+03		All FP64(30%) + FP32(70%)
; SOLVE-NOT: Applying solution
; SOLVE: [poseidon] preprocess__Z3dotPKdS0_Pdii_poseidon_body: tau=5.000000e-11 S=1.612859e+07 A0=5.545200e-29 selected cost=0.000000e+00 accCost=0.000000e+00 rel=1.854217e-18
; SOLVE-NEXT: [poseidon] pin: {{$}}
; SOLVE-NOT: Applying solution
; SOLVE: [poseidon] no rewrite applied for preprocess__Z3dotPKdS0_Pdii_poseidon_body; calling original body _Z3dotPKdS0_Pdii_poseidon_body (bit-transparent no-op)
; ERR: {{^}}relative error 0.00e+00 within the target{{$}}

;--- tester.ll
define double @tester(double %x, double %y, double %z) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %z
  ret double %mul
}

define double @test_opt(double %x, double %y, double %z) {
entry:
  %0 = tail call double (double (double, double, double)*, ...) @__poseidon_fp_optimize(double (double, double, double)* nonnull @tester, double %x, double %y, double %z)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

;--- dot.cpp
#include "poseidon/poseidon.h"
#include <cmath>
#include <cstdio>

constexpr int n = 512, k = 64;

POSEIDON_OPTIMIZE void dot(const double *x, const double *w, double *y,
                           int n, int k) {
  for (int i = 0; i < n; ++i) {
    double acc = 0.0;
#pragma clang loop unroll(disable) vectorize(disable)
    for (int j = 0; j < k; ++j)
      acc += w[j] * x[i * k + j];
    double p = x[i * k] * x[i * k + 1] * x[i * k + 2] * x[i * k + 3];
    y[i] = acc + p * w[0];
  }
}

__attribute__((noinline)) void reference(const double *x, const double *w,
                                         double *y, int n, int k) {
  for (int i = 0; i < n; ++i) {
    double acc = 0.0;
#pragma clang loop unroll(disable) vectorize(disable)
    for (int j = 0; j < k; ++j)
      acc += w[j] * x[i * k + j];
    double p = x[i * k] * x[i * k + 1] * x[i * k + 2] * x[i * k + 3];
    y[i] = acc + p * w[0];
  }
}

static double x[n * k], w[k], y[n], r[n];

int main() {
  unsigned s = 12345;
  auto next = [&s] {
    s = s * 1103515245u + 12345u;
    return 0.5 + 0.5 * ((s >> 8) & 0xffffff) / 16777216.0;
  };
  for (double &v : x)
    v = next();
  for (double &v : w)
    v = next();
  dot(x, w, y, n, k);
  reference(x, w, r, n, k);
#ifdef TAU
  double num = 0.0, den = 0.0;
  for (int i = 0; i < n; ++i) {
    num += (y[i] - r[i]) * (y[i] - r[i]);
    den += r[i] * r[i];
  }
  double rel = std::sqrt(num / den);
  printf("relative error %.2e %s the target\n", rel,
         rel <= TAU ? "within" : "ABOVE");
#endif
}

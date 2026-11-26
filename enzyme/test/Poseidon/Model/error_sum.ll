; DEFINE: %{opt} = %opt %t/tester.ll %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-cost-model=%cost_models/cm_cpu_native.csv \
; DEFINE:   -poseidon-tau=1e-12 -poseidon-print -S
; RUN: rm -rf %t && split-file %s %t && mkdir %t/alternating
; RUN: sed 's/SumGrad = .*/SumGrad = 0.0/' %S/../Inputs/fma_site_profile/tester.fpprofile > %t/alternating/tester.fpprofile
; RUN: %{opt} -poseidon-profile-use=%t/alternating -poseidon-cache=%t/c -poseidon-explain=%t/x -o %t/out.ll 2>&1 | FileCheck %s
; RUN: FileCheck --check-prefix=IR %s < %t/out.ll
; RUN: FileCheck --check-prefix=JSON %s < %t/x/tester.explain.json

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

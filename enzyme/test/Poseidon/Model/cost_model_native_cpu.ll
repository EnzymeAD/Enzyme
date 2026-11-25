; DEFINE: %{opt} = %opt %s %loadPoseidon -poseidon-profile-use=%S/../Inputs/joint_profiles \
; DEFINE:   -poseidon-enable-rewrite=false -poseidon-enable-precision=true -S
; RUN: rm -rf %t && mkdir %t
; RUN: %{opt} -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-print -poseidon-cost-model=%cost_models/cm_cpu_native.csv -poseidon-cache=%t/native 2>&1 \
; RUN:   | FileCheck --check-prefix=NATIVE %s
; RUN: not %{opt} -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-print -poseidon-cost-model=%cost_models/cm_cpu_missing.csv -poseidon-cache=%t/missing 2>&1 \
; RUN:   | FileCheck --check-prefix=MISSING %s
; RUN: not %{opt} -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-print -poseidon-cost-model=%cost_models/cm_cpu_foreign.csv -poseidon-cache=%t/foreign 2>&1 \
; RUN:   | FileCheck --check-prefix=FOREIGN %s
; RUN: not %{opt} -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-print -poseidon-cost-model=%cost_models/cm_cpu_noarch.csv -poseidon-cache=%t/noarch 2>&1 \
; RUN:   | FileCheck --check-prefix=NOARCH %s
; RUN: not %{opt} -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-print -poseidon-cache=%t/none 2>&1 \
; RUN:   | FileCheck --check-prefix=NOMODEL %s
; RUN: (cat %cost_models/cm_cpu_native.csv; echo 'fadd,float,1') > %t/twice.csv
; RUN: not %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/twice.csv -poseidon-cache=%t/twice 2>&1 \
; RUN:   | FileCheck --check-prefix=TWICE %s
; RUN: (cat %cost_models/cm_cpu_native.csv; echo 'fwibble,double,3') > %t/unknown.csv
; RUN: not %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/unknown.csv -poseidon-cache=%t/unknown 2>&1 \
; RUN:   | FileCheck --check-prefix=UNKNOWN -DLINE=$(wc -l < %t/unknown.csv) %s
; RUN: (cat %cost_models/cm_cpu_native.csv; echo 'fadd,float') > %t/short.csv
; RUN: not %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/short.csv -poseidon-cache=%t/short 2>&1 \
; RUN:   | FileCheck --check-prefix=SHORT -DLINE=$(wc -l < %t/short.csv) %s
; RUN: (cat %cost_models/cm_cpu_native.csv; echo 'fadd,bf16,nan') > %t/nan.csv
; RUN: not %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/nan.csv -poseidon-cache=%t/nan 2>&1 \
; RUN:   | FileCheck --check-prefix=NAN -DLINE=$(wc -l < %t/nan.csv) %s
; RUN: (cat %cost_models/cm_cpu_native.csv; echo 'fadd,bf16,-1') > %t/negative.csv
; RUN: not %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/negative.csv -poseidon-cache=%t/negative 2>&1 \
; RUN:   | FileCheck --check-prefix=NEGATIVE -DLINE=$(wc -l < %t/negative.csv) %s
; RUN: sed 's/^# scalar_types=double,float$/# scalar_types=double,float,fp16/' %cost_models/cm_cpu_native.csv > %t/scalar.csv
; RUN: not %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/scalar.csv -poseidon-cache=%t/scalar 2>&1 \
; RUN:   | FileCheck --check-prefix=SCALAR %s
; RUN: (cat %cost_models/cm_cpu_native.csv; echo '# native_arch=x86-64') > %t/repeated.csv
; RUN: not %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/repeated.csv -poseidon-cache=%t/repeated 2>&1 \
; RUN:   | FileCheck --check-prefix=REPEATED -DLINE=$(wc -l < %t/repeated.csv) %s
; RUN: sed 's/^fadd,float,1$/fadd,float,1e12/' %cost_models/cm_cpu_native.csv > %t/axis.csv
; RUN: not %{opt} -passes="poseidon,poseidon-finalize" -poseidon-cost-model=%t/axis.csv -poseidon-cache=%t/axis 2>&1 \
; RUN:   | FileCheck --check-prefix=AXIS %s
; The measured CSV is the only price source. A model whose native_arch matches the function's target-cpu is consumed as is (2 FP64 ops at 4 = 8); a model missing a required row, a model measured on another device, a model with no native_arch header, a model with a row twice, a line that is not a row, a row of no family Poseidon reads or with a price that is not a finite non-negative number, a header given twice, a scalar type that is no format, and no model at all each stop the compile with an error instead of substituting a price. So does a row that prices a candidate's cost change beyond what the DP's cost axis holds.

define double @tester_a(double %x, double %y) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %x
  ret double %mul
}

define double @site_a(double %x, double %y) #0 {
entry:
  %0 = tail call double (double (double, double)*, ...) @__poseidon_fp_optimize(double (double, double)* nonnull @tester_a, double %x, double %y, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(double (double, double)*, ...)

attributes #0 = { "target-cpu"="x86-64" }

; NATIVE: Initial ComputationCost: 8.000000e+00
; NATIVE: {{^}}1.132287e-04 -1.920000e+02 All FP64(0%) + FP32(100%){{$}}
; NATIVE: [poseidon] Finished Optimization

; MISSING: error: {{.*}}Poseidon: cost model {{.*}}cm_cpu_missing.csv has no row 'fmul,double'; measure it with poseidon-calibrate{{$}}
; MISSING-NOT: Initial ComputationCost
; MISSING-NOT: [poseidon] Finished Optimization

; FOREIGN: error: {{.*}}Poseidon: cost model {{.*}}cm_cpu_foreign.csv was measured on native_arch 'sm_120' but preprocess_tester_a compiles for target-cpu 'x86-64'{{$}}
; FOREIGN-NOT: Initial ComputationCost
; FOREIGN-NOT: [poseidon] Finished Optimization

; NOARCH: error: {{.*}}Poseidon: cost model {{.*}}cm_cpu_noarch.csv carries no '# native_arch=' header, so it cannot be matched against the compile target{{$}}
; NOARCH-NOT: Initial ComputationCost
; NOARCH-NOT: [poseidon] Finished Optimization

; NOMODEL: error: {{.*}}Poseidon: no cost model for target-cpu 'x86-64' in {{.*}}xdg-cache{{.}}poseidon; measure this host with 'poseidon-calibrate --cpu x86-64', or name a model with -poseidon-cost-model=<csv> or POSEIDON_COST_MODEL{{$}}
; NOMODEL-NOT: Initial ComputationCost
; NOMODEL-NOT: [poseidon] Finished Optimization

; TWICE: error: {{.*}}Poseidon: cost model {{.*}}twice.csv has the row 'fadd,float' twice{{$}}
; TWICE-NOT: [poseidon] Finished Optimization

; UNKNOWN: error: {{.*}}Poseidon: cost model {{.*}}unknown.csv line [[LINE]] has the row 'fwibble,double', of no row family Poseidon reads (poseidon-calibrate --help lists them); remove the line, or compile with the Poseidon whose poseidon-calibrate wrote it{{$}}
; UNKNOWN-NOT: [poseidon] Finished Optimization

; SHORT: error: {{.*}}Poseidon: cost model {{.*}}short.csv line [[LINE]] is not '<opcode>,<type>,<cost>': fadd,float{{$}}

; NAN: error: {{.*}}Poseidon: cost model {{.*}}nan.csv line [[LINE]] prices the row 'fadd,bf16' at nan, which is not a finite non-negative cost{{$}}

; NEGATIVE: error: {{.*}}Poseidon: cost model {{.*}}negative.csv line [[LINE]] prices the row 'fadd,bf16' at -1, which is not a finite non-negative cost{{$}}

; SCALAR: error: {{.*}}Poseidon: cost model {{.*}}scalar.csv line 3 lists the scalar type 'fp16', which is none of bf16, half, float, double, fp80 and fp128{{$}}

; REPEATED: error: {{.*}}Poseidon: cost model {{.*}}repeated.csv line [[LINE]] repeats the '# native_arch=' header{{$}}
; SHORT-NOT: [poseidon] Finished Optimization

; AXIS: error: {{.*}}in function preprocess_tester_a {{.*}}: Poseidon: precision change All FP64(0%) + FP32(100%) (#0) changes the cost of preprocess_tester_a by 6.400000e+13, which does not fit the DP's cost axis; a cost-model row prices it beyond 2^30 times the profiled cost{{$}}
; AXIS-NOT: [poseidon] Finished Optimization

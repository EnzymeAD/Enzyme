; RUN: rm -rf %t && mkdir %t
; RUN: printf '#!/bin/sh\nexec %python -S "$@"\n' > %t/python && chmod +x %t/python
; RUN: not %poseidon_calibrate --cpu x86-64 --python %t/python --out %t/cm.csv 2>&1 | FileCheck --check-prefix=PACKAGE %s
; RUN: not %poseidon_calibrate --cpu x86-64 --gpu 0 --out %t/cm.csv 2>&1 | FileCheck --check-prefix=BOTH %s
; RUN: not %poseidon_calibrate --cpu x86-64 --only ops,direct --out %t/cm.csv 2>&1 | FileCheck --check-prefix=GPUARM %s
; RUN: not %poseidon_calibrate --cpu x86-64 --only opz --out %t/cm.csv 2>&1 | FileCheck --check-prefix=NOARM %s
; RUN: test ! -e %t/cm.csv
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%S/../../Inputs/joint_profiles -poseidon-enable-algebraic=false -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=COMPILE %s

; poseidon-calibrate --cpu stops with one error and writes nothing; a compile
; for a target-cpu with no model in the user cache names the command for one.

define double @tester_a(double %x, double %y) #0 {
entry:
  %add = fadd fast double %x, %y
  %mul = fmul fast double %add, %x
  ret double %mul
}

define double @site_a(double %x, double %y) #0 {
entry:
  %0 = tail call double (ptr, ...) @__poseidon_fp_optimize(ptr nonnull @tester_a, double %x, double %y, metadata !"poseidon_tau", double 0.5)
  ret double %0
}

declare double @__poseidon_fp_optimize(ptr, ...)

attributes #0 = { "target-cpu"="x86-64-v3" }

; PACKAGE: poseidon-calibrate: {{.*}}python cannot import the Python package llvmlite, which the CPU microbenchmark needs; install it (pip install llvmlite) or pass --python <interpreter>{{$}}
; PACKAGE-NOT: poseidon-calibrate:

; BOTH: poseidon-calibrate: --cpu measures this host and --gpu a GPU; pass one of them{{$}}

; GPUARM: poseidon-calibrate: the direct arm measures a GPU, and --cpu a host{{$}}

; NOARM: poseidon-calibrate: --only: there is no arm 'opz'{{$}}

; COMPILE: error: {{.*}}Poseidon: no cost model for target-cpu 'x86-64-v3' in {{.*}}xdg-cache{{.}}poseidon; measure this host with 'poseidon-calibrate --cpu x86-64-v3', or name a model with -poseidon-cost-model=<csv> or POSEIDON_COST_MODEL{{$}}

; RUN: rm -rf %t && mkdir %t
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-tau=1e-3 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-cache=%t/normal 2>&1 | FileCheck --check-prefix=NORMAL %s
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-tau=1e-3 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../../Inputs/fp16_range/matmul_tiny -poseidon-early-prune -poseidon-cache=%t/tiny 2>&1 | FileCheck --check-prefix=TINY %s
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-0.5 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../../Inputs/fp16_range/matmul_tiny -poseidon-early-prune -poseidon-cache=%t/budget 2>&1 | FileCheck --check-prefix=BUDGET %s
; RUN: mkdir %t/kappa && sed '/^CanonicalHash = /a Kappa = 1e-12\nKappaMark = ok' %S/../../Inputs/fp16_range/matmul_tiny/matmul_body.fpprofile > %t/kappa/matmul_body.fpprofile
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-0.5 -poseidon-print -S -o /dev/null -poseidon-profile-use=%t/kappa -poseidon-early-prune -poseidon-cache=%t/kappa/c 2>&1 | FileCheck --check-prefixes=KAPPA,BUDGET %s
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-0.5 -poseidon-matmul-exclude-out-of-range=false -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../../Inputs/fp16_range/matmul_tiny -poseidon-early-prune -poseidon-cache=%t/nopenalty 2>&1 | FileCheck --check-prefix=NOPENALTY %s

; No FP16 raise or TCEC chain is offered for operands below FP16's smallest
; subnormal (Inputs/fp16_range/matmul_tiny: A[i] = 1e-9 * (1 + 0.001 i), recorded
; on an sm_120 GPU), on the tolerance and on the budget path.

; NORMAL: [poseidon] domain error matmul in=f16 acc=f32 p95_relErr=3.137966e-04
; NORMAL: wmma m16n16k16 f16/f32: dcost=-3.932160e+06 dacc=1.538569e+01
; NORMAL-NEXT: tcec n=2 wmma m16n16k16 f16/f32: dcost=-3.866624e+06 dacc=7.710461e-03
; NORMAL-NEXT: tcec n=3 wmma m16n16k16 f16/f32: dcost=-3.801088e+06 dacc=2.880546e-03
; NORMAL: [poseidon] preprocess_matmul_body: tau=1.000000e-03 S=4.903078e+04 A0=0.000000e+00 selected cost=-9.375000e-01 accCost=1.538569e+01 rel=3.137966e-04
; NORMAL: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)

; TINY: [poseidon] domain error matmul in=f16 acc=f32 p95_relErr=1.000000e+00
; TINY: [poseidon]   not offered: wmma m16n16k16 f16/f32: f16 flushes the smallest profiled operand magnitude, 1e-09, to zero
; TINY-NEXT: [poseidon]   not offered: tcec n=2 wmma m16n16k16 f16/f32: f16 flushes the smallest profiled operand magnitude, 1e-09, to zero
; TINY-NEXT: [poseidon]   not offered: tcec n=3 wmma m16n16k16 f16/f32: f16 flushes the smallest profiled operand magnitude, 1e-09, to zero
; TINY: (no candidates emitted)
; TINY-NOT: Applying solution for matmul[0]
; TINY: [poseidon] preprocess_matmul_body: tau=1.000000e-03 S=4.903078e-05 A0=0.000000e+00 selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00
; TINY-NOT: Applying solution for matmul[0]
; TINY: Finished optimizing preprocess_matmul_body

; A condition number of 1e-12 leaves them out all the same; with
; -poseidon-matmul-exclude-out-of-range=false each is offered at its error.
; BUDGET: not offered: wmma m16n16k16 f16/f32
; KAPPA: measured condition number 1.000000e-12 (ok), profiled sensitivity 4.903078e-05, error scale 2.039535e-08
; BUDGET: [poseidon] No solution found within the computation cost budget
; BUDGET-NOT: Applying solution for matmul[0]
; NOPENALTY: wmma m16n16k16 f16/f32: dcost=-3.932160e+06 dacc=2.435977e-02
; NOPENALTY: Computation cost budget used: -9.062500e-01
; NOPENALTY: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)

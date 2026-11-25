; RUN: rm -rf %t && mkdir %t
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-tau=1e-3 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-cache=%t/normal 2>&1 | FileCheck --check-prefix=NORMAL %s
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-tau=1e-3 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../../Inputs/fp16_range/matmul_tiny -poseidon-cache=%t/tiny 2>&1 | FileCheck --check-prefix=TINY %s
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-0.5 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../../Inputs/fp16_range/matmul_tiny -poseidon-cache=%t/budget 2>&1 | FileCheck --check-prefix=BUDGET %s
; RUN: mkdir %t/kappa && sed '/^CanonicalHash = /a Kappa = 1e-12\nKappaMark = ok' %S/../../Inputs/fp16_range/matmul_tiny/matmul_body.fpprofile > %t/kappa/matmul_body.fpprofile
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-0.5 -poseidon-print -S -o /dev/null -poseidon-profile-use=%t/kappa -poseidon-cache=%t/kappa/c 2>&1 | FileCheck --check-prefixes=KAPPA,BUDGET %s
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-rewrite=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-matmul-library-calls=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-cost-budget=-0.5 -poseidon-matmul-exclude-out-of-range=false -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../../Inputs/fp16_range/matmul_tiny -poseidon-cache=%t/nopenalty 2>&1 | FileCheck --check-prefix=NOPENALTY %s

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

; On the budget path as well no flushing format is offered, whatever the
; site's error scale: a condition number of 1e-12 scales every error by 2e-8
; and leaves the candidates out all the same.
; -poseidon-matmul-exclude-out-of-range=false offers each at its modelled
; error, and the cheapest one that reaches the budget is selected.
; BUDGET: not offered: wmma m16n16k16 f16/f32
; KAPPA: measured condition number 1.000000e-12 (ok), profiled sensitivity 4.903078e-05, error scale 2.039535e-08
; BUDGET: [poseidon] No solution found within the computation cost budget
; BUDGET-NOT: Applying solution for matmul[0]
; NOPENALTY: wmma m16n16k16 f16/f32: dcost=-3.932160e+06 dacc=2.435977e-02
; NOPENALTY: Computation cost budget used: -9.062500e-01
; NOPENALTY: Applying solution for matmul[0] -> tcec n=3 wmma m16n16k16 f16/f32 (#2)

; FP16 flush pricing of a matrix-product raise. The product of raise_scalar_tcec_nvptx.ll
; (Inputs/stale_cache/matmul_device.ll) under two profiles: Inputs/matmul_site_profile (A in [1, 1.255]) and Inputs/fp16_range/matmul_tiny (A scaled by 1e-9,
; below FP16's smallest subnormal 5.96e-8). Rounding A to FP16 flushes it to zero, so the direct FP16 raise
; models a p95 relative error of 1 and the tolerance 1e-3 that takes it on the first profile refuses it on
; the second; every candidate whose operands pass through the FP16 exponent range (the direct raise and
; both TCEC chains) is not offered.
; matmul_tiny profile: the source raise_scalar_tcec_nvptx.ll gives with A[i] = 1e-9 * (1.0 + 0.001 * i), built with
; the RUN lines of Integration/raise_scalar_matmul_f64.cu (Poseidon + Enzyme plugins,
; -poseidon-profile-generate, FPProfilerCUDA.cu, libposeidon_profile.a) and run on an sm_120 GPU.

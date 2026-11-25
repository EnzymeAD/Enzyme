; RUN: rm -rf %t && mkdir %t
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-tau=1e-3 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-cache=%t/normal 2>&1 | FileCheck --check-prefix=NORMAL %s
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-ozaki-host-dispatch=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic.csv -poseidon-tau=1e-3 -poseidon-print -S -o /dev/null -poseidon-profile-use=%S/../Inputs/fp16_range/matmul_tiny -poseidon-cache=%t/tiny 2>&1 | FileCheck --check-prefix=TINY %s

; NORMAL: [poseidon] domain error matmul in=f16 acc=f32 p95_relErr=3.137966e-04
; NORMAL: wmma m16n16k16 f16/f32: dcost=-3.932160e+06 dacc=9.867296e+00
; NORMAL-NEXT: tcec n=2 wmma m16n16k16 f16/f32: dcost=-3.866624e+06 dacc=5.093297e-03
; NORMAL-NEXT: tcec n=3 wmma m16n16k16 f16/f32: dcost=-3.801088e+06 dacc=1.896120e-03
; NORMAL: [poseidon] error budget: matmul -> wmma m16n16k16 f16/f32 (domain err 3.137966e-04 <= 1.000000e-03 at confidence 9.500000e-01)
; NORMAL: Applying solution for matmul[0] -> wmma m16n16k16 f16/f32 (#0)

; TINY: [poseidon] domain error matmul in=f16 acc=f32 p95_relErr=1.000000e+00
; TINY: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=INF
; TINY: #1 tcec n=2 wmma m16n16k16 f16/f32  compCost/MAC=4.000000e+01  rel=7.812500e-02  domainError=INF
; TINY: wmma m16n16k16 f16/f32: dcost=-3.932160e+06 dacc=1.000000e+30
; TINY-NEXT: tcec n=2 wmma m16n16k16 f16/f32: dcost=-3.866624e+06 dacc=1.000000e+30
; TINY-NEXT: tcec n=3 wmma m16n16k16 f16/f32: dcost=-3.801088e+06 dacc=1.000000e+30
; TINY-NOT: error budget: matmul -> wmma m16n16k16 f16/f32
; TINY-NOT: Applying solution for matmul[0]
; TINY: error budget: matmul -> baseline F64 (no rewrite clears 1.000000e-03
; TINY: Finished optimizing preprocess_matmul_body

; FP16 flush pricing of a matrix-product raise. The product of raise_scalar_tcec_nvptx.ll
; (Inputs/stale_cache/matmul_device.ll) under two profiles: Inputs/raise_tcec (A in [1, 1.255]) and Inputs/fp16_range/matmul_tiny (A scaled by 1e-9,
; below FP16's smallest subnormal 5.96e-8). Rounding A to FP16 flushes it to zero, so the direct FP16 raise
; models a p95 relative error of 1 and the tolerance 1e-3 that takes it on the first profile refuses it on
; the second; every candidate whose operands pass through the FP16 exponent range (the direct raise and
; both TCEC chains) also has -poseidon-exponent-penalty (1e30) added to its accuracy cost.
; matmul_tiny profile: the source raise_scalar_tcec_nvptx.ll gives with A[i] = 1e-9 * (1.0 + 0.001 * i), built with
; the RUN lines of Integration/raise_scalar_matmul_f64.cu (Poseidon + Enzyme plugins,
; -poseidon-profile-generate, FPProfilerCUDA.cu, libposeidon_profile.a) and run on an sm_120 GPU.

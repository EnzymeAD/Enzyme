; DEFINE: %{opt} = %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%t/prof -poseidon-early-prune -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul -poseidon-print -S -o /dev/null
; DEFINE: %{hip} = %clang -x hip --offload-arch=gfx90a --rocm-path=%rocm_path --cuda-device-only --no-gpu-bundle-output -cuid=0 -O2 -c -DTAU=1e-3 -I%poseidon_profile_inc %loadClangPoseidon -Xclang -mllvm -Xclang -poseidon-profile-use=%t/hip -Xclang -mllvm -Xclang -poseidon-compilation-id=gfx90a -Xclang -mllvm -Xclang -poseidon-print -Xclang -mllvm -Xclang -poseidon-enable-algebraic=0 -Xclang -mllvm -Xclang -poseidon-matmul-library-calls %S/runtime_shape_library_call.cu
; RUN: rm -rf %t && mkdir -p %t/prof %t/hip
; RUN: sed 's/^\tOperand\[0\] = .*/\tOperand[0] = [3.00000000000000008e-05, 1.25500000000000012e+00, 3.00000000000000008e-05]/' %S/../../Inputs/matmul_site_profile/matmul_body.fpprofile > %t/prof/matmul_body.fpprofile
; RUN: %{opt} -poseidon-tau=1e-3 -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_floor_subnormal.csv -poseidon-cache=%t/subnormal 2>&1 | FileCheck --check-prefixes=PRICED,KEEP --implicit-check-not=_operand_min, %s
; RUN: %{opt} -poseidon-tau=1e-3 -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_floor_normal.csv -poseidon-cache=%t/normal 2>&1 | FileCheck --check-prefixes=PRICED,FLUSH --implicit-check-not=_operand_min, %s
; RUN: grep -v _operand_min, %cost_models/cm_sm_120_synthetic_floor_subnormal.csv > %t/cm_sm_120_absent.csv
; RUN: %{opt} -poseidon-tau=1e-3 -poseidon-cost-model=%t/cm_sm_120_absent.csv -poseidon-cache=%t/absent 2>&1 | FileCheck --check-prefixes=ABSENT,PRICED,KEEP %s
; RUN: diff -r -x 'dp_*' -x 'launches_*' %t/subnormal %t/absent
; RUN: not grep '"operand min"' %t/subnormal/matmul_accuracy_*.json
; RUN: grep -h '"operand min"' %t/normal/matmul_accuracy_*.json | sort | FileCheck --check-prefix=KEY %s
; RUN: grep -v library_direct_operand_min,f16_f32, %cost_models/cm_sm_120_synthetic_floor_subnormal.csv > %t/cm_sm_120_partial.csv
; RUN: %{opt} -poseidon-tau=1e-3 -poseidon-cost-model=%t/cm_sm_120_partial.csv -poseidon-cache=%t/partial 2>&1 | FileCheck --check-prefix=PARTIAL %s
; RUN: %if tcec-cumpsgemm %{ cp %cost_models/cm_sm_120_synthetic_floor_subnormal.csv %t/tcec_subnormal.csv && echo library_tcec_rel,fp16tcec,0.02734375 >> %t/tcec_subnormal.csv %}
; RUN: %if tcec-cumpsgemm %{ cp %cost_models/cm_sm_120_synthetic_floor_normal.csv %t/tcec_normal.csv && echo library_tcec_rel,fp16tcec,0.02734375 >> %t/tcec_normal.csv %}
; RUN: %if tcec-cumpsgemm %{ %{opt} -poseidon-tau=1e-6 -poseidon-cost-model=%t/tcec_subnormal.csv -poseidon-cache=%t/tcec-subnormal 2>&1 | FileCheck --check-prefix=TCEC-KEEP %s %}
; RUN: %if tcec-cumpsgemm %{ %{opt} -poseidon-tau=1e-6 -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_tcec.csv -poseidon-cache=%t/tcec-absent 2>&1 | FileCheck --check-prefixes=TCEC-ABSENT,TCEC-KEEP %s %}
; RUN: %if tcec-cumpsgemm %{ diff -r -x 'dp_*' -x 'launches_*' %t/tcec-subnormal %t/tcec-absent %}
; RUN: %if tcec-cumpsgemm %{ %{opt} -poseidon-tau=1e-6 -poseidon-cost-model=%t/tcec_normal.csv -poseidon-cache=%t/tcec-normal 2>&1 | FileCheck --check-prefix=TCEC-FLUSH %s %}
; RUN: %if tcec-cumpsgemm %{ cat %t/tcec-normal/matmul_accuracy_*.json | FileCheck --check-prefix=TCEC-KEY %s %}
; RUN: %if amdgpu-registered-target && rocm %{ cp %S/../../Inputs/host_gemm_hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~7202e5615af3fda3.fpprofile %t/hip/_ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_~caa5c2a240f1bcf9.fpprofile %}
; RUN: %if amdgpu-registered-target && rocm %{ %{hip} -Xclang -mllvm -Xclang -poseidon-cache=%t/hip-subnormal -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx90a_synthetic_floor_subnormal.csv -o %t/hip-subnormal.co 2>&1 | FileCheck --check-prefixes=HIP,HIP-KEEP --implicit-check-not=_operand_min, %s %}
; RUN: %if amdgpu-registered-target && rocm %{ %{hip} -Xclang -mllvm -Xclang -poseidon-cache=%t/hip-normal -Xclang -mllvm -Xclang -poseidon-cost-model=%cost_models/cm_gfx90a_synthetic_floor_normal.csv -o %t/hip-normal.co 2>&1 | FileCheck --check-prefixes=HIP,HIP-FLUSH --implicit-check-not=_operand_min, %s %}
; RUN: %if amdgpu-registered-target && rocm %{ grep -v _operand_min, %cost_models/cm_gfx90a_synthetic_floor_subnormal.csv > %t/cm_gfx90a_absent.csv %}
; RUN: %if amdgpu-registered-target && rocm %{ %{hip} -Xclang -mllvm -Xclang -poseidon-cache=%t/hip-absent -Xclang -mllvm -Xclang -poseidon-cost-model=%t/cm_gfx90a_absent.csv -o %t/hip-absent.co 2>&1 | FileCheck --check-prefixes=HIP-ABSENT,HIP,HIP-KEEP %s %}
; RUN: %if amdgpu-registered-target && rocm %{ diff -r -x 'dp_*' -x 'launches_*' %t/hip-subnormal %t/hip-absent %}
; RUN: %if amdgpu-registered-target && rocm %{ grep -h '"operand min"' %t/hip-normal/matmul_accuracy_*.json | sort | FileCheck --check-prefix=HIP-KEY %s %}
; RUN: %if amdgpu-registered-target && rocm %{ grep -v library_direct_operand_min,bf16_f32, %cost_models/cm_gfx90a_synthetic_floor_normal.csv > %t/cm_gfx90a_partial.csv %}
; RUN: %if amdgpu-registered-target && rocm %{ %{hip} -Xclang -mllvm -Xclang -poseidon-cache=%t/hip-partial -Xclang -mllvm -Xclang -poseidon-cost-model=%t/cm_gfx90a_partial.csv -o %t/hip-partial.co 2>&1 | FileCheck --check-prefix=HIP-PARTIAL %s %}

; The *_operand_min rows of a cost model, on the product of matmul_device.ll
; with A down to 3e-5, between FP16's smallest subnormal and smallest normal;
; on the error-corrected library call with cuMpSGEMM; and on the gfx90a library
; calls of the reduce of runtime_shape_library_call.cu, at accuracy target 1e-3.

; ABSENT:      [poseidon] matmul[0] wmma m16n16k16 f16/f32: the cost model {{.*}}/cm_sm_120_absent.csv has no 'inkernel_operand_min,f16_f32' row, so its f16 operands are taken to be kept down to 5.960464e-08, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps
; ABSENT-NEXT: [poseidon] matmul[0] tcec n=2 wmma m16n16k16 f16/f32: the cost model {{.*}}/cm_sm_120_absent.csv has no 'inkernel_operand_min,tcec_n2_f16_f32' row, so its f16 operands are taken to be kept down to 5.960464e-08, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps
; ABSENT-NEXT: [poseidon] matmul[0] tcec n=3 wmma m16n16k16 f16/f32: the cost model {{.*}}/cm_sm_120_absent.csv has no 'inkernel_operand_min,tcec_n3_f16_f32' row, so its f16 operands are taken to be kept down to 5.960464e-08, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps
; ABSENT-NEXT: [poseidon] matmul[0] direct-dispatch cublas f16/f32: the cost model {{.*}}/cm_sm_120_absent.csv has no 'library_direct_operand_min,f16_f32' row, so its f16 operands are taken to be kept down to 5.960464e-08, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps
; ABSENT-NEXT: [poseidon] matmul[0] direct-dispatch cublas bf16/f32: the cost model {{.*}}/cm_sm_120_absent.csv has no 'library_direct_operand_min,bf16_f32' row, so its bf16 operands are taken to be kept down to 9.183550e-41, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps
; ABSENT-NEXT: [poseidon] matmul[0] direct-dispatch cublas tf32/f32: the cost model {{.*}}/cm_sm_120_absent.csv has no 'library_direct_operand_min,tf32_f32' row, so its tf32 operands are taken to be kept down to 1.147944e-41, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps
; ABSENT-NOT: _operand_min,

; KEEP:       [poseidon] domain error matmul in=f16 acc=f32 p95_relErr=[[F16:3.137966e-04]] (sumcost=9.867296e+00)
; KEEP-NEXT:  [poseidon] domain error matmul in=f32(22b) acc=f32 p95_relErr=[[N2:1.572576e-07]] (sumcost=5.093297e-03)
; KEEP-NEXT:  [poseidon] domain error matmul in=f32(24b) acc=f32 p95_relErr=[[N3:5.874975e-08]] (sumcost=1.896120e-03)
; FLUSH:      [poseidon] domain error matmul in=f16 min=6.103516e-05 acc=f32 p95_relErr=4.745738e-04 (sumcost=1.436528e+01)
; FLUSH-NEXT: [poseidon] domain error matmul in=f32(22b) min=6.103516e-05 acc=f32 p95_relErr=2.466484e-04 (sumcost=5.285686e+00)
; FLUSH-NEXT: [poseidon] domain error matmul in=f32(24b) min=6.103516e-05 acc=f32 p95_relErr=2.465581e-04 (sumcost=5.284999e+00)
; KEEP:       [poseidon] domain error matmul in=bf16 acc=f32 p95_relErr=[[BF16:2.340575e-03]] (sumcost=7.878135e+01)
; KEEP-NEXT:  [poseidon] domain error matmul in=tf32 acc=f32 p95_relErr=[[TF32:3.137466e-04]] (sumcost=9.864277e+00)
; FLUSH:      [poseidon] domain error matmul in=bf16 min=1.175494e-38 acc=f32 p95_relErr=[[BF16:2.340575e-03]] (sumcost=7.878135e+01)
; FLUSH-NEXT: [poseidon] domain error matmul in=tf32 min=1.175494e-38 acc=f32 p95_relErr=[[TF32:3.137466e-04]] (sumcost=9.864277e+00)

; PRICED-LABEL: [poseidon] matmul[0] pricing: baseline/MAC=5.120000e+02 executions=8192
; KEEP-NEXT:   [poseidon]   #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=[[F16]]
; KEEP-NEXT:   [poseidon]   #1 tcec n=2 wmma m16n16k16 f16/f32  compCost/MAC=4.000000e+01  rel=7.812500e-02  domainError=[[N2]]
; KEEP-NEXT:   [poseidon]   #2 tcec n=3 wmma m16n16k16 f16/f32  compCost/MAC=4.800000e+01  rel=9.375000e-02  domainError=[[N3]]
; KEEP-NEXT:   [poseidon]   #3 ozaki-ii nm=8 wmma m16n16k16 s8/s32  compCost/MAC=1.600000e+01  rel=3.125000e-02  domainError=7.986923e-09
; KEEP-NEXT:   [poseidon]   #4 ozaki-ii nm=9 wmma m16n16k16 s8/s32  compCost/MAC=1.800000e+01  rel=3.515625e-02  domainError=5.811304e-10
; KEEP-NEXT:   [poseidon]   #5 ozaki-ii nm=10 wmma m16n16k16 s8/s32  compCost/MAC=2.000000e+01  rel=3.906250e-02  domainError=4.205447e-11
; KEEP-NEXT:   [poseidon]   #6 ozaki-ii nm=11 wmma m16n16k16 s8/s32  compCost/MAC=2.200000e+01  rel=4.296875e-02  domainError=2.759331e-12
; KEEP-NEXT:   [poseidon]   #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; KEEP-NEXT:   [poseidon]   #8 ozaki-ii nm=13 wmma m16n16k16 s8/s32  compCost/MAC=2.600000e+01  rel=5.078125e-02  domainError=1.220929e-14
; KEEP-NEXT:   [poseidon]   #9 ozaki-ii nm=14 wmma m16n16k16 s8/s32  compCost/MAC=2.800000e+01  rel=5.468750e-02  domainError=8.567043e-16
; KEEP-NEXT:   [poseidon]   #10 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=[[F16]]
; KEEP-NEXT:   [poseidon]   #11 direct-dispatch cublas bf16/f32  compCost/MAC=6.000000e+00  rel=1.171875e-02  domainError=[[BF16]]
; KEEP-NEXT:   [poseidon]   #12 direct-dispatch cublas tf32/f32  compCost/MAC=8.000000e+00  rel=1.562500e-02  domainError=[[TF32]]
; KEEP-NEXT:   [poseidon]   #13 native-dgemm cublas f64/f64  compCost/MAC=2.560000e+02  rel=5.000000e-01  domainError=0.000000e+00
; FLUSH-NEXT:  [poseidon]   #0 ozaki-ii nm=8 wmma m16n16k16 s8/s32  compCost/MAC=1.600000e+01  rel=3.125000e-02  domainError=7.986923e-09
; FLUSH-NEXT:  [poseidon]   #1 ozaki-ii nm=9 wmma m16n16k16 s8/s32  compCost/MAC=1.800000e+01  rel=3.515625e-02  domainError=5.811304e-10
; FLUSH-NEXT:  [poseidon]   #2 ozaki-ii nm=10 wmma m16n16k16 s8/s32  compCost/MAC=2.000000e+01  rel=3.906250e-02  domainError=4.205447e-11
; FLUSH-NEXT:  [poseidon]   #3 ozaki-ii nm=11 wmma m16n16k16 s8/s32  compCost/MAC=2.200000e+01  rel=4.296875e-02  domainError=2.759331e-12
; FLUSH-NEXT:  [poseidon]   #4 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; FLUSH-NEXT:  [poseidon]   #5 ozaki-ii nm=13 wmma m16n16k16 s8/s32  compCost/MAC=2.600000e+01  rel=5.078125e-02  domainError=1.220929e-14
; FLUSH-NEXT:  [poseidon]   #6 ozaki-ii nm=14 wmma m16n16k16 s8/s32  compCost/MAC=2.800000e+01  rel=5.468750e-02  domainError=8.567043e-16
; FLUSH-NEXT:  [poseidon]   #7 direct-dispatch cublas bf16/f32  compCost/MAC=6.000000e+00  rel=1.171875e-02  domainError=[[BF16]]
; FLUSH-NEXT:  [poseidon]   #8 direct-dispatch cublas tf32/f32  compCost/MAC=8.000000e+00  rel=1.562500e-02  domainError=[[TF32]]
; FLUSH-NEXT:  [poseidon]   #9 native-dgemm cublas f64/f64  compCost/MAC=2.560000e+02  rel=5.000000e-01  domainError=0.000000e+00
; FLUSH:       [poseidon]   not offered: wmma m16n16k16 f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 3e-05, as zero
; FLUSH-NEXT:  [poseidon]   not offered: tcec n=2 wmma m16n16k16 f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 3e-05, as zero
; FLUSH-NEXT:  [poseidon]   not offered: tcec n=3 wmma m16n16k16 f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 3e-05, as zero
; FLUSH-NEXT:  [poseidon]   not offered: direct-dispatch cublas f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 3e-05, as zero
; KEEP:        tau=1.000000e-03 {{.*}} rel=[[F16]]{{$}}
; FLUSH:       tau=1.000000e-03 {{.*}} rel=[[TF32]]{{$}}
; KEEP:        Applying solution for matmul[0] -> direct-dispatch cublas f16/f32 (#10)
; FLUSH:       Applying solution for matmul[0] -> direct-dispatch cublas tf32/f32 (#8)
; KEEP:        [poseidon] pin: matmul_body:M0=direct-dispatch cublas f16/f32
; FLUSH:       [poseidon] pin: matmul_body:M0=direct-dispatch cublas tf32/f32

; KEY:      "operand min": "1.1754943508222875e-38",
; KEY-NEXT: "operand min": "1.1754943508222875e-38",
; KEY-NEXT: "operand min": "6.103515625e-05",
; KEY-NEXT: "operand min": "6.103515625e-05",
; KEY-NEXT: "operand min": "6.103515625e-05",
; KEY-NOT:  operand min

; PARTIAL:      warning: {{.*}}Poseidon: matrix product 0, 16x16x16: direct-dispatch cublas f16/f32 is not proposed: this device's cost model carries no measured 'library_direct_operand_min,f16_f32' row; poseidon-calibrate --gpu <id> --only operand-min measures it
; PARTIAL:      [poseidon]   #9 ozaki-ii nm=14 wmma m16n16k16 s8/s32  compCost/MAC=2.800000e+01  rel=5.468750e-02  domainError=8.567043e-16
; PARTIAL-NEXT: [poseidon]   #10 direct-dispatch cublas bf16/f32  compCost/MAC=6.000000e+00  rel=1.171875e-02  domainError=[[BF16:2.340575e-03]]
; PARTIAL-NEXT: [poseidon]   #11 direct-dispatch cublas tf32/f32  compCost/MAC=8.000000e+00  rel=1.562500e-02  domainError=[[TF32:3.137466e-04]]
; PARTIAL-NEXT: [poseidon]   #12 native-dgemm cublas f64/f64  compCost/MAC=2.560000e+02  rel=5.000000e-01  domainError=0.000000e+00
; PARTIAL:      tau=1.000000e-03 {{.*}} rel=[[TF32]]{{$}}
; PARTIAL:      Applying solution for matmul[0] -> direct-dispatch cublas tf32/f32 (#11)
; PARTIAL:      [poseidon] pin: matmul_body:M0=direct-dispatch cublas tf32/f32

; TCEC-ABSENT: [poseidon] matmul[0] tcec-dispatch wmma m16n16k16 f16/f32: the cost model {{.*}}/cm_sm_120_synthetic_tcec.csv has no 'library_tcec_operand_min,fp16tcec' row, so its f16 operands are taken to be kept down to 5.960464e-08, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps

; TCEC-KEEP:       [poseidon] domain error tcec library p95_relErr=[[ERR:1.075559e-07]] (sumcost=3.371004e-03)
; TCEC-FLUSH:      [poseidon] domain error tcec library min=6.103516e-05 p95_relErr=[[ERR:2.364759e-07]] (sumcost=6.642147e-03)
; TCEC-KEEP:       [poseidon]   #10 tcec-dispatch wmma m16n16k16 f16/f32  compCost/MAC=1.400000e+01  rel=2.734375e-02  domainError=[[ERR]]
; TCEC-KEEP-NEXT:  [poseidon]   #11 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=3.137966e-04
; TCEC-KEEP-NEXT:  [poseidon]   #12 direct-dispatch cublas bf16/f32  compCost/MAC=6.000000e+00  rel=1.171875e-02  domainError=2.340575e-03
; TCEC-KEEP-NEXT:  [poseidon]   #13 direct-dispatch cublas tf32/f32  compCost/MAC=8.000000e+00  rel=1.562500e-02  domainError=3.137466e-04
; TCEC-KEEP-NEXT:  [poseidon]   #14 native-dgemm cublas f64/f64  compCost/MAC=2.560000e+02  rel=5.000000e-01  domainError=0.000000e+00
; TCEC-FLUSH:      [poseidon]   #7 tcec-dispatch wmma m16n16k16 f16/f32  compCost/MAC=1.400000e+01  rel=2.734375e-02  domainError=[[ERR]]
; TCEC-FLUSH-NEXT: [poseidon]   #8 direct-dispatch cublas bf16/f32  compCost/MAC=6.000000e+00  rel=1.171875e-02  domainError=2.340575e-03
; TCEC-FLUSH-NEXT: [poseidon]   #9 direct-dispatch cublas tf32/f32  compCost/MAC=8.000000e+00  rel=1.562500e-02  domainError=3.137466e-04
; TCEC-FLUSH-NEXT: [poseidon]   #10 native-dgemm cublas f64/f64  compCost/MAC=2.560000e+02  rel=5.000000e-01  domainError=0.000000e+00
; TCEC-FLUSH-NEXT: [poseidon]   not offered: wmma m16n16k16 f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 3e-05, as zero
; TCEC-FLUSH-NEXT: [poseidon]   not offered: tcec n=2 wmma m16n16k16 f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 3e-05, as zero
; TCEC-FLUSH-NEXT: [poseidon]   not offered: tcec n=3 wmma m16n16k16 f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 3e-05, as zero
; TCEC-FLUSH-NEXT: [poseidon]   not offered: direct-dispatch cublas f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 3e-05, as zero
; TCEC-KEEP:       tau=1.000000e-06 {{.*}} rel=[[ERR]]{{$}}
; TCEC-FLUSH:      tau=1.000000e-06 {{.*}} rel=[[ERR]]{{$}}
; TCEC-KEEP:       Applying solution for matmul[0] -> tcec-dispatch wmma m16n16k16 f16/f32 (#10)
; TCEC-FLUSH:      Applying solution for matmul[0] -> tcec-dispatch wmma m16n16k16 f16/f32 (#7)
; TCEC-KEEP:       [poseidon] pin: matmul_body:M0=tcec-dispatch wmma m16n16k16 f16/f32
; TCEC-FLUSH:      [poseidon] pin: matmul_body:M0=tcec-dispatch wmma m16n16k16 f16/f32

; TCEC-KEY:      "TCEC backend": "cuMpSGEMM",
; TCEC-KEY:      "model": "TCEC library",
; TCEC-KEY-NEXT: "operand min": "6.103515625e-05",

; HIP-ABSENT:      [poseidon] matmul[0] direct-dispatch cublas f16/f32: the cost model {{.*}}/cm_gfx90a_absent.csv has no 'library_direct_operand_min,f16_f32' row, so its f16 operands are taken to be kept down to 5.960464e-08, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps
; HIP-ABSENT-NEXT: [poseidon] matmul[0] direct-dispatch cublas bf16/f32: the cost model {{.*}}/cm_gfx90a_absent.csv has no 'library_direct_operand_min,bf16_f32' row, so its bf16 operands are taken to be kept down to 9.183550e-41, the format's smallest subnormal; poseidon-calibrate --gpu <id> --only operand-min measures what this device keeps
; HIP-ABSENT-NOT:  _operand_min,

; HIP-KEEP:       [poseidon] domain error matmul in=f16 acc=f32 p95_relErr=[[HF16:2.927460e-04]] (sumcost=5.403221e+04)
; HIP-KEEP-NEXT:  [poseidon] domain error matmul in=bf16 acc=f32 p95_relErr=[[HBF16:2.360166e-03]] (sumcost=4.316881e+05)
; HIP-FLUSH:      [poseidon] domain error matmul in=f16 min=6.103516e-05 acc=f32 p95_relErr=3.374246e-04 (sumcost=6.217886e+04)
; HIP-FLUSH-NEXT: [poseidon] domain error matmul in=bf16 min=1.175494e-38 acc=f32 p95_relErr=[[HBF16:2.360166e-03]] (sumcost=4.316881e+05)

; HIP-LABEL:      [poseidon] matmul[0] pricing: baseline/MAC=6.433333e+01 executions=7077888
; HIP-NEXT:      [poseidon]   #0 ozaki-ii nm=8 wmma m16n16k16 s8/s32  compCost/MAC=2.010417e+00  rel=3.125000e-02  domainError=[[NM8:8.344801e-09]]
; HIP-NEXT:      [poseidon]   #1 ozaki-ii nm=9 wmma m16n16k16 s8/s32  compCost/MAC=2.261719e+00  rel=3.515625e-02  domainError=5.880977e-10
; HIP-NEXT:      [poseidon]   #2 ozaki-ii nm=10 wmma m16n16k16 s8/s32  compCost/MAC=2.513021e+00  rel=3.906250e-02  domainError=4.149549e-11
; HIP-NEXT:      [poseidon]   #3 ozaki-ii nm=11 wmma m16n16k16 s8/s32  compCost/MAC=2.764323e+00  rel=4.296875e-02  domainError=2.958041e-12
; HIP-NEXT:      [poseidon]   #4 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=3.015625e+00  rel=4.687500e-02  domainError=2.058277e-13
; HIP-NEXT:      [poseidon]   #5 ozaki-ii nm=13 wmma m16n16k16 s8/s32  compCost/MAC=3.266927e+00  rel=5.078125e-02  domainError=1.450488e-14
; HIP-NEXT:      [poseidon]   #6 ozaki-ii nm=14 wmma m16n16k16 s8/s32  compCost/MAC=3.518229e+00  rel=5.468750e-02  domainError=1.058689e-15
; HIP-KEEP-NEXT:  [poseidon]   #7 direct-dispatch cublas f16/f32  compCost/MAC=5.026042e-01  rel=7.812500e-03  domainError=[[HF16]]
; HIP-KEEP-NEXT:  [poseidon]   #8 direct-dispatch cublas bf16/f32  compCost/MAC=7.539062e-01  rel=1.171875e-02  domainError=[[HBF16]]
; HIP-KEEP-NEXT:  [poseidon]   #9 native-dgemm cublas f64/f64  compCost/MAC=3.216667e+01  rel=5.000000e-01  domainError=4.177700e-16
; HIP-FLUSH-NEXT: [poseidon]   #7 direct-dispatch cublas bf16/f32  compCost/MAC=7.539062e-01  rel=1.171875e-02  domainError=[[HBF16]]
; HIP-FLUSH-NEXT: [poseidon]   #8 native-dgemm cublas f64/f64  compCost/MAC=3.216667e+01  rel=5.000000e-01  domainError=4.177700e-16
; HIP-FLUSH:      [poseidon]   not offered: direct-dispatch cublas f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 1.92634e-05, as zero
; HIP-KEEP:       tau=1.000000e-03 {{.*}} rel=[[HF16]]{{$}}
; HIP-FLUSH:      tau=1.000000e-03 {{.*}} rel=[[NM8]]{{$}}
; HIP-KEEP:       Applying solution for matmul[0] -> direct-dispatch cublas f16/f32 (#7)
; HIP-FLUSH:      Applying solution for matmul[0] -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (#0)
; HIP-KEEP:       [poseidon] pin: _ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_:M0=direct-dispatch cublas f16/f32
; HIP-FLUSH:      [poseidon] pin: _ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_:M0=ozaki-ii nm=8 wmma m16n16k16 s8/s32

; HIP-KEY:      "operand min": "1.1754943508222875e-38",
; HIP-KEY-NEXT: "operand min": "6.103515625e-05",
; HIP-KEY-NOT:  operand min

; HIP-PARTIAL:      warning: Poseidon: matrix product 0, 192x1x192: direct-dispatch cublas bf16/f32 is not proposed: this device's cost model carries no measured 'library_direct_operand_min,bf16_f32' row; poseidon-calibrate --gpu <id> --only operand-min measures it
; HIP-PARTIAL:      [poseidon]   #6 ozaki-ii nm=14 wmma m16n16k16 s8/s32  compCost/MAC=3.518229e+00  rel=5.468750e-02  domainError=1.058689e-15
; HIP-PARTIAL-NEXT: [poseidon]   #7 native-dgemm cublas f64/f64  compCost/MAC=3.216667e+01  rel=5.000000e-01  domainError=4.177700e-16
; HIP-PARTIAL:      [poseidon]   not offered: direct-dispatch cublas f16/f32: this device's product keeps f16 operands down to 6.10352e-05 and reads the smallest profiled operand magnitude, 1.92634e-05, as zero
; HIP-PARTIAL:      tau=1.000000e-03 {{.*}} rel=8.344801e-09{{$}}
; HIP-PARTIAL:      Applying solution for matmul[0] -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (#0)
; HIP-PARTIAL:      [poseidon] pin: _ZL11HipKernel2DIZ6reduceiiiiPdS0_S0_iEUliE_EviT_:M0=ozaki-ii nm=8 wmma m16n16k16 s8/s32

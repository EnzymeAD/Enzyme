; DEFINE: %{opt} = %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../../Inputs/library_call/ozaki_stub_host.ll > %t/host.ll
; RUN: %{opt} -poseidon-cache=%t/tau12 -poseidon-compilation-id=tcec -poseidon-tau=1e-12 -o %t/tau12.ll 2>&1 | FileCheck --check-prefix=TAU12 %s
; RUN: cat %t/tau12/launches_*.json | FileCheck --check-prefix=DESC12 %s
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -mllvm -poseidon-early-prune -mllvm -poseidon-cache=%t/tau12 -mllvm -poseidon-compilation-id=tcec -S -emit-llvm -mllvm -poseidon-print -o - 2>&1 | FileCheck --check-prefixes=HOST,HOST12 %s
; RUN: sed 's/@cudaLaunchKernel(/@cudaLaunchKernel_ptsz(/g' %t/host.ll > %t/host_ptsz.ll
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %t/host_ptsz.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -mllvm -poseidon-early-prune -mllvm -poseidon-cache=%t/tau12 -mllvm -poseidon-compilation-id=tcec -S -emit-llvm -mllvm -poseidon-print -o - 2>&1 | FileCheck --check-prefix=PTSZ %s
; RUN: %{opt} -poseidon-cache=%t/tau7 -poseidon-tau=1e-7 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU7 %s
; RUN: %{opt} -poseidon-cache=%t/tau10 -poseidon-tau=1e-10 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU10 %s
; RUN: %{opt} -poseidon-cache=%t/tau16 -poseidon-compilation-id=tcec -poseidon-tau=1e-16 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU16 %s
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -mllvm -poseidon-early-prune -mllvm -poseidon-cache=%t/tau16 -mllvm -poseidon-compilation-id=tcec -S -emit-llvm -mllvm -poseidon-print -o - 2>&1 | FileCheck --check-prefixes=HOST,HOST16 %s
; RUN: %{opt} -poseidon-cache=%t/off -poseidon-tau=1e-12 -poseidon-matmul-library-calls=false -o /dev/null 2>&1 | FileCheck --check-prefix=OFF %s

; The cheapest candidate clearing tau wins, so the Ozaki-II moduli count rises as
; tau tightens; the host stub (Inputs/library_call/ozaki_stub_host.ll) calls
; __poseidon_ozakiii_dgemm with that nm, or __poseidon_native_dgemm.

; With a row for every candidate, the device and host halves of a dispatch and
; an in-kernel raise compile silently; the explanation names each cost row class.
; RUN: cp %cost_models/cm_sm_120_synthetic_tcec.csv %t/rows.csv
; RUN: printf 'wmma_mma_m8n8k4,f64_f64,1024\nlibrary_direct_rel,f32_f32,1\n' >> %t/rows.csv
; RUN: for c in f16_f16 bf16_f32 tf32_f32 f64_f64 tcec_n5_f16_f32 tcec_n2_bf16_f32 tcec_n2_tf32_f32 tcec_n3_tf32_f32 tcec_n5_tf32_f32; do echo "inkernel_rel,$c,1"; done >> %t/rows.csv
; RUN: sed -n 'p; s/^\(inkernel_rel,[a-z0-9_]*\),/\1_m16n16k16_browmajor,/p' %t/rows.csv > %t/complete.csv
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes=poseidon,poseidon-finalize -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-cost-model=%t/complete.csv -poseidon-cache=%t/quiet -poseidon-compilation-id=tcec -poseidon-tau=1e-12 -S -o %t/quiet.ll 2> %t/quiet.err
; RUN: %if nvptx-registered-target %{ count 0 < %t/quiet.err %}
; RUN: ls %t/quiet/launches_*.json
; RUN: %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -mllvm -poseidon-early-prune -mllvm -poseidon-cache=%t/quiet -mllvm -poseidon-compilation-id=tcec -S -emit-llvm -o %t/quiet_host.ll 2>&1 | count 0
; RUN: grep -q __poseidon_ozakiii_dgemm %t/quiet_host.ll
; RUN: %opt %S/../../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes=poseidon,poseidon-finalize -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-matmul-library-calls=false -poseidon-cost-model=%t/complete.csv -poseidon-cache=%t/quiet_raise -poseidon-tau=1e-6 -poseidon-explain=%t/quiet_raise -S -o %t/quiet_raise.ll 2> %t/quiet_raise.err
; RUN: %if nvptx-registered-target %{ count 0 < %t/quiet_raise.err %}
; RUN: grep -q llvm.nvvm.wmma %t/quiet_raise.ll
; RUN: FileCheck --check-prefix=CLASS %s < %t/quiet_raise/matmul_body.explain.json

; CLASS: "id": "M0_1",
; CLASS-NEXT: "inkernel_class": "f16_f32",
; CLASS: "id": "M0_5",
; CLASS-NEXT: "inkernel_class": "tcec_n2_f16_f32",

; TAU12: [poseidon] matmul[0] pricing: baseline/MAC=5.120000e+02 executions=8192
; TAU12-NEXT: #0 wmma m16n16k16 f16/f32  compCost/MAC=3.200000e+01  rel=6.250000e-02  domainError=3.137966e-04
; TAU12-NEXT: #1 tcec n=2 wmma m16n16k16 f16/f32  compCost/MAC=4.000000e+01  rel=7.812500e-02  domainError=1.572576e-07
; TAU12-NEXT: #2 tcec n=3 wmma m16n16k16 f16/f32  compCost/MAC=4.800000e+01  rel=9.375000e-02  domainError=5.874975e-08
; TAU12-NEXT: #3 ozaki-ii nm=8 wmma m16n16k16 s8/s32  compCost/MAC=1.600000e+01  rel=3.125000e-02  domainError=7.986923e-09
; TAU12-NEXT: #4 ozaki-ii nm=9 wmma m16n16k16 s8/s32  compCost/MAC=1.800000e+01  rel=3.515625e-02  domainError=5.811304e-10
; TAU12-NEXT: #5 ozaki-ii nm=10 wmma m16n16k16 s8/s32  compCost/MAC=2.000000e+01  rel=3.906250e-02  domainError=4.205447e-11
; TAU12-NEXT: #6 ozaki-ii nm=11 wmma m16n16k16 s8/s32  compCost/MAC=2.200000e+01  rel=4.296875e-02  domainError=2.759331e-12
; TAU12-NEXT: #7 ozaki-ii nm=12 wmma m16n16k16 s8/s32  compCost/MAC=2.400000e+01  rel=4.687500e-02  domainError=1.840864e-13
; TAU12-NEXT: #8 ozaki-ii nm=13 wmma m16n16k16 s8/s32  compCost/MAC=2.600000e+01  rel=5.078125e-02  domainError=1.220929e-14
; TAU12-NEXT: #9 ozaki-ii nm=14 wmma m16n16k16 s8/s32  compCost/MAC=2.800000e+01  rel=5.468750e-02  domainError=8.567043e-16
; TAU12-NEXT: #10 direct-dispatch cublas f16/f32  compCost/MAC=4.000000e+00  rel=7.812500e-03  domainError=3.137966e-04
; TAU12-NEXT: #11 direct-dispatch cublas bf16/f32  compCost/MAC=6.000000e+00  rel=1.171875e-02  domainError=2.340575e-03
; TAU12-NEXT: #12 direct-dispatch cublas tf32/f32  compCost/MAC=8.000000e+00  rel=1.562500e-02  domainError=3.137466e-04
; TAU12-NEXT: #13 native-dgemm cublas f64/f64  compCost/MAC=2.560000e+02  rel=5.000000e-01  domainError=0.000000e+00
; TAU12: tau=1.000000e-12 {{.*}} rel=1.840864e-13{{$}}
; TAU12: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#7)
; TAU12: [poseidon] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; DESC12: _Z13matmul_kernelPKdS0_PdS1_ 2 0 1 1 12 0 0 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00

; HOST12: [poseidon] replaced the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_ozakiii_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; HOST16: [poseidon] replaced the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_native_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; HOST-LABEL: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; HOST: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; HOST: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; HOST: %[[CS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 2|i64 16}}{{$}}
; HOST-NEXT: %[[CL:[0-9]+]] = load ptr, ptr %[[CS]], align 8
; HOST-NEXT: %[[C:[0-9]+]] = load ptr, ptr %[[CL]], align 8
; HOST-NEXT: %[[AS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 0|i64 0}}{{$}}
; HOST-NEXT: %[[AL:[0-9]+]] = load ptr, ptr %[[AS]], align 8
; HOST-NEXT: %[[A:[0-9]+]] = load ptr, ptr %[[AL]], align 8
; HOST-NEXT: %[[BS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; HOST-NEXT: %[[BL:[0-9]+]] = load ptr, ptr %[[BS]], align 8
; HOST-NEXT: %[[B:[0-9]+]] = load ptr, ptr %[[BL]], align 8
; HOST12-NEXT: call void @__poseidon_ozakiii_dgemm(ptr %[[C]], ptr %[[A]], ptr %[[B]], i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]], i32 12)
; HOST16-NEXT: call void @__poseidon_native_dgemm(ptr %[[C]], ptr %[[A]], ptr %[[B]], i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]])
; HOST-NEXT: br label
; HOST-NOT: cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernel

; A per-thread-stream launch's null stream is (cudaStream_t)2 to the library.
; PTSZ: [poseidon] replaced the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_ozakiii_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; PTSZ-LABEL: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; PTSZ: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; PTSZ: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; PTSZ: %[[ISNULL:[0-9]+]] = icmp eq ptr %[[STREAM]], null
; PTSZ-NEXT: %[[HANDLE:[0-9]+]] = select i1 %[[ISNULL]], ptr inttoptr (i64 2 to ptr), ptr %[[STREAM]]
; PTSZ-NEXT: call void @__poseidon_ozakiii_dgemm(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[HANDLE]], i32 12)
; PTSZ-NEXT: br label
; PTSZ-NOT: cudaLaunchKernel_ptsz(ptr {{.*}}@_Z28__device_stub__matmul_kernel

; TAU7: tau=1.000000e-07 {{.*}} rel=7.986923e-09{{$}}
; TAU7: Applying solution for matmul[0] -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (#3)
; TAU7: [poseidon] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=8 (standalone)

; TAU10: tau=1.000000e-10 {{.*}} rel=4.205447e-11{{$}}
; TAU10: Applying solution for matmul[0] -> ozaki-ii nm=10 wmma m16n16k16 s8/s32 (#5)
; TAU10: [poseidon] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=10 (standalone)

; TAU16: tau=1.000000e-16 {{.*}} rel=0.000000e+00{{$}}
; TAU16: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#13)
; TAU16: [poseidon] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=0 (standalone)

; OFF-NOT: ozaki-ii
; OFF: tau=1.000000e-12 {{.*}} selected cost=0.000000e+00 accCost=0.000000e+00 rel=0.000000e+00{{$}}
; OFF-NOT: Applying solution for matmul
; OFF-NOT: wrote descriptor

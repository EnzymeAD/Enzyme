; DEFINE: %{opt} = %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon \
; DEFINE:   -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" \
; DEFINE:   -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false \
; DEFINE:   -poseidon-enable-pt=false -poseidon-raise-wmma \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; DEFINE:   -poseidon-print -S
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/host.ll
; RUN: %{opt} -poseidon-cache=%t/tau12 -poseidon-unit=tcec -poseidon-tau=1e-12 -o %t/tau12.ll 2>&1 | FileCheck --check-prefix=TAU12 %s
; RUN: cat %t/tau12/launches_*.json | FileCheck --check-prefix=DESC12 %s
; RUN: %clang -O0 -x ir -fno-discard-value-names %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/tau12 -mllvm -poseidon-unit=tcec -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST12 %s
; RUN: sed 's/@cudaLaunchKernel(/@cudaLaunchKernel_ptsz(/g' %t/host.ll > %t/host_ptsz.ll
; RUN: %clang -O0 -x ir -fno-discard-value-names %t/host_ptsz.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/tau12 -mllvm -poseidon-unit=tcec -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=PTSZ %s
; RUN: %{opt} -poseidon-cache=%t/tau7 -poseidon-tau=1e-7 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU7 %s
; RUN: %{opt} -poseidon-cache=%t/tau10 -poseidon-tau=1e-10 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU10 %s
; RUN: %{opt} -poseidon-cache=%t/tau16 -poseidon-unit=tcec -poseidon-tau=1e-16 -o /dev/null 2>&1 | FileCheck --check-prefix=TAU16 %s
; RUN: %clang -O0 -x ir -fno-discard-value-names %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/tau16 -mllvm -poseidon-unit=tcec -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST16 %s
; RUN: %{opt} -poseidon-cache=%t/off -poseidon-tau=1e-12 -poseidon-ozaki-host-dispatch=false -o /dev/null 2>&1 \
; RUN:   | FileCheck --check-prefix=OFF %s

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
; TAU12: Poseidon error-budget: matmul -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (domain err 1.840864e-13 <= 1.000000e-12 at confidence 9.500000e-01)
; TAU12: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#7)
; TAU12: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; DESC12: _Z13matmul_kernelPKdS0_PdS1_ 2 0 1 1 12 0 0 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00

; HOST12: [ozaki-host-dispatch] replaced the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_ozaki_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; HOST12-LABEL: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST12: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; HOST12: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; HOST12: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; HOST12: %[[CS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 2|i64 16}}{{$}}
; HOST12-NEXT: %[[CL:[0-9]+]] = load ptr, ptr %[[CS]], align 8
; HOST12-NEXT: %[[C:[0-9]+]] = load ptr, ptr %[[CL]], align 8
; HOST12-NEXT: %[[AS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 0|i64 0}}{{$}}
; HOST12-NEXT: %[[AL:[0-9]+]] = load ptr, ptr %[[AS]], align 8
; HOST12-NEXT: %[[A:[0-9]+]] = load ptr, ptr %[[AL]], align 8
; HOST12-NEXT: %[[BS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; HOST12-NEXT: %[[BL:[0-9]+]] = load ptr, ptr %[[BS]], align 8
; HOST12-NEXT: %[[B:[0-9]+]] = load ptr, ptr %[[BL]], align 8
; HOST12-NEXT: call void @__poseidon_ozaki_dgemm(ptr %[[C]], ptr %[[A]], ptr %[[B]], i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]], i32 12)
; HOST12-NEXT: br label
; HOST12-NOT: cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernel

; A per-thread-stream launch's null stream is its thread's default stream,
; (cudaStream_t)2 to a library compiled without per-thread streams.
; PTSZ: [ozaki-host-dispatch] replaced the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_ozaki_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; PTSZ-LABEL: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; PTSZ: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; PTSZ: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; PTSZ: %[[ISNULL:[0-9]+]] = icmp eq ptr %[[STREAM]], null
; PTSZ-NEXT: %[[HANDLE:[0-9]+]] = select i1 %[[ISNULL]], ptr inttoptr (i64 2 to ptr), ptr %[[STREAM]]
; PTSZ-NEXT: call void @__poseidon_ozaki_dgemm(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[HANDLE]], i32 12)
; PTSZ-NEXT: br label
; PTSZ-NOT: cudaLaunchKernel_ptsz(ptr {{.*}}@_Z28__device_stub__matmul_kernel

; TAU7: Poseidon error-budget: matmul -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (domain err 7.986923e-09 <= 1.000000e-07 at confidence 9.500000e-01)
; TAU7: Applying solution for matmul[0] -> ozaki-ii nm=8 wmma m16n16k16 s8/s32 (#3)
; TAU7: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=8 (standalone)

; TAU10: Poseidon error-budget: matmul -> ozaki-ii nm=10 wmma m16n16k16 s8/s32 (domain err 4.205447e-11 <= 1.000000e-10 at confidence 9.500000e-01)
; TAU10: Applying solution for matmul[0] -> ozaki-ii nm=10 wmma m16n16k16 s8/s32 (#5)
; TAU10: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=10 (standalone)

; TAU16: Poseidon error-budget: matmul -> native-dgemm cublas f64/f64 (domain err 0.000000e+00 <= 1.000000e-16 at confidence 9.500000e-01)
; TAU16: Applying solution for matmul[0] -> native-dgemm cublas f64/f64 (#13)
; TAU16: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=0 (standalone)

; HOST16: [ozaki-host-dispatch] replaced the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_native_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; HOST16-LABEL: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST16: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; HOST16: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; HOST16: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; HOST16: %[[CS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 2|i64 16}}{{$}}
; HOST16-NEXT: %[[CL:[0-9]+]] = load ptr, ptr %[[CS]], align 8
; HOST16-NEXT: %[[C:[0-9]+]] = load ptr, ptr %[[CL]], align 8
; HOST16-NEXT: %[[AS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 0|i64 0}}{{$}}
; HOST16-NEXT: %[[AL:[0-9]+]] = load ptr, ptr %[[AS]], align 8
; HOST16-NEXT: %[[A:[0-9]+]] = load ptr, ptr %[[AL]], align 8
; HOST16-NEXT: %[[BS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; HOST16-NEXT: %[[BL:[0-9]+]] = load ptr, ptr %[[BS]], align 8
; HOST16-NEXT: %[[B:[0-9]+]] = load ptr, ptr %[[BL]], align 8
; HOST16-NEXT: call void @__poseidon_native_dgemm(ptr %[[C]], ptr %[[A]], ptr %[[B]], i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]])
; HOST16-NEXT: br label
; HOST16-NOT: cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernel

; OFF-NOT: ozaki-ii
; OFF: Poseidon error-budget: matmul -> baseline F64 (no rewrite clears 1.000000e-12 at confidence 9.500000e-01 faster than F64)
; OFF-NOT: Applying solution for matmul
; OFF-NOT: wrote descriptor

; Ozaki-II host dispatch selected by a tolerance: the scalar FP64 product of
; raise_scalar_tcec_nvptx.ll (Inputs/stale_cache/matmul_device.ll, same
; profile) priced against the in-kernel raises and the host-dispatch families
; of cm_sm_120_synthetic_dispatch.csv. The cheapest candidate whose modelled
; domain error clears tau wins, so the Ozaki-II moduli count rises as tau
; tightens. The device compile writes the dispatch descriptor into the unit's
; manifest (tenth field 2 = Ozaki-II, sixth field = nm), and the host compile
; of the kernel's launch stub (Inputs/ozaki_dispatch/ozaki_stub_host.ll, clang
; -x cuda --cuda-host-only -O0 -S -emit-llvm of
;   __global__ void matmul_kernel(const double *A, const double *B, double *D,
;                                 double *dD) {}
;   void launch(const double *A, const double *B, double *D, double *dD) {
;     matmul_kernel<<<1, dim3(16, 16)>>>(A, B, D, dD);
;   }
; renamed to this module's source and compiled as the same unit) calls
; __poseidon_ozaki_dgemm with that nm, or __poseidon_native_dgemm for the
; native DGEMM.

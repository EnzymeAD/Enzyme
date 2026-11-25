; DEFINE: %{clang} = %clang -O0 -x ir -fno-discard-value-names %loadClangPoseidon \
; DEFINE:   -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec \
; DEFINE:   -mllvm -poseidon-cache=%t/c -S -emit-llvm -o -
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/same.ll
; RUN: sed 's/ozaki_stub\.cu/other_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/other.ll
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/c -poseidon-unit=u -S -o /dev/null 2>&1 | FileCheck --check-prefix=DEV %s
; RUN: %{clang} %t/same.ll -mllvm -poseidon-unit=u 2>&1 | FileCheck --check-prefix=SAME %s
; RUN: %{clang} %t/other.ll -mllvm -poseidon-unit=u 2>&1 | FileCheck --check-prefixes=OTHER,KEPT --implicit-check-not=__poseidon_ozaki_dgemm --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s
; RUN: %{clang} %t/same.ll -mllvm -poseidon-unit=v 2>&1 | FileCheck --check-prefixes=OTHERUNIT,KEPT --implicit-check-not=__poseidon_ozaki_dgemm --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s

; The device compile of tcec_matmul.cu as unit u leaves a descriptor for
; matmul_kernel in the cache. The host compile of tcec_matmul.cu as unit u
; applies it; the host compile of other_matmul.cu as unit u, whose kernel has
; the same mangled name, must not, and neither must the host compile of
; tcec_matmul.cu as unit v. The halves are compiled with different options, so
; every compile names its unit.

; DEV: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; SAME: [ozaki-host-dispatch] replaced the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_ozaki_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; SAME-LABEL: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; SAME: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; SAME: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[STREAMP:[0-9]+]])
; SAME: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; SAME: %[[CS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 2|i64 16}}{{$}}
; SAME-NEXT: %[[CL:[0-9]+]] = load ptr, ptr %[[CS]], align 8
; SAME-NEXT: %[[C:[0-9]+]] = load ptr, ptr %[[CL]], align 8
; SAME-NEXT: %[[AS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 0|i64 0}}{{$}}
; SAME-NEXT: %[[AL:[0-9]+]] = load ptr, ptr %[[AS]], align 8
; SAME-NEXT: %[[A:[0-9]+]] = load ptr, ptr %[[AL]], align 8
; SAME-NEXT: %[[BS:[0-9]+]] = getelementptr inbounds {{ptr|i8}}, ptr %[[ARGS]], {{i32 1|i64 8}}{{$}}
; SAME-NEXT: %[[BL:[0-9]+]] = load ptr, ptr %[[BS]], align 8
; SAME-NEXT: %[[B:[0-9]+]] = load ptr, ptr %[[BL]], align 8
; SAME-NEXT: call void @__poseidon_ozaki_dgemm(ptr %[[C]], ptr %[[A]], ptr %[[B]], i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr %[[STREAM]], i32 12)
; SAME-NEXT: br label
; SAME-NOT: call {{.*}}@cudaLaunchKernel(

; OTHER: source_filename = "other_matmul.cu"
; OTHERUNIT: source_filename = "tcec_matmul.cu"

; KEPT-LABEL: define dso_local void @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; KEPT: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; KEPT: call i32 @__cudaPopCallConfiguration(ptr %{{[0-9]+}}, ptr %{{[0-9]+}}, ptr %[[SHMEMP:[0-9]+]], ptr %[[STREAMP:[0-9]+]])
; KEPT-NEXT: %[[SHMEM:[0-9]+]] = load i64, ptr %[[SHMEMP]], align 8
; KEPT-NEXT: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; KEPT: call noundef i32 @cudaLaunchKernel(ptr noundef @_Z28__device_stub__matmul_kernelPKdS0_PdS1_, {{.*}}, ptr noundef %[[ARGS]], i64 noundef %[[SHMEM]], ptr noundef %[[STREAM]])


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

; SAME: [ozaki-host-dispatch] replaced stub body _Z28__device_stub__matmul_kernelPKdS0_PdS1_ -> __poseidon_ozaki_dgemm (C=arg2 A=arg0 B=arg1 16x16x16){{$}}
; SAME: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(ptr noundef [[A:%[0-9]+]], ptr noundef [[B:%[0-9]+]], ptr noundef [[C:%[0-9]+]], ptr noundef %{{[0-9]+}})
; SAME-NEXT: entry:
; SAME-NEXT: call void @__poseidon_ozaki_dgemm(ptr [[C]], ptr [[A]], ptr [[B]], i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr null, i32 12)
; SAME-NEXT: ret void
; SAME-NEXT: }
; SAME-NOT: call {{.*}}@cudaLaunchKernel(

; OTHER: source_filename = "other_matmul.cu"
; OTHERUNIT: source_filename = "tcec_matmul.cu"

; KEPT-LABEL: define dso_local void @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; KEPT: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; KEPT: call i32 @__cudaPopCallConfiguration(ptr %[[GRID:[0-9]+]], ptr %[[BLOCK:[0-9]+]], ptr %[[SHMEMP:[0-9]+]], ptr %[[STREAMP:[0-9]+]])
; KEPT-NEXT: %[[SHMEM:[0-9]+]] = load i64, ptr %[[SHMEMP]], align 8
; KEPT-NEXT: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; KEPT-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[GC:[0-9]+]], ptr align 8 %[[GRID]], i64 12, i1 false)
; KEPT-NEXT: %[[GXYP:[0-9]+]] = getelementptr {{.*}}ptr %[[GC]], i32 0, i32 0
; KEPT-NEXT: %[[GXY:[0-9]+]] = load i64, ptr %[[GXYP]], align 8
; KEPT-NEXT: %[[GZP:[0-9]+]] = getelementptr {{.*}}ptr %[[GC]], i32 0, i32 1
; KEPT-NEXT: %[[GZ:[0-9]+]] = load i32, ptr %[[GZP]], align 8
; KEPT-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[BC:[0-9]+]], ptr align 8 %[[BLOCK]], i64 12, i1 false)
; KEPT-NEXT: %[[BXYP:[0-9]+]] = getelementptr {{.*}}ptr %[[BC]], i32 0, i32 0
; KEPT-NEXT: %[[BXY:[0-9]+]] = load i64, ptr %[[BXYP]], align 8
; KEPT-NEXT: %[[BZP:[0-9]+]] = getelementptr {{.*}}ptr %[[BC]], i32 0, i32 1
; KEPT-NEXT: %[[BZ:[0-9]+]] = load i32, ptr %[[BZP]], align 8
; KEPT-NEXT: %{{[0-9]+}} = call noundef i32 @cudaLaunchKernel(ptr noundef @_Z28__device_stub__matmul_kernelPKdS0_PdS1_, i64 %[[GXY]], i32 %[[GZ]], i64 %[[BXY]], i32 %[[BZ]], ptr noundef %[[ARGS]], i64 noundef %[[SHMEM]], ptr noundef %[[STREAM]])


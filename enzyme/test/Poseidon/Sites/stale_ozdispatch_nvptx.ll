; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/host.ll
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/c -poseidon-unit=tcec -S -o /dev/null 2>&1 | FileCheck --check-prefix=DEV1 %s
; RUN: %clang -O0 -x ir -fno-discard-value-names %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=tcec -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST1 %s
; RUN: %opt %S/../Inputs/stale_cache/matmul_device.ll %loadPoseidon -passes="poseidon,poseidon-finalize,function(mem2reg,instsimplify,simplifycfg)" -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-enable-herbie=false -poseidon-enable-pt=false -poseidon-raise-wmma=false -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-tau=1e-12 -poseidon-cache=%t/c -poseidon-unit=tcec -S -o /dev/null 2>&1 | FileCheck --check-prefix=DEV2 %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/c -mllvm -poseidon-unit=tcec -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST2 --implicit-check-not=__poseidon_ozaki_dgemm --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s

; A device compile that dispatches the product writes its descriptor into the
; manifest of the unit; a later device compile of the same kernel and unit into
; the same cache that dispatches nothing (raising off) must leave the host
; compile's launch stub unchanged. The halves are compiled with different
; options, so all four compiles name the unit.

; DEV1: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; HOST1: [ozaki-host-dispatch] replaced stub body _Z28__device_stub__matmul_kernelPKdS0_PdS1_ -> __poseidon_ozaki_dgemm (C=arg2 A=arg0 B=arg1 16x16x16){{$}}
; HOST1: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(ptr noundef [[A:%[0-9]+]], ptr noundef [[B:%[0-9]+]], ptr noundef [[C:%[0-9]+]], ptr noundef %{{[0-9]+}})
; HOST1-NEXT: entry:
; HOST1-NEXT: call void @__poseidon_ozaki_dgemm(ptr [[C]], ptr [[A]], ptr [[B]], i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr null, i32 12)
; HOST1-NEXT: ret void
; HOST1-NEXT: }
; HOST1-NOT: call {{.*}}@cudaLaunchKernel(

; DEV2-NOT: wrote descriptor
; DEV2: [poseidon] no rewrite applied for preprocess_matmul_body; calling original body matmul_body (bit-transparent no-op){{$}}
; DEV2-NOT: wrote descriptor

; HOST2-LABEL: define dso_local void @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST2: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; HOST2: call i32 @__cudaPopCallConfiguration(ptr %[[GRID:[0-9]+]], ptr %[[BLOCK:[0-9]+]], ptr %[[SHMEMP:[0-9]+]], ptr %[[STREAMP:[0-9]+]])
; HOST2-NEXT: %[[SHMEM:[0-9]+]] = load i64, ptr %[[SHMEMP]], align 8
; HOST2-NEXT: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; HOST2-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[GC:[0-9]+]], ptr align 8 %[[GRID]], i64 12, i1 false)
; HOST2-NEXT: %[[GXYP:[0-9]+]] = getelementptr {{.*}}ptr %[[GC]], i32 0, i32 0
; HOST2-NEXT: %[[GXY:[0-9]+]] = load i64, ptr %[[GXYP]], align 8
; HOST2-NEXT: %[[GZP:[0-9]+]] = getelementptr {{.*}}ptr %[[GC]], i32 0, i32 1
; HOST2-NEXT: %[[GZ:[0-9]+]] = load i32, ptr %[[GZP]], align 8
; HOST2-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[BC:[0-9]+]], ptr align 8 %[[BLOCK]], i64 12, i1 false)
; HOST2-NEXT: %[[BXYP:[0-9]+]] = getelementptr {{.*}}ptr %[[BC]], i32 0, i32 0
; HOST2-NEXT: %[[BXY:[0-9]+]] = load i64, ptr %[[BXYP]], align 8
; HOST2-NEXT: %[[BZP:[0-9]+]] = getelementptr {{.*}}ptr %[[BC]], i32 0, i32 1
; HOST2-NEXT: %[[BZ:[0-9]+]] = load i32, ptr %[[BZP]], align 8
; HOST2-NEXT: %{{[0-9]+}} = call noundef i32 @cudaLaunchKernel(ptr noundef @_Z28__device_stub__matmul_kernelPKdS0_PdS1_, i64 %[[GXY]], i32 %[[GZ]], i64 %[[BXY]], i32 %[[BZ]], ptr noundef %[[ARGS]], i64 noundef %[[SHMEM]], ptr noundef %[[STREAM]])

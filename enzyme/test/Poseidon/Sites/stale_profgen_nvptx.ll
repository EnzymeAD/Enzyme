; REQUIRES: enzyme, nvptx-device-profiler
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/host.ll
; RUN: grep -v llvm.global.annotations %S/../Inputs/stale_cache/matmul_annotated_device.ll > %t/plain.ll
; RUN: %opt %S/../Inputs/stale_cache/matmul_annotated_device.ll %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,globaldce" -enzyme-preopt=false -poseidon-profile-generate -poseidon-cache=%t/c -poseidon-print -S -o /dev/null 2>&1 | FileCheck --check-prefix=DEV1 %s
; RUN: cat %t/c/launches_*.json | FileCheck --check-prefix=MANIFEST1 %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST1 --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s
; RUN: %opt %t/plain.ll %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,globaldce" -enzyme-preopt=false -poseidon-profile-generate -poseidon-cache=%t/c -poseidon-print -S -o /dev/null 2>&1 | FileCheck --allow-empty --check-prefix=DEV2 %s
; RUN: ls %t/c | FileCheck --allow-empty --check-prefix=MANIFEST2 %s
; RUN: %clang -O0 -x ir %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/c -mllvm -poseidon-print -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=HOST2 --implicit-check-not=__poseidon_launch_profiled --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s

; A profile-generate device compile of an annotated kernel writes its
; descriptor into the manifest of the unit; after the annotation is removed, a
; device compile into the same cache removes the manifest, and the host compile
; must launch the kernel as written instead of with shadow buffers it no longer
; takes.

; DEV1: [poseidon-profgen] _Z13matmul_kernelPKdS0_PdS1_: site 0, 4 shadow buffer(s), output(s) 2

; HOST1: [poseidon-profgen] launch of _Z13matmul_kernelPKdS0_PdS1_ now allocates and seeds 4 shadow buffer(s)
; MANIFEST1:      "host dispatch": [],
; MANIFEST1-NEXT: "profile generation": [
; MANIFEST1-NEXT:   "_Z13matmul_kernelPKdS0_PdS1_ 0 4 4 0 1 2 3 0 0 8 0"
; MANIFEST1-NEXT: ],

; HOST1: @poseidon.ptr._Z13matmul_kernelPKdS0_PdS1_ = private unnamed_addr constant [4 x i32] [i32 0, i32 1, i32 2, i32 3]
; HOST1-NEXT: @poseidon.off._Z13matmul_kernelPKdS0_PdS1_ = private unnamed_addr constant [4 x i32] zeroinitializer
; HOST1-NEXT: @poseidon.seed._Z13matmul_kernelPKdS0_PdS1_ = private unnamed_addr constant [4 x i32] [i32 0, i32 0, i32 8, i32 0]
; HOST1-LABEL: define dso_local void @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST1: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; HOST1: call i32 @__cudaPopCallConfiguration(ptr %[[GRID:[0-9]+]], ptr %[[BLOCK:[0-9]+]], ptr %[[SHMEMP:[0-9]+]], ptr %[[STREAMP:[0-9]+]])
; HOST1-NEXT: %[[SHMEM:[0-9]+]] = load i64, ptr %[[SHMEMP]], align 8
; HOST1-NEXT: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; HOST1: call void @__poseidon_launch_profiled(ptr @_Z28__device_stub__matmul_kernelPKdS0_PdS1_, i32 0, ptr %[[GRID]], ptr %[[BLOCK]], i64 %[[SHMEM]], ptr %[[STREAM]], i32 4, ptr %[[ARGS]], i32 4, ptr @poseidon.ptr._Z13matmul_kernelPKdS0_PdS1_, ptr @poseidon.off._Z13matmul_kernelPKdS0_PdS1_, ptr @poseidon.seed._Z13matmul_kernelPKdS0_PdS1_)

; DEV2-NOT: [poseidon-profgen]

; MANIFEST2-NOT: launches_

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

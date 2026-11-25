; DEFINE: %{dev} = %S/../Inputs/stale_cache/matmul_device.ll
; DEFINE: %{pass} = %loadPoseidon -passes="poseidon,poseidon-finalize"
; DEFINE: %{flags} = -poseidon-profile-use=%S/../Inputs/raise_tcec \
; DEFINE:   -poseidon-enable-herbie=false -poseidon-enable-pt=false \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv \
; DEFINE:   -poseidon-tau=1e-12 -S -o /dev/null
; RUN: rm -rf %t && mkdir %t

; The launch manifest a device compile writes for the products it dispatches
; to the host, and what the host compile of the same unit does with it.

; The device compile of tcec_matmul.cu as unit u writes the manifest of the
; unit, named after the source, with the descriptor of the dispatched product.
; A later device compile of the unit that dispatches nothing (raising off)
; removes it, and a device compile of a module without a site, under profile
; use and under profile generation, creates neither a manifest nor the cache.
; A host compile whose unit has no manifest in the cache, under profile use and
; under profile generation, says nothing, keeps the launch and creates no
; cache.
; RUN: mkdir %t/write
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/write/host.ll
; RUN: %opt %{dev} %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/write/c -poseidon-unit=u 2>&1 | FileCheck --check-prefix=WRITE-WROTE %s
; RUN: ls %t/write/c | FileCheck --check-prefix=WRITE-FILES --implicit-check-not=launches_ %s
; RUN: cat %t/write/c/launches_*.json | FileCheck --check-prefix=WRITE-FULL %s
; RUN: %opt %{dev} %{pass} -poseidon-raise-wmma=false %{flags} -poseidon-cache=%t/write/c -poseidon-unit=u 2>&1 | FileCheck --check-prefix=WRITE-NONE %s
; RUN: ls %t/write/c | FileCheck --allow-empty --check-prefix=WRITE-AFTER %s
; RUN: grep -v __poseidon_fp_optimize %{dev} > %t/write/siteless.ll
; RUN: %opt %t/write/siteless.ll %{pass} -poseidon-profile-use=%S/../Inputs/raise_tcec -poseidon-raise-wmma -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -poseidon-cache=%t/write/fresh -S -o /dev/null
; RUN: not ls %t/write/fresh
; RUN: %opt %t/write/siteless.ll %{pass} -poseidon-profile-generate -poseidon-cache=%t/write/freshgen -S -o /dev/null
; RUN: not ls %t/write/freshgen
; RUN: %clang -O0 -Wno-override-module -x ir %t/write/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/write/none -mllvm -poseidon-unit=u -S -emit-llvm -o %t/write/use.ll 2> %t/write/use.err
; RUN: count 0 < %t/write/use.err
; RUN: FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s < %t/write/use.ll
; RUN: %clang -O0 -Wno-override-module -x ir %t/write/host.ll %loadClangPoseidon -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/write/none -mllvm -poseidon-print -mllvm -poseidon-unit=u -S -emit-llvm -o %t/write/gen.ll 2> %t/write/gen.err
; RUN: count 0 < %t/write/gen.err
; RUN: FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s < %t/write/gen.ll
; RUN: not ls %t/write/none

; WRITE-WROTE: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; WRITE-FILES: {{^}}launches_tcec_matmul.cu_{{[0-9A-F]+}}.json{{$}}

; WRITE-FULL: "key": {
; WRITE-FULL-NEXT: "format": "1",
; WRITE-FULL-NEXT: "source file": "tcec_matmul.cu",
; WRITE-FULL-NEXT: "unit": "u"
; WRITE-FULL-NEXT: },
; WRITE-FULL-NEXT: "value": {
; WRITE-FULL-NEXT: "host dispatch": [
; WRITE-FULL-NEXT: "_Z13matmul_kernelPKdS0_PdS1_ 2 0 1 16 1 12 16 16 16 16 16 16 0 0 0 0"
; WRITE-FULL-NEXT: ],
; WRITE-FULL-NEXT: "profile generation": []

; WRITE-NONE-NOT: wrote descriptor
; WRITE-NONE: [poseidon] no rewrite applied for preprocess_matmul_body; calling original body matmul_body (bit-transparent no-op){{$}}
; WRITE-NONE-NOT: wrote descriptor

; WRITE-AFTER-NOT: launches_

; LAUNCH-LABEL: define dso_local void @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; LAUNCH: %[[ARGS:[0-9]+]] = alloca ptr, i64 4
; LAUNCH: call i32 @__cudaPopCallConfiguration(ptr %[[GRID:[0-9]+]], ptr %[[BLOCK:[0-9]+]], ptr %[[SHMEMP:[0-9]+]], ptr %[[STREAMP:[0-9]+]])
; LAUNCH-NEXT: %[[SHMEM:[0-9]+]] = load i64, ptr %[[SHMEMP]], align 8
; LAUNCH-NEXT: %[[STREAM:[0-9]+]] = load ptr, ptr %[[STREAMP]], align 8
; LAUNCH-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[GC:[0-9]+]], ptr align 8 %[[GRID]], i64 12, i1 false)
; LAUNCH-NEXT: %[[GXYP:[0-9]+]] = getelementptr {{.*}}ptr %[[GC]], i32 0, i32 0
; LAUNCH-NEXT: %[[GXY:[0-9]+]] = load i64, ptr %[[GXYP]], align 8
; LAUNCH-NEXT: %[[GZP:[0-9]+]] = getelementptr {{.*}}ptr %[[GC]], i32 0, i32 1
; LAUNCH-NEXT: %[[GZ:[0-9]+]] = load i32, ptr %[[GZP]], align 8
; LAUNCH-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[BC:[0-9]+]], ptr align 8 %[[BLOCK]], i64 12, i1 false)
; LAUNCH-NEXT: %[[BXYP:[0-9]+]] = getelementptr {{.*}}ptr %[[BC]], i32 0, i32 0
; LAUNCH-NEXT: %[[BXY:[0-9]+]] = load i64, ptr %[[BXYP]], align 8
; LAUNCH-NEXT: %[[BZP:[0-9]+]] = getelementptr {{.*}}ptr %[[BC]], i32 0, i32 1
; LAUNCH-NEXT: %[[BZ:[0-9]+]] = load i32, ptr %[[BZP]], align 8
; LAUNCH-NEXT: %{{[0-9]+}} = call noundef i32 @cudaLaunchKernel(ptr noundef @_Z28__device_stub__matmul_kernelPKdS0_PdS1_, i64 %[[GXY]], i32 %[[GZ]], i64 %[[BXY]], i32 %[[BZ]], ptr noundef %[[ARGS]], i64 noundef %[[SHMEM]], ptr noundef %[[STREAM]])

; Without -poseidon-unit, the two halves of a compile name their unit by the
; values of every Poseidon flag, so a host compile with the device compile's
; flags takes its manifest and one with another tolerance does not; and, ahead
; of that, by HIP's compilation-unit id, which ties halves compiled with
; different flags and keeps apart halves with different ids. A -poseidon-unit
; both halves are given takes precedence over the id.
; RUN: mkdir %t/unit
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../Inputs/ozaki_dispatch/ozaki_stub_host.ll > %t/unit/host.ll
; RUN: sed '/^target triple/a @__hip_cuid_0123abcd = global i8 0' %{dev} > %t/unit/dev_cuid.ll
; RUN: sed '/^target triple/a @__hip_cuid_0123abcd = global i8 0' %t/unit/host.ll > %t/unit/host_cuid.ll
; RUN: sed '/^target triple/a @__hip_cuid_89abcdef = global i8 0' %t/unit/host.ll > %t/unit/host_other_cuid.ll
; RUN: %opt %{dev} %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/unit/options 2>&1 | FileCheck --check-prefix=UNIT-WROTE %s
; RUN: %clang -O0 -x ir -fno-discard-value-names %t/unit/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-enable-herbie=false -mllvm -poseidon-enable-pt=false -mllvm -poseidon-raise-wmma -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -mllvm -poseidon-tau=1e-12 -mllvm -poseidon-cache=%t/unit/options -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=UNIT-DISPATCH %s
; RUN: %clang -O0 -x ir %t/unit/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-enable-herbie=false -mllvm -poseidon-enable-pt=false -mllvm -poseidon-raise-wmma -mllvm -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_dispatch.csv -mllvm -poseidon-tau=1e-7 -mllvm -poseidon-cache=%t/unit/options -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon_ozaki_dgemm --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s
; RUN: %opt %t/unit/dev_cuid.ll %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/unit/cuid 2>&1 | FileCheck --check-prefix=UNIT-WROTE %s
; RUN: cat %t/unit/cuid/launches_*.json | FileCheck --check-prefix=UNIT-CUID %s
; RUN: %clang -O0 -x ir -fno-discard-value-names %t/unit/host_cuid.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/unit/cuid -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=UNIT-DISPATCH %s
; RUN: %clang -O0 -x ir %t/unit/host_other_cuid.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/unit/cuid -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon_ozaki_dgemm --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s
; RUN: %opt %t/unit/dev_cuid.ll %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/unit/named -poseidon-unit=u 2>&1 | FileCheck --check-prefix=UNIT-WROTE %s
; RUN: %clang -O0 -x ir -fno-discard-value-names %t/unit/host_other_cuid.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/unit/named -mllvm -poseidon-unit=u -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=UNIT-DISPATCH %s
; RUN: %clang -O0 -x ir %t/unit/host_cuid.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/raise_tcec -mllvm -poseidon-cache=%t/unit/named -S -emit-llvm -o - 2>&1 | FileCheck --check-prefix=LAUNCH --implicit-check-not=__poseidon_ozaki_dgemm --implicit-check-not='call {{.*}}@cudaLaunchKernel(' %s

; UNIT-WROTE: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; UNIT-CUID: "unit": "0123abcd"

; UNIT-DISPATCH: [ozaki-host-dispatch] replaced stub body _Z28__device_stub__matmul_kernelPKdS0_PdS1_ -> __poseidon_ozaki_dgemm (C=arg2 A=arg0 B=arg1 16x16x16){{$}}
; UNIT-DISPATCH: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(ptr noundef [[A:%[0-9]+]], ptr noundef [[B:%[0-9]+]], ptr noundef [[C:%[0-9]+]], ptr noundef %{{[0-9]+}})
; UNIT-DISPATCH-NEXT: entry:
; UNIT-DISPATCH-NEXT: call void @__poseidon_ozaki_dgemm(ptr [[C]], ptr [[A]], ptr [[B]], i32 16, i32 16, i32 16, i32 16, i32 0, i32 0, double 1.000000e+00, double 0.000000e+00, ptr null, i32 12)
; UNIT-DISPATCH-NEXT: ret void
; UNIT-DISPATCH-NEXT: }
; UNIT-DISPATCH-NOT: call {{.*}}@cudaLaunchKernel(

; The two kernels of the module below dispatch the product of matmul_body,
; matmul_kernel first in the module and alpha_kernel, with A and B swapped,
; second. Their descriptors are in the manifest, and their messages on stderr,
; in order of kernel name, whichever site is solved first: in the per-site
; solve, where alpha_kernel reuses the solve of matmul_kernel, and in the joint
; solve, which records both after the joint materialization. A repeated
; compile writes the same manifest and the same messages.
; RUN: mkdir %t/order
; RUN: %opt %s %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/order/c -poseidon-unit=u 2> %t/order/site1.err
; RUN: cp %t/order/c/launches_tcec_matmul.cu_*.json %t/order/site1.json
; RUN: %opt %s %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/order/c -poseidon-unit=u 2> %t/order/site2.err
; RUN: FileCheck --check-prefix=ORDER-LOG %s < %t/order/site1.err
; RUN: FileCheck --check-prefix=ORDER-MANIFEST -DNMA=12 -DNMM=12 %s < %t/order/site1.json
; RUN: grep 'wrote descriptor' %t/order/site1.err > %t/order/site1.wrote
; RUN: grep 'wrote descriptor' %t/order/site2.err > %t/order/site2.wrote
; RUN: diff %t/order/site1.wrote %t/order/site2.wrote
; RUN: cmp %t/order/site1.json %t/order/c/launches_tcec_matmul.cu_*.json
; RUN: %opt %s %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/order/c -poseidon-unit=u -poseidon-joint-dp 2> %t/order/joint1.err
; RUN: cp %t/order/c/launches_tcec_matmul.cu_*.json %t/order/joint1.json
; RUN: %opt %s %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/order/c -poseidon-unit=u -poseidon-joint-dp 2> %t/order/joint2.err
; RUN: FileCheck --check-prefix=ORDER-JOINT %s < %t/order/joint1.err
; RUN: FileCheck --check-prefix=ORDER-MANIFEST -DNMA=14 -DNMM=13 %s < %t/order/joint1.json
; RUN: grep 'wrote descriptor' %t/order/joint1.err > %t/order/joint1.wrote
; RUN: grep 'wrote descriptor' %t/order/joint2.err > %t/order/joint2.wrote
; RUN: diff %t/order/joint1.wrote %t/order/joint2.wrote
; RUN: cmp %t/order/joint1.json %t/order/c/launches_tcec_matmul.cu_*.json

; ORDER-LOG: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#7){{$}}
; ORDER-LOG: [poseidon] matmul_body: same body and target as an earlier site; reusing its solve (preprocess_matmul_body){{$}}
; ORDER-LOG: [ozaki-host-dispatch] wrote descriptor for _Z12alpha_kernelPKdS0_PdS1_: C=arg2 A=arg1 B=arg0 16x16x16 nm=12 (standalone)
; ORDER-LOG-NEXT: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=12 (standalone)

; ORDER-JOINT: JOINT: minimum accuracy cost within budget: {{[-+.e0-9]+}}; computation cost used: {{[-+.e0-9]+}}{{$}}
; ORDER-JOINT: [ozaki-host-dispatch] wrote descriptor for _Z12alpha_kernelPKdS0_PdS1_: C=arg2 A=arg1 B=arg0 16x16x16 nm=14 (standalone)
; ORDER-JOINT-NEXT: [ozaki-host-dispatch] wrote descriptor for _Z13matmul_kernelPKdS0_PdS1_: C=arg2 A=arg0 B=arg1 16x16x16 nm=13 (standalone)

; ORDER-MANIFEST:      "host dispatch": [
; ORDER-MANIFEST-NEXT:   "_Z12alpha_kernelPKdS0_PdS1_ 2 1 0 16 1 [[NMA]] 16 16 16 16 16 16 0 0 0 0",
; ORDER-MANIFEST-NEXT:   "_Z13matmul_kernelPKdS0_PdS1_ 2 0 1 16 1 [[NMM]] 16 16 16 16 16 16 0 0 0 0"
; ORDER-MANIFEST-NEXT: ],
; ORDER-MANIFEST-NEXT: "profile generation": []

; A device compile whose descriptors cannot be written stops with the path and
; the system's reason, for a dispatched product and for a profiled kernel: its
; device code relies on the host half rewriting the launch stubs. The cache
; path lies under a regular file. A compile that records no descriptor writes
; nothing and succeeds against the same path. Profile generation for an NVPTX
; kernel needs Enzyme and the NVPTX device profiler.
; RUN: %if enzyme && nvptx-device-profiler %{ mkdir %t/unwritable && echo > %t/unwritable/file %}
; RUN: %if enzyme && nvptx-device-profiler %{ not %opt %{dev} %{pass} -poseidon-raise-wmma %{flags} -poseidon-cache=%t/unwritable/file/c 2>&1 | FileCheck --check-prefix=UNWRITABLE-DISPATCH -DCACHE=%t/unwritable/file/c %s %}
; RUN: %if enzyme && nvptx-device-profiler %{ not %opt %S/../Inputs/stale_cache/matmul_annotated_device.ll %loadPoseidonEnzyme -passes="poseidon,enzyme,poseidon-finalize,globaldce" -enzyme-preopt=false -poseidon-profile-generate -poseidon-cache=%t/unwritable/file/c -poseidon-print -S -o /dev/null 2>&1 | FileCheck --check-prefix=UNWRITABLE-PROFILE -DCACHE=%t/unwritable/file/c %s %}
; RUN: %if enzyme && nvptx-device-profiler %{ %opt %{dev} %{pass} -poseidon-raise-wmma=false %{flags} -poseidon-cache=%t/unwritable/file/c 2>&1 | FileCheck --check-prefix=UNWRITABLE-NONE %s %}

; UNWRITABLE-DISPATCH: Applying solution for matmul[0] -> ozaki-ii nm=12 wmma m16n16k16 s8/s32 (#7){{$}}
; UNWRITABLE-DISPATCH-NOT: wrote descriptor
; UNWRITABLE-DISPATCH: error: Poseidon: cannot write the launch manifest [[CACHE]]/launches_tcec_matmul.cu_{{[0-9A-F]+}}.json: Not a directory. The manifest is kept in the -poseidon-cache directory, which has to be writable.{{$}}
; UNWRITABLE-DISPATCH-NOT: wrote descriptor

; UNWRITABLE-PROFILE-NOT: [poseidon-profgen] _Z13matmul_kernelPKdS0_PdS1_: site
; UNWRITABLE-PROFILE: error: Poseidon: cannot write the launch manifest [[CACHE]]/launches_tcec_matmul.cu_{{[0-9A-F]+}}.json: Not a directory. The manifest is kept in the -poseidon-cache directory, which has to be writable.{{$}}
; UNWRITABLE-PROFILE-NOT: [poseidon-profgen] _Z13matmul_kernelPKdS0_PdS1_: site

; UNWRITABLE-NONE: [poseidon] no rewrite applied for preprocess_matmul_body; calling original body matmul_body (bit-transparent no-op){{$}}
; UNWRITABLE-NONE-NOT: launch manifest

source_filename = "tcec_matmul.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@enzyme_const = dso_local addrspace(1) externally_initialized global i32 0, align 4
@enzyme_dup = dso_local addrspace(1) externally_initialized global i32 0, align 4
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr)], section "llvm.metadata"

define dso_local void @matmul_body(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) #0 {
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %6 = shl nuw nsw i32 %5, 4
  br label %11

7:                                                ; preds = %11
  %8 = add nuw nsw i32 %6, %4
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw [8 x i8], ptr %2, i64 %9
  store double %23, ptr %10, align 8
  ret void

11:                                               ; preds = %3, %11
  %12 = phi i32 [ 0, %3 ], [ %24, %11 ]
  %13 = phi double [ 0.000000e+00, %3 ], [ %23, %11 ]
  %14 = add nuw nsw i32 %12, %6
  %15 = zext nneg i32 %14 to i64
  %16 = getelementptr inbounds nuw [8 x i8], ptr %0, i64 %15
  %17 = load double, ptr %16, align 8
  %18 = shl nuw nsw i32 %12, 4
  %19 = add nuw nsw i32 %18, %4
  %20 = zext nneg i32 %19 to i64
  %21 = getelementptr inbounds nuw [8 x i8], ptr %1, i64 %20
  %22 = load double, ptr %21, align 8
  %23 = tail call double @llvm.fmuladd.f64(double %17, double %22, double %13)
  %24 = add nuw nsw i32 %12, 1
  %25 = icmp eq i32 %24, 16
  br i1 %25, label %7, label %11
}

declare double @llvm.fmuladd.f64(double, double, double) #1

define dso_local ptx_kernel void @_Z13matmul_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %5, ptr noundef %0, i32 noundef %5, ptr noundef %1, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

define dso_local ptx_kernel void @_Z12alpha_kernelPKdS0_PdS1_(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3) local_unnamed_addr #2 {
  %5 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_const to ptr), align 4
  %6 = load i32, ptr addrspacecast (ptr addrspace(1) @enzyme_dup to ptr), align 4
  tail call void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef nonnull @matmul_body, i32 noundef %5, ptr noundef %1, i32 noundef %5, ptr noundef %0, i32 noundef %6, ptr noundef %2, ptr noundef %3) #5
  ret void
}

declare dso_local void @_Z22__poseidon_fp_optimizeIvJiPKdiS1_iPdS2_EET_PvDpT0_(ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #3

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4

declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_120" "target-features"="+ptx88,+sm_120" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent nounwind "uniform-work-group-size" }

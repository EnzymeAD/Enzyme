; DEFINE: %{body} = sed -e 's/^@enzyme_dup = .*/&\n@hits = dso_local addrspace(1) global i32 0, align 4/' \
; DEFINE:   -e 's/^declare double @llvm.fmuladd.f64/declare void @record(ptr)\ndeclare i32 @pure(i32) nounwind willreturn memory(none)\n&/' \
; DEFINE:   %S/../../Inputs/stale_cache/matmul_device.ll
; DEFINE: %{opt} = %opt %loadPoseidon -passes="poseidon,poseidon-finalize" \
; DEFINE:   -poseidon-profile-use=%S/../../Inputs/matmul_site_profile -poseidon-early-prune \
; DEFINE:   -poseidon-enable-algebraic=false -poseidon-enable-precision=false -poseidon-enable-matmul \
; DEFINE:   -poseidon-cost-model=%cost_models/cm_sm_120_synthetic_library.csv -poseidon-tau=1e-12 \
; DEFINE:   -poseidon-compilation-id=tcec -S
; DEFINE: %{host} = %clang -O0 -x ir --target=x86_64-unknown-linux-gnu -fno-discard-value-names %t/host.ll \
; DEFINE:   %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../../Inputs/matmul_site_profile \
; DEFINE:   -mllvm -poseidon-early-prune -mllvm -poseidon-compilation-id=tcec -mllvm -poseidon-print -S -emit-llvm -o -
; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/ozaki_stub\.cu/tcec_matmul.cu/g' %S/../../Inputs/library_call/ozaki_stub_host.ll > %t/host.ll
; RUN: %{body} | sed '0,/^  ret void$/s//  %%hit = atomicrmw add ptr addrspace(1) @hits, i32 1 monotonic, align 4\n  ret void/' > %t/atomic.ll
; RUN: %{opt} %t/atomic.ll -poseidon-cache=%t/atomic -o - | FileCheck --check-prefixes=DEVICE,ATOMIC %s
; RUN: cat %t/atomic/launches_*.json | FileCheck --check-prefix=FUSED %s
; RUN: %{host} -mllvm -poseidon-cache=%t/atomic 2>&1 | FileCheck --check-prefix=HOST-FUSED %s
; RUN: %{body} | sed '0,/^  ret void$/s//  store i32 1, ptr addrspace(1) @hits, align 4\n  ret void/' > %t/global.ll
; RUN: %{opt} %t/global.ll -poseidon-cache=%t/global -o - | FileCheck --check-prefixes=DEVICE,GLOBAL %s
; RUN: cat %t/global/launches_*.json | FileCheck --check-prefix=FUSED %s
; RUN: %{body} | sed '0,/^  ret void$/s//  call void @record(ptr %%2)\n  ret void/' > %t/call.ll
; RUN: %{opt} %t/call.ll -poseidon-cache=%t/call -o - | FileCheck --check-prefixes=DEVICE,CALL %s
; RUN: cat %t/call/launches_*.json | FileCheck --check-prefix=FUSED %s
; RUN: %{body} | sed '0,/^  ret void$/s//  %%p = call i32 @pure(i32 %%5)\n  ret void/' > %t/pure.ll
; RUN: %{opt} %t/pure.ll -poseidon-cache=%t/pure -o /dev/null
; RUN: cat %t/pure/launches_*.json | FileCheck --check-prefix=STANDALONE %s
; RUN: %{host} -mllvm -poseidon-cache=%t/pure 2>&1 | FileCheck --check-prefix=HOST-STANDALONE %s
; RUN: %{body} | sed '0,/^  ret void$/s//  %%c = load i32, ptr %%2, align 4\n  %%hit = atomicrmw add ptr addrspace(1) @hits, i32 %%c monotonic, align 4\n  ret void/' > %t/reads.ll
; RUN: %{opt} %t/reads.ll -poseidon-cache=%t/reads -poseidon-print -o /dev/null 2>&1 | FileCheck --check-prefix=READS --implicit-check-not='wrote descriptor' %s

; The product of Inputs/stale_cache/matmul_device.ll, whose body also does
; other work after storing it: an atomic add, a store to a global, or a call
; that may write memory. A library call takes the place of a kernel's launch
; only when the product's store is the body's one effect, so each of these is
; fused: the call precedes the launch, which is kept, and the body keeps its
; other work and loses the product's store and the reduction loop, so C is
; written once, by the call. A call of a function that reads no memory,
; returns and does not throw is not an effect, and that body's launch is
; replaced. A fused body that reads C would read what the call wrote before
; the launch, so a body whose other work reads C gets no library call.

; FUSED: "_Z13matmul_kernelPKdS0_PdS1_ 2 0 1 0 12 0 0 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"
; STANDALONE: "_Z13matmul_kernelPKdS0_PdS1_ 2 0 1 1 12 0 0 0 2 rd 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0 1 16 0.000000e+00"

; DEVICE-LABEL: define dso_local void @preprocess_matmul_body(
; DEVICE-NEXT: {{^  }}
; ATOMIC-SAME: %hit = atomicrmw add ptr addrspace(1) @hits, i32 1 monotonic, align 4
; GLOBAL-SAME: store i32 1, ptr addrspace(1) @hits, align 4
; CALL-SAME: call void @record(ptr %2)
; DEVICE-NEXT: ret void
; DEVICE-NEXT: }

; HOST-FUSED: [poseidon] prepended the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_ozakiii_dgemm (C=arg2 A=arg0 B=arg1){{$}}
; HOST-FUSED-LABEL: define {{.*}} @_Z28__device_stub__matmul_kernelPKdS0_PdS1_(
; HOST-FUSED: call void @__poseidon_ozakiii_dgemm(
; HOST-FUSED: cudaLaunchKernel(ptr {{.*}}@_Z28__device_stub__matmul_kernel
; READS: [poseidon] preprocess_matmul_body: the body reads C, which the library call writes before the kernel that does the body's other work runs{{$}}
; HOST-STANDALONE: [poseidon] replaced the launch in stub _Z28__device_stub__matmul_kernelPKdS0_PdS1_ with __poseidon_ozakiii_dgemm (C=arg2 A=arg0 B=arg1){{$}}

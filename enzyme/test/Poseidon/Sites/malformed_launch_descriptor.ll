; DEFINE: %{clang} = %clang -O0 -x ir -mllvm -poseidon-cache=%t -mllvm -poseidon-unit=fixture -S -emit-llvm -o /dev/null %loadClangPoseidon
; RUN: rm -rf %t && mkdir -p %t
; RUN: cp %S/../Inputs/hoststub_cache/launches_*.json %S/../Inputs/lambda_stub/launches_*.json %t/
; RUN: sed -i 's/ 0 0 0 2 rd [^"]*"/ 0 0 0"/' %t/launches_hoststub.cu_*.json
; RUN: not %{clang} %s -mllvm -poseidon-profile-use=%S/../Inputs/hoststub_cache 2>&1 | FileCheck --check-prefix=SHORT -DCACHE=%t %s
; RUN: cp %S/../Inputs/hoststub_cache/launches_*.json %t/
; RUN: sed -i 's/2 0 1 1 3 0 0 0 1 rd/2 0 1 1 99 0 0 0 1 rd/' %t/launches_hoststub.cu_*.json
; RUN: not %{clang} %s -mllvm -poseidon-profile-use=%S/../Inputs/hoststub_cache 2>&1 | FileCheck --check-prefix=MODULI -DCACHE=%t %s
; RUN: sed 's/hoststub\.cu/lambda_stub.cu/' %s > %t/lambda.ll
; RUN: sed -i 's/ off 16 40 64"/ off 16 40"/' %t/launches_lambda_stub.cu_*.json
; RUN: not %{clang} %t/lambda.ll -mllvm -poseidon-profile-generate 2>&1 | FileCheck --check-prefix=OFFSETS -DCACHE=%t %s
; RUN: sed -i 's/"_Z11gemm_kernelPKdS0_Pd [^"]*"/7/' %t/launches_hoststub.cu_*.json
; RUN: not %{clang} %s -mllvm -poseidon-profile-use=%S/../Inputs/hoststub_cache 2>&1 | FileCheck --check-prefix=NOTSTRING -DCACHE=%t %s

; A descriptor of the launch manifest the device compile wrote into the cache
; that does not parse stops the host compile, naming the manifest and the
; descriptor.

source_filename = "hoststub.cu"
target triple = "x86_64-unknown-linux-gnu"

define void @f() {
  ret void
}

; SHORT: {{^}}error: Poseidon: the launch manifest [[CACHE]]/launches_hoststub.cu_{{[0-9A-F]+}}.json holds the host dispatch descriptor '_Z11gemm_kernelPKdS0_Pd 2 0 1 1 1 0 0 0', which does not parse: it ends before its scheme. The device compile of this unit writes it into the -poseidon-cache directory; remove that directory and compile the unit again.{{$}}
; MODULI: {{^}}error: Poseidon: the launch manifest [[CACHE]]/launches_hoststub.cu_{{[0-9A-F]+}}.json holds the host dispatch descriptor '_Z16gemm_tall_kernelPKdS0_Pdi 2 0 1 1 99 0 0 0 1 rd 0 1 128 0 1 512 0 1 256 0 1 256 0 1 256 0 1 512 0.000000e+00', which does not parse: its modulus field 99 is above 14. The device compile of this unit writes it into the -poseidon-cache directory; remove that directory and compile the unit again.{{$}}
; OFFSETS: {{^}}error: Poseidon: the launch manifest [[CACHE]]/launches_lambda_stub.cu_{{[0-9A-F]+}}.json holds the profile generation descriptor '_ZL10CuKernel2DIZ6reduceiiiPdS0_S0_EUliE_EviT___intern__0123abcd 0 2 3 1 1 1 0 0 8 off 16 40', which does not parse: it ends before its pointer offset. The device compile of this unit writes it into the -poseidon-cache directory; remove that directory and compile the unit again.{{$}}
; NOTSTRING: {{^}}error: Poseidon: the launch manifest [[CACHE]]/launches_hoststub.cu_{{[0-9A-F]+}}.json holds a host dispatch entry that is not a string; remove the -poseidon-cache directory and compile the unit again{{$}}

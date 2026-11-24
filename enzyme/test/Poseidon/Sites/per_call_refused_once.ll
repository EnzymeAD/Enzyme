; RUN: rm -rf %t && mkdir %t
; RUN: sed 's/^target triple = .*/target triple = "nvptx64-nvidia-cuda"/' %s > %t/nvptx.ll
; RUN: sed 's/^target triple = .*/target triple = "amdgcn-amd-amdhsa"/' %s > %t/amdgpu.ll
; RUN: not %clang -O0 -x ir -Wno-override-module %s %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/host -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: not %clang -O0 -x ir -Wno-override-module %t/nvptx.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/nvptx -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: not %clang -O0 -x ir -Wno-override-module %t/amdgpu.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/amdgpu -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: %clang -O0 -x ir -Wno-override-module %s %loadClangPoseidon -S -emit-llvm -o %t/off.ll 2>&1 | FileCheck --check-prefix=OFF --implicit-check-not=warning: %s
; RUN: FileCheck --check-prefix=OFFIR %s < %t/off.ll

; What Sites/per_call_refused.ll leaves out: a refused call in device code and
; an identity with more than one use are reported once per compile too, and a
; compile through clang without a profile action, which lowers these calls at
; the start of the pipeline and again before the optimizer, warns once and
; emits each call of the identity as a call of what the identity was given.

; ERR: error: <unknown>:0:0: in function indirect double (ptr, double): Poseidon: __poseidon_optimize needs a function name and a call of it
; ERR: error: <unknown>:0:0: in function other_signature double (double): Poseidon: sq is called with another signature, so this call cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function two_calls double (double): Poseidon: __poseidon_optimize needs a function name and a call of it
; ERR: 3 errors generated.

; OFF: warning: <unknown>:0:0: in function indirect double (ptr, double): Poseidon: __poseidon_optimize needs a function name and a call of it; the call runs as written
; OFF: warning: <unknown>:0:0: in function other_signature double (double): Poseidon: sq is called with another signature, so this call is not a Poseidon site. The call runs as written.
; OFF: warning: <unknown>:0:0: in function two_calls double (double): Poseidon: __poseidon_optimize needs a function name and a call of it; the call runs as written
; OFF: 3 warnings generated.

target triple = "x86_64-unknown-linux-gnu"

define double @sq(double %x) {
  %m = fmul double %x, %x
  ret double %m
}

; OFFIR-LABEL: define double @indirect(ptr %fp, double %x)
; OFFIR-NEXT: %r = call double %fp(double %x)
; OFFIR-NEXT: ret double %r
define double @indirect(ptr %fp, double %x) {
  %h = call ptr @__poseidon_site(ptr %fp)
  %r = call double %h(double %x)
  ret double %r
}

; OFFIR-LABEL: define double @other_signature(double %x)
; OFFIR-NEXT: %r = call double @sq(float 1.000000e+00)
; OFFIR-NEXT: ret double %r
define double @other_signature(double %x) {
  %h = call ptr @__poseidon_site(ptr @sq)
  %r = call double %h(float 1.0)
  ret double %r
}

; OFFIR-LABEL: define double @two_calls(double %x)
; OFFIR-NEXT: %a = call double @sq(double %x)
; OFFIR-NEXT: %b = call double @sq(double %a)
; OFFIR-NEXT: ret double %b
define double @two_calls(double %x) {
  %h = call ptr @__poseidon_site(ptr @sq)
  %a = call double %h(double %x)
  %b = call double %h(double %a)
  ret double %b
}

declare ptr @__poseidon_site(ptr)

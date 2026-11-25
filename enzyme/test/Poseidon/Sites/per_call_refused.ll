; RUN: rm -rf %t && split-file %s %t
; RUN: sed 's/^target triple = .*/target triple = "nvptx64-nvidia-cuda"/' %t/host.ll > %t/nvptx.ll
; RUN: sed 's/^target triple = .*/target triple = "amdgcn-amd-amdhsa"/' %t/host.ll > %t/amdgpu.ll
; RUN: not %clang -O0 -x ir -Wno-override-module %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/c -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: not %clang -O0 -x ir -Wno-override-module %t/nvptx.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/nvptx -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: not %clang -O0 -x ir -Wno-override-module %t/amdgpu.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/amdgpu -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s
; RUN: %if enzyme %{ not %clang -O0 -x ir -Wno-override-module %t/host.ll %loadClangPoseidonEnzyme -mllvm -poseidon-profile-generate -mllvm -poseidon-cache=%t/g -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR --implicit-check-not=error: %s %}
; RUN: %clang -O0 -x ir -Wno-override-module %t/host.ll %loadClangPoseidon -S -emit-llvm -o %t/off.ll 2>&1 | FileCheck --check-prefix=OFF --implicit-check-not=warning: %s
; RUN: FileCheck --check-prefix=OFFIR %s < %t/off.ll
; RUN: not %opt %t/invoke_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/other_sites -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=INVOKE --implicit-check-not=error: %s
; A call through __poseidon_optimize that cannot be a site is an error in a
; compile that generates or uses a profile, and the call runs as written with
; a warning in any other: the identity's argument is no function, or the
; function is variadic, has no body here or is called with another signature;
; so is an identity with more than one use. In device code the copy made for
; the call is reached through an invoke. The opt of LLVM 21 and 22 exits at its
; first error, so the compiles with several refused calls run through clang,
; which reports every error and warning of a compile once, for a host or a
; device triple, although it lowers these calls at the start of the pipeline
; and again before the optimizer.

; INVOKE: error: <unknown>:0:0: in function k void (ptr): Poseidon: vol at call 0 in k is reached through an invoke, so this call cannot be a Poseidon site

;--- host.ll
target triple = "x86_64-unknown-linux-gnu"

define double @variadic(double %x, ...) {
  %m = fmul double %x, %x
  ret double %m
}

declare double @elsewhere(double)

define double @sq(double %x) {
  %m = fmul double %x, %x
  ret double %m
}

; ERR: error: <unknown>:0:0: in function indirect double (ptr, double): Poseidon: __poseidon_optimize needs a function name and a call of it
; OFF: warning: <unknown>:0:0: in function indirect double (ptr, double): Poseidon: __poseidon_optimize needs a function name and a call of it; the call runs as written
; Without a profile action every such call goes through no identity: it calls
; what the identity was given.
; OFFIR-LABEL: define double @indirect(ptr %fp, double %x)
; OFFIR-NEXT: %r = call double %fp(double %x)
; OFFIR-NEXT: ret double %r
; OFFIR-LABEL: define double @refused(double %x)
; OFFIR-NEXT: %r0 = call double (double, ...) @variadic(double %x)
; OFFIR-NEXT: %r1 = call double @elsewhere(double %r0)
; OFFIR-NEXT: %r2 = call double @sq(float 1.000000e+00)
; OFFIR-NEXT: %s = fadd double %r1, %r2
; OFFIR-NEXT: ret double %s
define double @indirect(ptr %fp, double %x) {
  %h = call ptr @__poseidon_site(ptr %fp)
  %r = call double %h(double %x)
  ret double %r
}

; ERR: error: <unknown>:0:0: in function refused double (double): Poseidon: variadic is variadic, so this call cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function refused double (double): Poseidon: elsewhere has no body in this translation unit, so this call cannot be a Poseidon site
; ERR: error: <unknown>:0:0: in function refused double (double): Poseidon: sq is called with another signature, so this call cannot be a Poseidon site
; OFF: warning: <unknown>:0:0: in function refused double (double): Poseidon: variadic is variadic, so this call is not a Poseidon site and runs as written
; OFF: warning: <unknown>:0:0: in function refused double (double): Poseidon: elsewhere has no body in this translation unit, so this call is not a Poseidon site and runs as written
; OFF: warning: <unknown>:0:0: in function refused double (double): Poseidon: sq is called with another signature, so this call is not a Poseidon site and runs as written
define double @refused(double %x) {
  %h0 = call ptr @__poseidon_site(ptr @variadic)
  %r0 = call double (double, ...) %h0(double %x)
  %h1 = call ptr @__poseidon_site(ptr @elsewhere)
  %r1 = call double %h1(double %r0)
  %h2 = call ptr @__poseidon_site(ptr @sq)
  %r2 = call double %h2(float 1.0)
  %s = fadd double %r1, %r2
  ret double %s
}

; ERR: error: <unknown>:0:0: in function two_calls double (double): Poseidon: __poseidon_optimize needs a function name and a call of it
; ERR: 5 errors generated.
; OFF: warning: <unknown>:0:0: in function two_calls double (double): Poseidon: __poseidon_optimize needs a function name and a call of it; the call runs as written
; OFF: 5 warnings generated.
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

;--- invoke_nvptx.ll
target triple = "nvptx64-nvidia-cuda"

define internal void @vol.poseidon.k.0(ptr %v) #0 {
  store double 1.0, ptr %v
  ret void
}

define ptx_kernel void @k(ptr %v) personality ptr @personality {
  invoke void @vol.poseidon.k.0(ptr %v)
          to label %done unwind label %cleanup
done:
  ret void
cleanup:
  %l = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l
}

declare i32 @personality(...)

attributes #0 = { "poseidon-site-body"="" }

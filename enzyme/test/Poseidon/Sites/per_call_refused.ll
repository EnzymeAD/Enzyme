; RUN: rm -rf %t && split-file %s %t
; RUN: not %clang -O0 -x ir -Wno-override-module %t/host.ll %loadClangPoseidon -mllvm -poseidon-profile-use=%S/../Inputs/other_sites -mllvm -poseidon-cache=%t/c -S -emit-llvm -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s
; RUN: %opt %t/host.ll %loadPoseidon -passes=poseidon -disable-output 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: not %opt %t/invoke_nvptx.ll %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/other_sites -poseidon-cache=%t/c -disable-output 2>&1 | FileCheck --check-prefix=INVOKE %s

; A call through __poseidon_optimize that cannot be a site is an error in a
; compile that generates or uses a profile, and the call runs as written with
; a warning in any other: the identity's argument is no function, or the
; function is variadic, has no body here or is called with another signature.
; In device code the copy made for the call is reached through an invoke.
; The opt of LLVM 21 and 22 exits at its first error, so the compile with
; several refused calls runs through clang, which reports every error of a
; compile.

; INVOKE: error: {{.*}}in function k {{.*}}Poseidon: vol at call 0 in k is reached through an invoke, so this call cannot be a Poseidon site

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

; ERR: error: {{.*}}in function indirect {{.*}}Poseidon: __poseidon_optimize needs a function name and a call of it
; OFF: warning: {{.*}}in function indirect {{.*}}Poseidon: __poseidon_optimize needs a function name and a call of it; the call runs as written
define double @indirect(ptr %fp, double %x) {
  %h = call ptr @__poseidon_site(ptr %fp)
  %r = call double %h(double %x)
  ret double %r
}

; ERR: error: {{.*}}in function refused {{.*}}Poseidon: variadic is variadic, so this call cannot be a Poseidon site
; ERR: error: {{.*}}in function refused {{.*}}Poseidon: elsewhere has no body in this translation unit, so this call cannot be a Poseidon site
; ERR: error: {{.*}}in function refused {{.*}}Poseidon: sq is called with another signature, so this call cannot be a Poseidon site
; OFF: warning: {{.*}}in function refused {{.*}}Poseidon: variadic is variadic, so this call is not a Poseidon site. The call runs as written.
; OFF: warning: {{.*}}in function refused {{.*}}Poseidon: elsewhere has no body in this translation unit, so this call is not a Poseidon site. The call runs as written.
; OFF: warning: {{.*}}in function refused {{.*}}Poseidon: sq is called with another signature, so this call is not a Poseidon site. The call runs as written.
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

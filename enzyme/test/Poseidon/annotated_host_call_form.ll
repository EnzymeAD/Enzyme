; RUN: rm -rf %t
; RUN: not %opt %s %loadPoseidon -passes="poseidon,poseidon-finalize" -poseidon-profile-use=%t/none -poseidon-cache=%t -S -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: poseidon

; A function that carries POSEIDON_OPTIMIZE is a site at every call, so naming
; it in __poseidon_fp_optimize as well is a compile error at that call.

target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"host.cpp\00", section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @tester, ptr @.str, ptr @.str.1, i32 3, ptr null }], section "llvm.metadata"

define double @tester(double %x, double %y) {
  %m = fmul double %x, %y
  %a = fadd double %m, %x
  ret double %a
}

define double @site(double %x, double %y) {
  %r = call double (ptr, ...) @__poseidon_fp_optimize(ptr @tester, double %x, double %y)
  ret double %r
}

declare double @__poseidon_fp_optimize(ptr, ...)

; CHECK: error: {{.*}}in function site {{.*}}Poseidon: tester carries POSEIDON_OPTIMIZE, so every call of it is already a site; remove this __poseidon_fp_optimize or the annotation

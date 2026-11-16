; RUN: rm -rf %t && mkdir %t
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-generate -poseidon-cache=%t/gen -disable-output 2>&1 | FileCheck --check-prefix=ERR %s
; RUN: not %opt %s %loadPoseidon -passes=poseidon -poseidon-profile-use=%S/../Inputs/other_sites -poseidon-cache=%t/use -disable-output 2>&1 | FileCheck --check-prefix=ERR %s
; RUN: %opt %s %loadPoseidon -passes=poseidon -disable-output 2>&1 | FileCheck --check-prefix=OFF %s

; A host function carrying POSEIDON_OPTIMIZE whose signature has no activity
; for every parameter and its value (annotated_host_function.ll) cannot be a
; site: in a compile that generates or uses a profile that is an error, every
; such function reported. Without either the function compiles as written.

; ERR: error: {{.*}}Poseidon: pair returns a value that is not a floating-point or integer scalar, so it cannot be a Poseidon site
; ERR: error: {{.*}}Poseidon: vec returns a value that is not a floating-point or integer scalar, so it cannot be a Poseidon site
; ERR: error: {{.*}}Poseidon: viasret returns its value through memory (sret), so it cannot be a Poseidon site
; ERR: error: {{.*}}Poseidon: variadic is variadic, so it cannot be a Poseidon site
; ERR: error: {{.*}}Poseidon: recur calls itself, so it cannot be a Poseidon site
; OFF-COUNT-5: warning: {{.*}}compiled without -poseidon-profile-generate or -poseidon-profile-use=<dir>
; OFF-NOT: error

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"poseidon\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"host.cpp\00", section "llvm.metadata"
@llvm.global.annotations = appending global [5 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @pair, ptr @.str, ptr @.str.1, i32 15, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @vec, ptr @.str, ptr @.str.1, i32 20, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @viasret, ptr @.str, ptr @.str.1, i32 24, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @variadic, ptr @.str, ptr @.str.1, i32 28, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @recur, ptr @.str, ptr @.str.1, i32 32, ptr null }], section "llvm.metadata"

define { double, double } @pair(double %x) {
  %m = fmul double %x, %x
  %r = insertvalue { double, double } undef, double %m, 0
  ret { double, double } %r
}

define <2 x double> @vec(<2 x double> %x) {
  %m = fmul <2 x double> %x, %x
  ret <2 x double> %m
}

define void @viasret(ptr sret({ double, double }) %r, double %x) {
  %m = fmul double %x, %x
  store double %m, ptr %r
  ret void
}

define double @variadic(double %x, ...) {
  %m = fmul double %x, %x
  ret double %m
}

define double @recur(double %x) {
  %m = fmul double %x, %x
  %c = fcmp olt double %m, 1.000000e+00
  br i1 %c, label %done, label %again
again:
  %h = fmul double %x, 5.000000e-01
  %r = call double @recur(double %h)
  ret double %r
done:
  ret double %m
}
